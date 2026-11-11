"""Deterministic SQL guard. The LLM's SQL is untrusted input.

Pipeline: parse -> single read-only statement -> table allowlist -> blocked-column check
-> rewrite every base table into a scoped, PII-safe subquery -> enforce LIMIT.

Scoping is done by rewriting the AST, not by asking the model to behave, so a prompt
injection cannot widen what a user can see.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .access import UserAccess

BASE_TABLES = {"orders", "order_items", "products", "users"}

# Direct identifiers and precise location: never queryable.
BLOCKED_COLUMNS = {
    "first_name", "last_name", "email", "street_address", "postal_code",
    "latitude", "longitude", "user_geom",
}
# The only users columns the scoped view exposes (so even SELECT * is safe).
SAFE_USER_COLUMNS = ["id", "age", "gender", "state", "city", "country", "traffic_source", "created_at"]


class SQLGuardError(ValueError):
    """Rejected SQL. The message is safe to feed back to the model for self-correction."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class GuardResult:
    sql: str
    tables: set[str]


def _lit(values) -> str:
    return ", ".join(exp.Literal.string(v).sql(dialect="bigquery") for v in values)


def _product_ids_sql(access: UserAccess, ds: str) -> str:
    where = ""
    if not access.unrestricted:
        conds = []
        if access.brands:
            conds.append(f"p.brand IN ({_lit(access.brands)})")
        if access.departments:
            conds.append(f"p.department IN ({_lit(access.departments)})")
        where = " WHERE " + " OR ".join(conds)
    return f"SELECT p.id FROM `{ds}.products` AS p{where}"


def _view_sql(table: str, access: UserAccess, ds: str) -> str:
    ids = _product_ids_sql(access, ds)
    unrestricted = access.unrestricted
    items_of_scope = f"SELECT oi.{{col}} FROM `{ds}.order_items` AS oi" + (
        "" if unrestricted else f" WHERE oi.product_id IN ({ids})"
    )
    if table == "products":
        return ids.replace("SELECT p.id", "SELECT p.*", 1)
    if table == "order_items":
        return f"SELECT oi.* FROM `{ds}.order_items` AS oi" + (
            "" if unrestricted else f" WHERE oi.product_id IN ({ids})"
        )
    if table == "orders":
        return f"SELECT o.* FROM `{ds}.orders` AS o" + (
            "" if unrestricted else f" WHERE o.order_id IN ({items_of_scope.format(col='order_id')})"
        )
    if table == "users":
        cols = ", ".join(f"u.{c}" for c in SAFE_USER_COLUMNS)
        return f"SELECT {cols} FROM `{ds}.users` AS u" + (
            "" if unrestricted else f" WHERE u.id IN ({items_of_scope.format(col='user_id')})"
        )
    raise SQLGuardError("table_not_allowed", f"Table '{table}' is not available.")


def _check_qualifier(node: exp.Table, dataset: str) -> None:
    catalog, db = node.catalog, node.db
    if not catalog and not db:
        return
    project, _, ds_name = dataset.rpartition(".")
    ok = (db == ds_name) and (not catalog or catalog == project)
    if not ok:
        raise SQLGuardError("dataset_not_allowed", "Only the thelook_ecommerce dataset can be queried.")


def _apply_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    if isinstance(tree, exp.Select):
        lim = tree.args.get("limit")
        if lim is not None:
            e = lim.expression
            if isinstance(e, exp.Literal) and e.is_int and int(e.this) <= max_rows:
                return tree
        return tree.limit(max_rows)
    # UNION etc.: wrap so the cap applies to the combined result
    return exp.select("*").from_(tree.subquery("q")).limit(max_rows)


def guard_sql(sql: str, access: UserAccess, *, dataset: str, max_rows: int = 1000) -> GuardResult:
    if not access.has_scope:
        raise SQLGuardError("no_scope", "No data access is configured for this user.")

    try:
        statements = sqlglot.parse(sql, dialect="bigquery")
    except sqlglot.errors.SqlglotError as e:
        raise SQLGuardError("parse_error", f"SQL could not be parsed: {str(e)[:200]}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLGuardError("multi_statement", "Exactly one SQL statement is allowed.")

    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SQLGuardError("not_select", "Only read-only SELECT queries are allowed.")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    if cte_names & BASE_TABLES:
        raise SQLGuardError("cte_shadows_table", "A CTE cannot reuse a base table name; rename it.")

    for col in tree.find_all(exp.Column):
        if col.name.lower() in BLOCKED_COLUMNS:
            raise SQLGuardError("blocked_column", f"Column '{col.name}' is restricted and cannot be queried.")

    used: set[str] = set()
    to_replace: list[tuple[exp.Table, str]] = []
    for node in tree.find_all(exp.Table):
        if not isinstance(node.this, exp.Identifier):
            raise SQLGuardError("table_not_allowed", "Table functions are not allowed.")
        name = node.name.lower()
        if not node.db and not node.catalog and name in cte_names:
            continue  # reference to a CTE, already validated at its definition
        _check_qualifier(node, dataset)
        if name not in BASE_TABLES:
            raise SQLGuardError("table_not_allowed", f"Table '{node.name}' is not available.")
        to_replace.append((node, name))
        used.add(name)

    # Collect first, then replace, so the inserted subqueries are never re-processed.
    for node, name in to_replace:
        view = sqlglot.parse_one(_view_sql(name, access, dataset), dialect="bigquery")
        alias = node.alias or node.name
        node.replace(exp.Subquery(this=view, alias=exp.TableAlias(this=exp.to_identifier(alias))))

    tree = _apply_limit(tree, max_rows)
    return GuardResult(sql=tree.sql(dialect="bigquery"), tables=used)

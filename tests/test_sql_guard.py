import pytest
import sqlglot
from sqlglot import exp

from src.access import UnknownUserError, UserAccess, load_access
from src.sql_guard import SQLGuardError, guard_sql

D = "bigquery-public-data.thelook_ecommerce"
NIKE = UserAccess("u1", brands=("Nike",))
ADMIN = UserAccess("admin", unrestricted=True)


def g(sql, access=NIKE, **kw):
    return guard_sql(sql, access, dataset=D, **kw)


def rejected(sql, code, access=NIKE):
    with pytest.raises(SQLGuardError) as e:
        g(sql, access)
    assert e.value.code == code


def test_select_is_scoped_and_limited():
    r = g("SELECT brand, COUNT(*) c FROM products GROUP BY brand")
    assert "'Nike'" in r.sql and "LIMIT 1000" in r.sql and r.tables == {"products"}


def test_fully_qualified_table_is_scoped():
    r = g("SELECT COUNT(*) FROM `bigquery-public-data.thelook_ecommerce.order_items`")
    assert "product_id IN" in r.sql


def test_join_of_all_tables_reparses():
    r = g(
        "SELECT p.brand, COUNT(DISTINCT u.id) FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.id = oi.product_id "
        "JOIN users u ON u.id = o.user_id GROUP BY 1"
    )
    assert r.tables == {"orders", "order_items", "products", "users"}
    sqlglot.parse_one(r.sql, dialect="bigquery")


@pytest.mark.parametrize("sql", [
    "DELETE FROM users WHERE 1=1",
    "DROP TABLE users",
    "INSERT INTO products (id) VALUES (1)",
    "UPDATE products SET cost = 0 WHERE TRUE",
])
def test_non_select_rejected(sql):
    rejected(sql, "not_select")


def test_multi_statement_rejected():
    rejected("SELECT 1; SELECT 2", "multi_statement")


def test_garbage_rejected():
    with pytest.raises(SQLGuardError):
        g("SELEC FROM WHERE ((")


def test_unknown_table_rejected():
    rejected("SELECT * FROM `bigquery-public-data.thelook_ecommerce.inventory_items`", "table_not_allowed")


def test_other_dataset_rejected():
    rejected("SELECT * FROM `other-project.other_ds.users`", "dataset_not_allowed")


def test_information_schema_rejected():
    with pytest.raises(SQLGuardError):
        g("SELECT * FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.TABLES`")


@pytest.mark.parametrize("sql", [
    "SELECT email FROM users",
    "SELECT u.last_name FROM users u",
    "SELECT latitude, longitude FROM users",
])
def test_blocked_columns_rejected(sql):
    rejected(sql, "blocked_column")


def test_select_star_on_users_cannot_leak_pii():
    sql = g("SELECT * FROM users").sql.lower()
    for col in ("email", "first_name", "last_name", "street_address", "postal_code", "latitude", "user_geom"):
        assert col not in sql


def test_cte_named_like_base_table_rejected():
    rejected("WITH orders AS (SELECT 1 AS x) SELECT * FROM orders", "cte_shadows_table")


def test_normal_cte_allowed_and_not_replaced():
    r = g("WITH t AS (SELECT product_id, SUM(sale_price) s FROM order_items GROUP BY 1) SELECT * FROM t")
    assert r.tables == {"order_items"} and "WITH t AS" in r.sql


def test_limit_is_capped_but_small_limits_kept():
    assert "LIMIT 1000" in g("SELECT id FROM products LIMIT 999999").sql
    assert "LIMIT 10" in g("SELECT id FROM products LIMIT 10").sql


def test_union_is_wrapped_with_limit():
    r = g("SELECT id FROM products UNION ALL SELECT id FROM products")
    assert r.sql.rstrip().endswith("LIMIT 1000")


def test_no_scope_fails_closed():
    rejected("SELECT 1 FROM products", "no_scope", access=UserAccess("nobody"))


def test_brand_or_department_scope():
    r = g("SELECT id FROM products", UserAccess("m", brands=("Nike",), departments=("Women",)))
    assert "'Nike'" in r.sql and "'Women'" in r.sql and " OR " in r.sql


def test_unrestricted_has_no_row_filter_but_users_stay_pii_safe():
    r = g("SELECT * FROM products", ADMIN)
    assert "brand IN" not in r.sql
    assert "email" not in g("SELECT * FROM users", ADMIN).sql.lower()


def test_scope_values_cannot_inject_sql():
    evil = "Nike' OR 1=1 --"
    r = g("SELECT id FROM products", UserAccess("e", brands=(evil,)))
    tree = sqlglot.parse_one(r.sql, dialect="bigquery")
    assert any(l.is_string and l.this == evil for l in tree.find_all(exp.Literal))


def test_load_access(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("users:\n  a:\n    brands: [Nike]\n  b:\n    unrestricted: true\n")
    assert load_access("a", p).brands == ("Nike",)
    assert load_access("b", p).unrestricted
    with pytest.raises(UnknownUserError):
        load_access("ghost", p)

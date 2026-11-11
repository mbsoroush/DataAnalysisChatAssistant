"""Saved-reports library with a two-phase, confirmation-gated delete.

Design rules (requirement 3, "High-Stakes Oversight"):
  * Delete is a SOFT delete (deleted_at) with an audit-log row, so a mistaken confirmation is recoverable
    with "undo" for UNDO_WINDOW_S. A production job would purge soft-deleted rows after a retention period.
  * Ownership is enforced in SQL: every read and delete carries `owner = ?`. A user can never see or
    delete another user's report, whatever the LLM or the chat text says.
  * Delete is two-phase. Phase 1 resolves the request to a concrete list of report IDs and stores it as a
    pending action (per user + conversation, 5 min TTL). Nothing is deleted. Phase 2 deletes exactly
    those stored IDs. The search is never re-run, so nothing extra can be removed.
  * The confirmation is parsed deterministically ("yes" / "no"), never by the LLM, so a prompt injection
    cannot confirm a deletion, and confirming still works while the LLM is down.
  * Any other message discards the pending action. A new delete request replaces it.
  * Report bodies are never sent to the LLM in this flow; only the user's message is.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import prompts

PENDING_TTL_S = 300
UNDO_WINDOW_S = 3600
MAX_LISTED = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at REAL NOT NULL,
    deleted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports(owner, conversation_id);
CREATE TABLE IF NOT EXISTS pending_deletes (
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    report_ids TEXT NOT NULL,
    description TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    action TEXT NOT NULL,
    report_ids TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class Report:
    id: int
    owner: str
    conversation_id: str
    title: str
    body: str
    created_at: float


@dataclass(frozen=True)
class PendingDelete:
    user_id: str
    conversation_id: str
    report_ids: tuple[int, ...]
    description: str
    expires_at: float


def clean_title(text: str, limit: int = 80) -> str:
    """Titles come from user text: strip control characters and newlines, collapse spaces, truncate."""
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[: limit - 1] + "…") if len(t) > limit else (t or "Untitled report")


def _like(term: str) -> str:
    return "%" + term.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class ReportStore:
    def __init__(self, path: str | Path = ":memory:", clock: Callable[[], float] = time.time):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._clock = clock
        self._migrate()

    def _migrate(self) -> None:
        """Databases created before soft delete existed get the new column in place."""
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(reports)")}
        if "deleted_at" not in cols:
            with self._db:
                self._db.execute("ALTER TABLE reports ADD COLUMN deleted_at REAL")

    # ------------------------------------------------------------ reports
    @staticmethod
    def _row(r: sqlite3.Row) -> Report:
        return Report(r["id"], r["owner"], r["conversation_id"], r["title"], r["body"], r["created_at"])

    def save(self, owner: str, conversation_id: str, title: str, body: str) -> Report:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO reports (owner, conversation_id, title, body, created_at) VALUES (?,?,?,?,?)",
                (owner, conversation_id, clean_title(title), body, self._clock()))
        return self.get(owner, cur.lastrowid)  # type: ignore[return-value]

    def get(self, owner: str, report_id: int) -> Optional[Report]:
        r = self._db.execute("SELECT * FROM reports WHERE owner = ? AND id = ? AND deleted_at IS NULL", (owner, report_id)).fetchone()
        return self._row(r) if r else None

    def list(self, owner: str, *, conversation_id: Optional[str] = None,
             mention: Optional[str] = None) -> list[Report]:
        sql, args = "SELECT * FROM reports WHERE owner = ? AND deleted_at IS NULL", [owner]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            args.append(conversation_id)
        if mention:
            sql += " AND (lower(title) LIKE ? ESCAPE '\\' OR lower(body) LIKE ? ESCAPE '\\')"
            args += [_like(mention), _like(mention)]
        rows = self._db.execute(sql + " ORDER BY id", args).fetchall()
        return [self._row(r) for r in rows]

    # ------------------------------------------------------------ two-phase delete
    def begin_delete(self, user_id: str, conversation_id: str, reports: list[Report],
                     description: str) -> PendingDelete:
        """Phase 1: store the exact IDs to delete. Re-checks ownership; replaces any earlier pending action."""
        ids = tuple(r.id for r in reports if r.owner == user_id)
        if not ids:
            raise ValueError("nothing to delete")
        exp = self._clock() + PENDING_TTL_S
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO pending_deletes VALUES (?,?,?,?,?)",
                (user_id, conversation_id, json.dumps(ids), description, exp))
        return PendingDelete(user_id, conversation_id, ids, description, exp)

    def get_pending(self, user_id: str, conversation_id: str) -> Optional[PendingDelete]:
        """Returns the pending action; expired ones are reported as such by confirm_delete, not hidden."""
        r = self._db.execute("SELECT * FROM pending_deletes WHERE user_id = ? AND conversation_id = ?",
                             (user_id, conversation_id)).fetchone()
        if not r:
            return None
        return PendingDelete(r["user_id"], r["conversation_id"], tuple(json.loads(r["report_ids"])),
                             r["description"], r["expires_at"])

    def clear_pending(self, user_id: str, conversation_id: str) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM pending_deletes WHERE user_id = ? AND conversation_id = ?",
                                   (user_id, conversation_id))
        return cur.rowcount > 0

    def confirm_delete(self, user_id: str, conversation_id: str) -> tuple[str, int]:
        """Phase 2. Returns (status, deleted_count); status is 'deleted', 'expired' or 'none'.
        Atomic: the pending row is consumed first, so a double "yes" cannot delete twice."""
        with self._db:
            p = self.get_pending(user_id, conversation_id)
            if p is None:
                return "none", 0
            self._db.execute("DELETE FROM pending_deletes WHERE user_id = ? AND conversation_id = ?",
                             (user_id, conversation_id))
            if self._clock() > p.expires_at:
                return "expired", 0
            marks = ",".join("?" * len(p.report_ids))
            now = self._clock()
            cur = self._db.execute(
                f"UPDATE reports SET deleted_at = ? WHERE owner = ? AND deleted_at IS NULL AND id IN ({marks})",
                (now, user_id, *p.report_ids))
            if cur.rowcount:
                self._db.execute(
                    "INSERT INTO audit_log (ts, user_id, conversation_id, action, report_ids) VALUES (?,?,?,?,?)",
                    (now, user_id, conversation_id, "delete", json.dumps(p.report_ids)))
        return "deleted", cur.rowcount

    def undo_last_delete(self, user_id: str) -> tuple[str, int]:
        """Restore the reports of this user's most recent un-undone deletion, within UNDO_WINDOW_S.
        Returns (status, restored_count); status is 'restored', 'expired' or 'none'."""
        with self._db:
            r = self._db.execute(
                "SELECT * FROM audit_log WHERE user_id = ? AND action = 'delete' AND undone = 0 "
                "ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
            if r is None:
                return "none", 0
            if self._clock() > r["ts"] + UNDO_WINDOW_S:
                return "expired", 0
            ids = json.loads(r["report_ids"])
            marks = ",".join("?" * len(ids))
            cur = self._db.execute(
                f"UPDATE reports SET deleted_at = NULL WHERE owner = ? AND deleted_at IS NOT NULL AND id IN ({marks})",
                (user_id, *ids))
            self._db.execute("UPDATE audit_log SET undone = 1 WHERE id = ?", (r["id"],))
            self._db.execute(
                "INSERT INTO audit_log (ts, user_id, conversation_id, action, report_ids, undone) VALUES (?,?,?,?,?,1)",
                (self._clock(), user_id, r["conversation_id"], "undo", json.dumps(ids)))
        return "restored", cur.rowcount


# ---------------------------------------------------------------------- confirmation parsing
_YES = {"yes", "y", "yep", "yeah", "confirm", "confirmed", "go ahead", "do it", "proceed", "sure", "ok",
        "okay", "yes delete", "yes delete them", "yes please", "yes confirm", "delete them", "yes do it"}
_NO = {"no", "n", "nope", "cancel", "stop", "abort", "never mind", "nevermind", "dont", "dont delete",
       "no cancel", "no dont", "keep them"}


def classify_reply(text: str) -> Optional[str]:
    """'yes' / 'no' only when the whole message is a plain confirmation or cancellation, else None."""
    t = re.sub(r"[^\w\s]", "", text.lower().replace("'", ""))
    t = re.sub(r"\s+", " ", t).strip()
    if t in _YES:
        return "yes"
    if t in _NO:
        return "no"
    return None


# ---------------------------------------------------------------------- the manager
MSG_HELP = ("I can list your saved reports, show one, or delete reports (for example \"delete all reports "
            "mentioning Client X\" or \"delete the reports from this conversation\"). Reports are saved "
            "automatically when you ask me to write one.")
MSG_CANCELLED = "Okay, nothing was deleted."
MSG_NOTHING_PENDING = "There is nothing waiting for confirmation."
MSG_EXPIRED = "That confirmation expired, so nothing was deleted. Ask me again if you still want to delete."
MSG_NO_MATCH = "I couldn't find any saved reports of yours matching that, so nothing was deleted."
MSG_NOTHING_TO_UNDO = "There is no recent deletion to undo."
MSG_UNDO_EXPIRED = "The undo window for your last deletion has passed, so those reports can't be restored here."
MSG_NEED_MENTION = "Which client, product or keyword should the reports mention?"


def _plural(n: int) -> str:
    return f"{n} report" + ("" if n == 1 else "s")


def _fmt(r: Report) -> str:
    return f"#{r.id} {r.title} ({datetime.fromtimestamp(r.created_at):%Y-%m-%d %H:%M})"


def _listing(reports: list[Report]) -> str:
    lines = [f"- {_fmt(r)}" for r in reports[:MAX_LISTED]]
    if len(reports) > MAX_LISTED:
        lines.append(f"- ...and {len(reports) - MAX_LISTED} more")
    return "\n".join(lines)


class ReportManager:
    """Everything 'manage_reports' does. `llm` only turns the user's sentence into a structured command;
    it never sees report contents and never confirms anything."""

    def __init__(self, store: ReportStore, llm: Any, *, scrub: Callable[[str], str] = lambda s: s,
                 event: Callable[..., None] = lambda *a, **k: None):
        self.store, self.llm, self.scrub, self.event = store, llm, scrub, event

    def pending_reply(self, user_id: str, conversation_id: str, text: str) -> Optional[str]:
        """'yes'/'no' when a deletion is awaiting confirmation and the message is a plain reply, else None."""
        if self.store.get_pending(user_id, conversation_id) is None:
            return None
        return classify_reply(text)

    def discard_pending(self, user_id: str, conversation_id: str) -> None:
        if self.store.clear_pending(user_id, conversation_id):
            self.event("delete_discarded", reason="other_message")

    # ------------------------------------------------------------------
    def handle(self, user_id: str, conversation_id: str, question: str, history: list,
               confirm: Optional[str] = None) -> str:
        if confirm == "yes":
            return self._confirm(user_id, conversation_id)
        if confirm == "no":
            self.store.clear_pending(user_id, conversation_id)
            self.event("delete_cancelled")
            return MSG_CANCELLED

        res = self.llm.generate(prompts.report_command_prompt(history, question),
                                system=prompts.REPORT_COMMAND_SYSTEM, schema=prompts.ReportCommand,
                                max_output_tokens=150)
        cmd = res.parsed
        if cmd.action == "list":
            return self._list(user_id, conversation_id, cmd)
        if cmd.action == "show":
            return self._show(user_id, cmd)
        if cmd.action == "delete":
            return self._propose_delete(user_id, conversation_id, cmd)
        if cmd.action == "undo":
            return self._undo(user_id)
        return MSG_HELP if not self.store.get_pending(user_id, conversation_id) else MSG_NOTHING_PENDING

    # ------------------------------------------------------------------
    def _resolve(self, user_id: str, conversation_id: str, cmd) -> tuple[list[Report], str]:
        if cmd.scope == "conversation":
            return self.store.list(user_id, conversation_id=conversation_id), "from this conversation"
        if cmd.scope == "mention" and cmd.mention.strip():
            m = cmd.mention.strip()
            return self.store.list(user_id, mention=m), f'mentioning "{m}"'
        if cmd.scope == "id" and cmd.report_id:
            r = self.store.get(user_id, cmd.report_id)
            return ([r] if r else []), f"#{cmd.report_id}"
        if cmd.scope == "all":
            return self.store.list(user_id), "(all of your saved reports)"
        return [], ""

    def _list(self, user_id: str, conversation_id: str, cmd) -> str:
        reports, what = self._resolve(user_id, conversation_id, cmd) if cmd.scope != "none" else (
            self.store.list(user_id), "")
        if not reports:
            return "You have no saved reports" + (f" {what}." if what else ".")
        return f"{_plural(len(reports))} {what}:".replace("  ", " ") + "\n" + _listing(reports)

    def _show(self, user_id: str, cmd) -> str:
        r = self.store.get(user_id, cmd.report_id) if cmd.report_id else None
        if r is None:
            return "I couldn't find that report among your saved reports."
        return f"**{r.title}**\n\n{self.scrub(r.body)}"

    def _propose_delete(self, user_id: str, conversation_id: str, cmd) -> str:
        if cmd.scope == "mention" and not cmd.mention.strip():
            return MSG_NEED_MENTION
        reports, what = self._resolve(user_id, conversation_id, cmd)
        if not reports:
            self.store.clear_pending(user_id, conversation_id)
            self.event("delete_no_match", scope=cmd.scope)
            return MSG_NO_MATCH
        self.store.begin_delete(user_id, conversation_id, reports, what)
        self.event("delete_proposed", scope=cmd.scope, count=len(reports))
        return (f"I found {_plural(len(reports))} {what}:\n{_listing(reports)}\n\n"
                f"This can't be undone. Reply **yes** to delete "
                f"{'it' if len(reports) == 1 else 'them'}, or **no** to cancel.")

    def _confirm(self, user_id: str, conversation_id: str) -> str:
        status, n = self.store.confirm_delete(user_id, conversation_id)
        self.event("delete_confirmed", status=status, count=n)
        if status == "expired":
            return MSG_EXPIRED
        if status == "none":
            return MSG_NOTHING_PENDING
        return (f"Deleted {_plural(n)}. Say \"undo\" within the hour to restore "
                f"{'it' if n == 1 else 'them'}.")

    def _undo(self, user_id: str) -> str:
        status, n = self.store.undo_last_delete(user_id)
        self.event("delete_undone", status=status, count=n)
        if status == "expired":
            return MSG_UNDO_EXPIRED
        if status == "none" or n == 0:
            return MSG_NOTHING_TO_UNDO
        return f"Restored {_plural(n)}."

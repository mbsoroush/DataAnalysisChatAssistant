"""PII masking applied at the data boundary: results are masked right after BigQuery returns,
so unmasked values never enter agent state, prompts, logs or output.

Layers (defense in depth):
  1. schema shown to the LLM omits personal columns          (semantic_layer.yaml)
  2. SQL guard rejects personal columns and scopes users      (sql_guard.py)
  3. mask_frame: drop blocked columns, pseudonymise ids, drop re-identifying combos, suppress small
     age/city groups, scrub string cells
  4. scan_text: final output scanner on the LLM's answer
"""
from __future__ import annotations

import hashlib
import re
from typing import Tuple

import pandas as pd

from .sql_guard import BLOCKED_COLUMNS

ID_COLUMNS = {"user_id", "customer_id", "customer", "buyer_id", "users_id"}
QUASI_COLUMNS = {"age", "city"}
MIN_GROUP_SIZE = 5  # age/city rows describing fewer customers than this are suppressed
COUNT_COLUMNS = {"customers", "customer_count", "num_customers", "n_customers", "unique_customers",
                 "users", "user_count", "num_users", "n_users", "count", "cnt", "n"}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\w)")
REDACTED = "[redacted]"


def pseudonym(value, salt: str) -> str:
    return "cust_" + hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:8]


def scan_text(text: str) -> Tuple[str, list[str]]:
    """Redact emails and phone numbers from free text. Returns (clean_text, findings)."""
    findings: list[str] = []
    if EMAIL_RE.search(text):
        findings.append("email")
        text = EMAIL_RE.sub(REDACTED, text)
    if PHONE_RE.search(text):
        findings.append("phone")
        text = PHONE_RE.sub(REDACTED, text)
    return text, findings


def _scrub_cell(v):
    return scan_text(v)[0] if isinstance(v, str) else v


def mask_frame(df: pd.DataFrame, salt: str, min_group: int = MIN_GROUP_SIZE) -> Tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()

    drop = [c for c in out.columns if str(c).lower() in BLOCKED_COLUMNS]
    if drop:
        out = out.drop(columns=drop)
        notes.append(f"dropped blocked columns: {drop}")

    id_cols = [c for c in out.columns if str(c).lower() in ID_COLUMNS]
    for c in id_cols:
        out[c] = out[c].map(lambda v: None if pd.isna(v) else pseudonym(v, salt))

    if id_cols:  # id + age/city on the same row could re-identify a person
        quasi = [c for c in out.columns if str(c).lower() in QUASI_COLUMNS]
        if quasi:
            out = out.drop(columns=quasi)
            notes.append(f"dropped quasi-identifiers next to customer ids: {quasi}")

    quasi_left = [c for c in out.columns if str(c).lower() in QUASI_COLUMNS]
    if quasi_left:  # aggregates by age/city: a group of 1-4 customers can point at a person
        count_cols = [c for c in out.columns if str(c).lower() in COUNT_COLUMNS]
        if count_cols:
            small = pd.Series(False, index=out.index)
            for c in count_cols:
                small |= pd.to_numeric(out[c], errors="coerce") < min_group
            if small.any():
                out = out[~small].reset_index(drop=True)
                notes.append(f"suppressed {int(small.sum())} group(s) smaller than {min_group} by {quasi_left}")

    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(_scrub_cell)
    return out, notes

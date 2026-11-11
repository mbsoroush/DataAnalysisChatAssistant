from pathlib import Path

import pandas as pd
import yaml

from src.masking import mask_frame, scan_text
from src.sql_guard import BLOCKED_COLUMNS


def test_blocked_columns_dropped():
    out, notes = mask_frame(pd.DataFrame({"email": ["a@b.com"], "state": ["CA"]}), "s")
    assert list(out.columns) == ["state"] and notes


def test_ids_pseudonymised_deterministically_and_salted():
    df = pd.DataFrame({"customer_id": [1, 1, 2]})
    a, _ = mask_frame(df, "salt1")
    b, _ = mask_frame(df, "salt2")
    assert a["customer_id"][0] == a["customer_id"][1] != a["customer_id"][2]
    assert a["customer_id"][0].startswith("cust_") and a["customer_id"][0] != b["customer_id"][0]


def test_quasi_identifiers_dropped_next_to_ids():
    out, _ = mask_frame(pd.DataFrame({"customer_id": [1], "age": [30], "city": ["X"], "state": ["CA"]}), "s")
    assert set(out.columns) == {"customer_id", "state"}


def test_age_city_kept_in_aggregates():
    out, _ = mask_frame(pd.DataFrame({"city": ["X"], "avg_spend": [10.0]}), "s")
    assert "city" in out.columns


def test_string_cells_scrubbed():
    out, _ = mask_frame(pd.DataFrame({"note": ["mail me at a@b.com"]}), "s")
    assert "@" not in out["note"][0]


def test_scan_text_redacts_emails_and_phones_but_not_numbers_or_dates():
    t, f = scan_text("a@b.com, (415) 555 0134, revenue $1,234,567.89 on 2025-01-15")
    assert set(f) == {"email", "phone"} and "1,234,567.89" in t and "2025-01-15" in t


def test_semantic_layer_never_exposes_personal_columns():
    sem = yaml.safe_load((Path(__file__).parent.parent / "config" / "semantic_layer.yaml").read_text())
    for table, spec in sem["tables"].items():
        assert not (set(spec["columns"]) & BLOCKED_COLUMNS), table

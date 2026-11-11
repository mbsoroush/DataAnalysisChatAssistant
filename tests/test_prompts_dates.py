"""The model must never do calendar arithmetic: period boundaries are computed in code."""
from datetime import date

from src import prompts


def test_ranges_for_mid_november():
    txt = prompts.date_context(date(2026, 11, 11))
    assert "today is 2026-11-11" in txt
    assert "last completed quarter: [2026-07-01, 2026-10-01)" in txt
    assert "current quarter so far: [2026-10-01, 2026-11-12)" in txt
    assert "last completed month: [2026-10-01, 2026-11-01)" in txt
    assert "last 6 completed months: [2026-05-01, 2026-11-01)" in txt
    assert "year to date: [2026-01-01, 2026-11-12)" in txt
    assert "last calendar year: [2025-01-01, 2026-01-01)" in txt


def test_ranges_cross_the_year_boundary():
    txt = prompts.date_context(date(2026, 1, 15))
    assert "last completed quarter: [2025-10-01, 2026-01-01)" in txt
    assert "last completed month: [2025-12-01, 2026-01-01)" in txt
    assert "last 6 completed months: [2025-07-01, 2026-01-01)" in txt


def test_first_day_of_a_quarter():
    txt = prompts.date_context(date(2026, 4, 1))
    assert "last completed quarter: [2026-01-01, 2026-04-01)" in txt
    assert "current quarter so far: [2026-04-01, 2026-04-02)" in txt


def test_end_of_year_tomorrow_rolls_over():
    assert "year to date: [2026-01-01, 2027-01-01)" in prompts.date_context(date(2026, 12, 31))


def test_plan_system_includes_the_date_reference():
    system = prompts.plan_system("SCHEMA-TEXT")
    assert "DATE REFERENCE - today is" in system
    assert system.index("DATE REFERENCE") < system.index("SCHEMA-TEXT")

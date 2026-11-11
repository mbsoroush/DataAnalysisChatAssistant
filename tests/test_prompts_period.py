from src import prompts


def test_analyst_prompt_reports_last_included_day():
    text = prompts.analyst_system("Plain business English.", True, {})
    assert "last INCLUDED day" in text
    assert "never the exclusive end date" in text

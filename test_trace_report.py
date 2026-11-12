import json

from src import trace_report as tr


def ev(trace, name, ts, **kw):
    return {"ts": ts, "trace_id": trace, "event": name, **kw}


def sample():
    return [
        ev("a", "turn_start", 100.0, question="monthly revenue"),
        ev("a", "intent", 100.1, intent="analysis"),
        ev("a", "plan", 100.2, status="ok", ms=3000, attempt=1),
        ev("a", "guard_reject", 100.3, code="blocked_column", query=1),
        ev("a", "retry_needed", 100.4, failure="sql_error", attempt=1),
        ev("a", "llm_call", 100.5, model="m-strong", attempts=1, ms=2000, input_tokens=100, output_tokens=40,
           thought_tokens=10, fallback_used=False, slow=False),
        ev("a", "analyze", 103.0, status="ok", ms=2500),
        ev("a", "turn_end", 103.1, status="ok", intent="analysis", attempts=2, ms=3100),
        ev("b", "turn_start", 200.0, question="customer emails please"),
        ev("b", "blocked_request", 200.1, reason="malicious_or_pii"),
        ev("b", "turn_end", 200.2, status="refused", intent="malicious", attempts=0, ms=1800),
        ev("c", "turn_start", 300.0, question="crashed turn"),
        ev("c", "llm_rate_limited", 300.5, model="gemini-pro-latest", cooldown_s=328),
        ev("c", "llm_error", 300.4, model="gemini-pro-latest", attempt=1, code=429, error="ClientError", ms=300),
        ev("c", "llm_call", 301.0, model="fallback", attempts=2, ms=12000, input_tokens=5, output_tokens=5,
           thought_tokens=0, fallback_used=True, slow=True),
    ]


def test_summary_counts_statuses_retries_and_llm_health():
    s = tr.summarize(sample())
    assert "3 started, 2 finished, 1 without an end event" in s
    assert "ok=1, refused=1" in s
    assert "1/1 analysis turns needed >1 attempt; 1 of those ended ok" in s
    assert "blocked_column=1" in s and "sql_error=1" in s
    assert "blocked requests: 1" in s
    assert "fallback used: 1/2; slow (>10 s): 1" in s
    assert "ClientError(429)=1" in s and "gemini-pro-latest=1" in s
    assert "105 in / 45 out" in s


def test_failures_lists_only_non_ok_turns_with_their_question():
    out = tr.failures(sample())
    assert "b" in out and "refused" in out and "customer emails" in out and "monthly revenue" not in out


def test_timeline_is_ordered_with_relative_offsets():
    out = tr.timeline(sample(), "a").splitlines()
    assert out[0] == "trace a" and "turn_start" in out[1] and "turn_end" in out[-1]
    assert "+      0 ms" in out[1]


def test_unknown_trace_and_empty_failures_are_handled():
    assert "No events" in tr.timeline(sample(), "zzz")
    assert tr.failures([ev("a", "turn_end", 1, status="ok")]) == "No failed turns."


def test_load_skips_corrupt_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(ev("a", "turn_start", 1)) + "\n{not json\n\n" + json.dumps(ev("a", "turn_end", 2, status="ok")) + "\n")
    assert len(tr.load(p)) == 2


def test_percentile_nearest_rank():
    assert tr.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert tr.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert tr.percentile([], 50) is None


def test_main_reports_missing_file(tmp_path, capsys):
    assert tr.main(["--file", str(tmp_path / "nope.jsonl")]) == 1
    assert "No trace file" in capsys.readouterr().out

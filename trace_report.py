"""Turn logs/traces.jsonl into agent-level metrics and per-turn timelines.

    python -m src.trace_report                    # health summary over the whole file
    python -m src.trace_report --failures         # list turns that did not end in status ok
    python -m src.trace_report --trace 0c5416e01abb   # step-by-step timeline of one turn

The prototype writes one JSON event per line (see src/obs.py). This module is the "deep dive" tool:
the same aggregations are what dashboards and alerts would be built on in production.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

STEP_EVENTS = ("route", "plan", "bq_query", "analyze", "manage_reports")


def load(path: str | Path) -> list[dict]:
    recs: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written last line must not break the report
    return recs


def percentile(values: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def _fmt_counter(c: Counter) -> str:
    return ", ".join(f"{k}={v}" for k, v in c.most_common()) or "none"


def _ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"p50 {percentile(values, 50):.0f} / p95 {percentile(values, 95):.0f} / max {max(values):.0f} ms (n={len(values)})"


def summarize(recs: Iterable[dict]) -> str:
    recs = list(recs)
    by = defaultdict(list)
    for r in recs:
        by[r.get("event")].append(r)

    ends = by["turn_end"]
    started = {r["trace_id"] for r in by["turn_start"]}
    finished = {r["trace_id"] for r in ends}
    lines = ["== Turns"]
    lines.append(f"turns: {len(started)} started, {len(ends)} finished, {len(started - finished)} without an end event")
    lines.append(f"status: {_fmt_counter(Counter(r.get('status') for r in ends))}")
    lines.append(f"intent: {_fmt_counter(Counter(r.get('intent') or 'none' for r in ends))}")
    lines.append(f"latency: {_ms([r['ms'] for r in ends if 'ms' in r])}")
    multi = [r for r in ends if (r.get("attempts") or 0) > 1]
    analysed = [r for r in ends if (r.get("attempts") or 0) >= 1]
    ok_after_retry = [r for r in multi if r.get("status") == "ok"]
    if analysed:
        lines.append(f"self-correction: {len(multi)}/{len(analysed)} analysis turns needed >1 attempt; "
                     f"{len(ok_after_retry)} of those ended ok")

    lines.append("\n== Steps (successful spans)")
    for name in STEP_EVENTS:
        spans = by[name]
        ok = [r["ms"] for r in spans if r.get("status") == "ok" and "ms" in r]
        err = sum(1 for r in spans if r.get("status") == "error")
        if spans:
            lines.append(f"{name:15} {_ms(ok)}" + (f", {err} error(s)" if err else ""))

    lines.append("\n== LLM")
    calls = by["llm_call"]
    lines.append(f"calls: {len(calls)} by model: {_fmt_counter(Counter(r.get('model') for r in calls))}")
    if calls:
        fb = sum(1 for r in calls if r.get("fallback_used"))
        slow = sum(1 for r in calls if r.get("slow"))
        lines.append(f"fallback used: {fb}/{len(calls)}; slow (>10 s): {slow}")
        lines.append(f"tokens: {sum(r.get('input_tokens') or 0 for r in calls)} in / "
                     f"{sum(r.get('output_tokens') or 0 for r in calls)} out "
                     f"(+{sum(r.get('thought_tokens') or 0 for r in calls)} thinking)")
        lines.append(f"call latency: {_ms([r['ms'] for r in calls if 'ms' in r])}")
    errs = by["llm_error"]
    lines.append(f"errors: {_fmt_counter(Counter(f"{r.get('error')}({r.get('code')})" for r in errs))}")
    lines.append(f"rate limited: {_fmt_counter(Counter(r.get('model') for r in by['llm_rate_limited']))}; "
                 f"circuit open skips: {len(by['llm_circuit_open'])}; deadline hits: {len(by['llm_deadline'])}; "
                 f"turns lost to LLM outage: {len(by['llm_unavailable'])}")

    lines.append("\n== Data path")
    lines.append(f"guard rejections: {_fmt_counter(Counter(r.get('code') for r in by['guard_reject']))}")
    lines.append(f"retries needed: {_fmt_counter(Counter(r.get('failure') for r in by['retry_needed']))}; "
                 f"invalid plan output: {len(by['plan_invalid_output'])}")
    lines.append(f"masking applied: {len(by['masked'])} result sets")

    lines.append("\n== Safety")
    lines.append(f"blocked requests: {len(by['blocked_request'])}; "
                 f"PII redacted from model output: {len(by['pii_output_redacted'])} (should stay near 0)")

    lines.append("\n== Reports and deletes")
    lines.append(f"reports saved: {len(by['report_saved'])}, save failures: {len(by['report_save_failed'])}")
    lines.append(f"delete funnel: proposed={len(by['delete_proposed'])}, "
                 f"confirmed={_fmt_counter(Counter(r.get('status') for r in by['delete_confirmed']))}, "
                 f"cancelled={len(by['delete_cancelled'])}, discarded={len(by['delete_discarded'])}, "
                 f"no match={len(by['delete_no_match'])}, "
                 f"undone={_fmt_counter(Counter(r.get('status') for r in by['delete_undone']))}")
    return "\n".join(lines)


def failures(recs: Iterable[dict]) -> str:
    recs = list(recs)
    questions = {r["trace_id"]: r.get("question", "") for r in recs if r.get("event") == "turn_start"}
    rows = [r for r in recs if r.get("event") == "turn_end" and r.get("status") != "ok"]
    if not rows:
        return "No failed turns."
    out = ["trace_id      status       intent           ms  question"]
    for r in rows:
        q = questions.get(r["trace_id"], "")[:60]
        out.append(f"{r['trace_id']:13} {str(r.get('status')):12} {str(r.get('intent') or '-'):14} "
                   f"{r.get('ms', 0):>6}  {q}")
    return "\n".join(out)


def timeline(recs: Iterable[dict], trace_id: str) -> str:
    rows = sorted((r for r in recs if r.get("trace_id") == trace_id), key=lambda r: r.get("ts", 0))
    if not rows:
        return f"No events for trace {trace_id}."
    t0 = rows[0].get("ts", 0)
    out = [f"trace {trace_id}"]
    for r in rows:
        extra = {k: v for k, v in r.items() if k not in ("ts", "trace_id", "event")}
        out.append(f"+{(r.get('ts', t0) - t0) * 1000:7.0f} ms  {r.get('event'):20} "
                   + " ".join(f"{k}={v}" for k, v in extra.items()))
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise agent traces")
    ap.add_argument("--file", default="logs/traces.jsonl")
    ap.add_argument("--trace", help="show the timeline of one trace id")
    ap.add_argument("--failures", action="store_true", help="list turns that did not end in status ok")
    args = ap.parse_args(argv)
    if not Path(args.file).exists():
        print(f"No trace file at {args.file}. Run the chat first.")
        return 1
    recs = load(args.file)
    print(timeline(recs, args.trace) if args.trace else failures(recs) if args.failures else summarize(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal structured tracing: one JSON line per event, keyed by trace_id (one id per user turn).

`logs/traces.jsonl` lets you reconstruct a whole turn (route -> plan -> guard -> execute -> analyze)
with latencies and error types. The production design swaps this sink for OpenTelemetry/Langfuse
without touching call sites.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class Tracer:
    def __init__(self, path: Optional[str] = "logs/traces.jsonl"):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("agent.trace")

    def event(self, trace_id: str, name: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "trace_id": trace_id, "event": name, **fields}
        line = json.dumps(rec, default=str)
        self.log.info(line)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @contextmanager
    def span(self, trace_id: str, name: str, **fields):
        t0 = time.monotonic()
        try:
            yield
        except Exception as e:  # noqa: BLE001
            self.event(trace_id, name, status="error", error=type(e).__name__,
                       ms=int((time.monotonic() - t0) * 1000), **fields)
            raise
        else:
            self.event(trace_id, name, status="ok", ms=int((time.monotonic() - t0) * 1000), **fields)

"""Resilience tests for src/llm.py. No network, no API key: LLMClient._call is replaced with a scripted fake."""
import time
from types import SimpleNamespace

from pydantic import BaseModel

from src import llm
from src.llm import CircuitBreaker, LLMClient, LLMPermanentError, LLMUnavailableError


class Answer(BaseModel):
    x: int


class FakeAPIError(Exception):
    def __init__(self, code, msg="boom"):
        super().__init__(msg)
        self.code = code


def _resp(text, parsed=None):
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3)
    return SimpleNamespace(text=text, parsed=parsed, usage_metadata=usage)


def _client(script, models=("A", "B"), **kw):
    """script(model, call_no) returns a response or raises. Returns (client, events, calls)."""
    events, calls = [], []

    def fake_call(model, prompt, system, schema, temperature, max_output_tokens):
        calls.append(model)
        return script(model, len(calls))

    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("breaker", CircuitBreaker())
    c = LLMClient(None, list(models), on_event=lambda n, d: events.append((n, d)), **kw)
    c._call = fake_call
    return c, events, calls


def test_invalid_structured_output_falls_over_to_next_model():
    def script(model, n):
        return _resp("not json at all") if model == "A" else _resp('{"x": 7}')

    c, events, calls = _client(script, max_retries=1)
    res = c.generate("q", schema=Answer)
    assert res.model == "B" and res.parsed.x == 7 and res.fallback_used is True
    assert calls == ["A", "A", "B"]  # A retried once, then fell over
    assert [d["error"] for n, d in events if n == "llm_error"] == ["_EmptyResponse", "_EmptyResponse"]


def test_deadline_stops_retry_loop():
    def script(model, n):
        raise FakeAPIError(503)

    c, events, calls = _client(script, max_retries=10, deadline_s=0.05, sleep=lambda s: time.sleep(0.06))
    try:
        c.generate("q")
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError as exc:
        assert "deadline" in str(exc)
    assert len(calls) < 11
    assert any(n == "llm_deadline" for n, _ in events)


def test_success_after_failover_emits_llm_call():
    def script(model, n):
        if model == "A":
            raise FakeAPIError(503)
        return _resp("hello")

    c, events, _ = _client(script, max_retries=0)
    res = c.generate("q")
    assert res.model == "B" and res.attempts == 2 and res.fallback_used
    calls = [d for n, d in events if n == "llm_call"]
    assert len(calls) == 1
    assert calls[0]["model"] == "B" and calls[0]["attempts"] == 2 and calls[0]["fallback_used"] is True
    assert calls[0]["input_tokens"] == 10 and calls[0]["output_tokens"] == 5 and calls[0]["thought_tokens"] == 3
    assert calls[0]["slow"] is False


def test_rate_limit_fails_over_without_sleeping():
    slept = []

    def script(model, n):
        if model == "A":
            raise FakeAPIError(429, "quota retryDelay: 30s")
        return _resp("ok")

    c, events, calls = _client(script, sleep=slept.append)
    res = c.generate("q")
    assert res.model == "B" and calls == ["A", "B"] and slept == []
    limited = [d for n, d in events if n == "llm_rate_limited"]
    assert limited and limited[0]["model"] == "A" and limited[0]["cooldown_s"] == 30.0
    assert c.breaker.allow("A") is False  # parked for the server-suggested delay


def test_permanent_error_is_not_retried():
    def script(model, n):
        raise FakeAPIError(400, "bad request")

    c, _, calls = _client(script)
    try:
        c.generate("q")
        assert False, "expected LLMPermanentError"
    except LLMPermanentError:
        pass
    assert calls == ["A"]


def test_all_models_failing_raises_unavailable():
    def script(model, n):
        raise FakeAPIError(500)

    c, _, calls = _client(script, max_retries=1)
    try:
        c.generate("q")
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError as exc:
        assert exc.attempts == 4 and calls == ["A", "A", "B", "B"]


def test_broken_event_hook_never_breaks_a_call():
    c = LLMClient(None, ["A"], on_event=lambda n, d: 1 / 0, sleep=lambda s: None, breaker=CircuitBreaker())
    c._call = lambda *a, **k: _resp("fine")
    assert c.generate("q").text == "fine"


def test_global_event_hook_is_used_when_no_client_hook():
    seen = []
    llm.set_event_hook(lambda n, d: seen.append(n))
    try:
        c = LLMClient(None, ["A"], sleep=lambda s: None, breaker=CircuitBreaker())
        c._call = lambda *a, **k: _resp("fine")
        c.generate("q")
    finally:
        llm.set_event_hook(None)
    assert seen == ["llm_call"]

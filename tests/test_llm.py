import pytest

from src.llm import CircuitBreaker, LLMClient, LLMPermanentError, LLMUnavailableError


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"fake {code}")
        self.code = code


class FakeUsage:
    prompt_token_count = 10
    candidates_token_count = 5


class FakeResp:
    def __init__(self, text="ok"):
        self.text = text
        self.usage_metadata = FakeUsage()
        self.parsed = None


class FakeModels:
    """Script per-model behaviour: a list of outcomes (exception or response), consumed in order."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []
        self.configs = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        self.configs.append(config)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def make(script, models, **kw):
    client = FakeClient(script)
    llm = LLMClient(client, models, sleep=lambda s: None, **kw)
    return llm, client


def test_retries_transient_then_succeeds():
    llm, client = make({"a": [FakeAPIError(503), FakeAPIError(503), FakeResp("pong")]}, ["a"])
    res = llm.generate("hi")
    assert res.text == "pong" and res.model == "a" and res.attempts == 3 and not res.fallback_used


def test_falls_back_to_next_model_when_retries_exhausted():
    llm, client = make(
        {"a": [FakeAPIError(503)] * 3, "b": [FakeResp("from b")]},
        ["a", "b"],
    )
    res = llm.generate("hi")
    assert res.model == "b" and res.fallback_used and client.models.calls == ["a", "a", "a", "b"]


def test_permanent_error_is_not_retried():
    llm, client = make({"a": [FakeAPIError(400)], "b": [FakeResp()]}, ["a", "b"])
    with pytest.raises(LLMPermanentError):
        llm.generate("hi")
    assert client.models.calls == ["a"]


def test_unknown_model_is_skipped_without_retry():
    llm, client = make({"a": [FakeAPIError(404)], "b": [FakeResp("b ok")]}, ["a", "b"])
    assert llm.generate("hi").model == "b"
    assert client.models.calls == ["a", "b"]


def test_empty_response_is_retried():
    llm, _ = make({"a": [FakeResp(""), FakeResp("real")]}, ["a"])
    assert llm.generate("hi").text == "real"


def test_all_models_down_raises_unavailable():
    llm, _ = make({"a": [FakeAPIError(503)] * 3, "b": [FakeAPIError(429)] * 3}, ["a", "b"])
    with pytest.raises(LLMUnavailableError) as e:
        llm.generate("hi")
    assert e.value.attempts == 4  # 3 tries on a (503), 1 on b (429 fails over immediately, no retries)


def test_circuit_breaker_skips_failing_model():
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    llm, client = make(
        {"a": [FakeAPIError(503)] * 10, "b": [FakeResp("b1"), FakeResp("b2")]},
        ["a", "b"],
        breaker=breaker,
    )
    llm.generate("first")   # a fails twice -> circuit opens -> falls to b
    client.models.calls.clear()
    llm.generate("second")  # a is skipped entirely
    assert client.models.calls == ["b"]
    now[0] = 31             # cooldown over: a gets a probe call again
    client.models.script["a"] = [FakeResp("a back")]
    assert llm.generate("third").model == "a"


def test_rate_limit_fails_over_immediately_and_parks_the_model():
    err = FakeAPIError(429)
    err.details = {"error": {"details": [{"retryDelay": "40s"}]}}
    llm, client = make({"a": [err], "b": [FakeResp("b1"), FakeResp("b2")]}, ["a", "b"])
    assert llm.generate("x").model == "b"
    assert client.models.calls == ["a", "b"]          # no retries on the exhausted model
    client.models.calls.clear()
    assert llm.generate("y").model == "b"
    assert client.models.calls == ["b"]               # parked model is skipped, no wasted latency


def test_retry_delay_parsing():
    e = FakeAPIError(429)
    e.details = {"error": {"details": [{"retryDelay": "40s"}]}}
    assert LLMClient._retry_delay(e) == 40.0
    assert LLMClient._retry_delay(FakeAPIError(429)) == 60.0


def test_request_timeout_is_passed_to_the_sdk():
    llm, client = make({"a": [FakeResp()]}, ["a"], timeout_s=30)
    llm.generate("hi")
    assert client.models.configs[0].http_options.timeout == 30000


def test_timeout_error_fails_over_to_next_model():
    class ReadTimeout(Exception):
        pass

    llm, client = make({"a": [ReadTimeout()] * 3, "b": [FakeResp("b")]}, ["a", "b"])
    assert llm.generate("hi").model == "b"


def test_from_env_clients_share_one_circuit_breaker(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_MODEL_FAST", "m1")
    monkeypatch.setenv("GEMINI_MODEL_STRONG", "m2")
    assert LLMClient.from_env("fast").breaker is LLMClient.from_env("strong").breaker

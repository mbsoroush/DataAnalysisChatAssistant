"""Thin LLM layer around google-genai.

Responsibilities (kept out of the graph nodes on purpose):
  * retries with exponential backoff + jitter on transient errors (429/5xx/timeouts/empty output)
  * ordered model fallback chain (e.g. flash -> flash-lite)
  * per-model circuit breaker, so a down model is skipped instead of burning latency
  * structured output (pydantic schema) and token/latency telemetry on every call
  * a total deadline per call, so a trivial request can never stall the UI for minutes
  * telemetry events (llm_call, llm_error, llm_rate_limited, llm_circuit_open, llm_deadline)
    delivered through a hook, so the tracer can write them to traces.jsonl

Permanent errors (bad request, bad key) are NOT retried: retrying them only inflates cost.
"""
from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

log = logging.getLogger(__name__)

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
SKIP_MODEL_CODES = {404}  # model name not available for this key -> try the next one

EventHook = Callable[[str, dict], None]
_EVENT_HOOK: Optional[EventHook] = None  # process-wide hook, set once at startup


def set_event_hook(hook: Optional[EventHook]) -> None:
    """Register one function that receives every LLM telemetry event: hook(name, data).
    Wire it to the tracer once at startup instead of passing it to every client."""
    global _EVENT_HOOK
    _EVENT_HOOK = hook


class LLMUnavailableError(RuntimeError):
    """Every model in the chain failed after retries (or the deadline passed)."""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class LLMPermanentError(RuntimeError):
    """Non-retryable failure (invalid request, auth)."""


class _EmptyResponse(Exception):
    """Model returned no text, or text that does not match the schema. Treated as transient."""


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    attempts: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    fallback_used: bool


class CircuitBreaker:
    """Opens after `threshold` consecutive failures (or an explicit trip, e.g. a 429 with a retry delay)
    and lets a probe call through once the cooldown has passed."""

    def __init__(self, threshold: int = 4, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        until = self._open_until.get(key)
        if until is None:
            return True
        if self._clock() >= until:
            # half-open: allow a probe call; one more failure re-opens immediately
            self._open_until.pop(key, None)
            self._failures[key] = self.threshold - 1
            return True
        return False

    def record_success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        n = self._failures.get(key, 0) + 1
        self._failures[key] = n
        if n >= self.threshold:
            self._open_until[key] = self._clock() + self.cooldown_s

    def trip(self, key: str, cooldown_s: float) -> None:
        self._open_until[key] = self._clock() + cooldown_s


_SHARED_BREAKER = CircuitBreaker()  # model health is global: plan/analyst/router share it


class LLMClient:
    def __init__(
        self,
        client: Any,
        models: list[str],
        *,
        max_retries: int = 2,
        base_delay_s: float = 1.0,
        max_delay_s: float = 10.0,
        breaker: Optional[CircuitBreaker] = None,
        timeout_s: Optional[float] = None,
        deadline_s: Optional[float] = None,
        on_event: Optional[EventHook] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not models:
            raise ValueError("at least one model is required")
        self.client = client
        self.models = models
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.breaker = breaker or CircuitBreaker()
        self.timeout_s = timeout_s
        self.deadline_s = deadline_s
        self._on_event = on_event
        self._sleep = sleep

    @classmethod
    def from_env(cls, tier: str = "fast", on_event: Optional[EventHook] = None) -> "LLMClient":
        """tier='fast' uses GEMINI_MODEL_FAST, tier='strong' uses GEMINI_MODEL_STRONG.
        GEMINI_FALLBACKS is a comma-separated list tried in order after the primary.

        Fast-tier calls (router, classifiers) fail quickly: 1 retry, 12s per attempt, 20s total.
        Strong-tier calls (planning, analysis) get more room: 2 retries, 40s per attempt, 90s total.
        Override with LLM_TIMEOUT_S_<TIER> and LLM_DEADLINE_S_<TIER>."""
        from google import genai

        tier = tier.lower()
        fast = tier == "fast"
        primary = os.environ[f"GEMINI_MODEL_{tier.upper()}"]
        fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "").split(",") if m.strip()]
        chain = [primary] + [m for m in fallbacks if m != primary]
        return cls(
            genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
            chain,
            max_retries=1 if fast else 2,
            breaker=_SHARED_BREAKER,
            timeout_s=float(os.getenv(f"LLM_TIMEOUT_S_{tier.upper()}", "12" if fast else "40")),
            deadline_s=float(os.getenv(f"LLM_DEADLINE_S_{tier.upper()}", "20" if fast else "90")),
            on_event=on_event,
        )

    # ------------------------------------------------------------------
    def _emit(self, name: str, **data: Any) -> None:
        hook = self._on_event or _EVENT_HOOK
        if hook is None:
            return
        try:
            hook(name, data)
        except Exception:  # noqa: BLE001 - telemetry must never break a call
            log.exception("llm event hook failed for %s", name)

    def _over_deadline(self, t0: float) -> bool:
        return self.deadline_s is not None and (time.monotonic() - t0) >= self.deadline_s

    @staticmethod
    def _error_code(exc: Exception) -> Optional[int]:
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, (_EmptyResponse, TimeoutError, ConnectionError)):
            return True
        code = self._error_code(exc)
        if code is not None:
            return code in TRANSIENT_CODES
        # httpx / network errors from the SDK carry no status code
        return type(exc).__name__ in {"ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
                                   "WriteTimeout", "PoolTimeout", "TimeoutException"}

    @staticmethod
    def _retry_delay(exc: Exception) -> float:
        """Server-suggested retry delay from a 429 (RetryInfo), clamped to 15s..10min; default 60s."""
        m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", f"{getattr(exc, 'details', '')} {exc}")
        return max(15.0, min(float(m.group(1)), 600.0)) if m else 60.0

    def _backoff(self, attempt_idx: int) -> float:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt_idx))
        return delay * (0.5 + random.random() / 2)  # jitter

    @staticmethod
    def _parse(resp: Any, schema: Optional[Type[BaseModel]]) -> Any:
        """Parse structured output. A response that does not match the schema counts as a transient
        failure (retry, then fall over to the next model) instead of crashing the turn."""
        if schema is None:
            return None
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            return parsed
        try:
            return schema.model_validate_json(resp.text)
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            raise _EmptyResponse(f"unparseable structured output: {type(exc).__name__}") from exc

    def _call(self, model: str, prompt: Any, system: Optional[str], schema: Optional[Type[BaseModel]],
              temperature: float, max_output_tokens: Optional[int]):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)) if self.timeout_s else None,
        )
        resp = self.client.models.generate_content(model=model, contents=prompt, config=config)
        if not getattr(resp, "text", None):
            raise _EmptyResponse("empty model response")
        return resp

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Any,
        *,
        system: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResult:
        attempts = 0
        last_exc: Optional[Exception] = None
        t0 = time.monotonic()

        for idx, model in enumerate(self.models):
            if not self.breaker.allow(model):
                log.warning("llm.circuit_open model=%s (skipping)", model)
                self._emit("llm_circuit_open", model=model)
                continue

            for try_no in range(self.max_retries + 1):
                if self._over_deadline(t0):
                    ms = int((time.monotonic() - t0) * 1000)
                    log.warning("llm.deadline_exceeded deadline_s=%s attempts=%d", self.deadline_s, attempts)
                    self._emit("llm_deadline", attempts=attempts, ms=ms, deadline_s=self.deadline_s)
                    raise LLMUnavailableError(
                        f"deadline of {self.deadline_s}s exceeded after {attempts} attempts", attempts)
                attempts += 1
                t_try = time.monotonic()
                try:
                    resp = self._call(model, prompt, system, schema, temperature, max_output_tokens)
                    parsed = self._parse(resp, schema)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    code = self._error_code(exc)
                    self._emit("llm_error", model=model, attempt=try_no + 1, code=code,
                               error=type(exc).__name__, ms=int((time.monotonic() - t_try) * 1000))
                    if code in SKIP_MODEL_CODES:
                        log.warning("llm.model_unavailable model=%s code=%s", model, code)
                        break
                    if code == 429:
                        # Quota/rate limit: retrying the same model just burns time. Park it for the
                        # server-suggested delay and fail over to the next model right away.
                        cooldown = self._retry_delay(exc)
                        self.breaker.trip(model, cooldown)
                        log.warning("llm.rate_limited model=%s cooldown_s=%.0f", model, cooldown)
                        self._emit("llm_rate_limited", model=model, cooldown_s=cooldown)
                        break
                    if not self._is_transient(exc):
                        raise LLMPermanentError(f"{type(exc).__name__}: {exc}") from exc
                    self.breaker.record_failure(model)
                    log.warning("llm.transient_error model=%s attempt=%d code=%s err=%s",
                                model, try_no + 1, code, type(exc).__name__)
                    if try_no < self.max_retries and self.breaker.allow(model):
                        self._sleep(self._backoff(try_no))
                        continue
                    break  # retries exhausted or circuit opened -> next model
                else:
                    self.breaker.record_success(model)
                    usage = getattr(resp, "usage_metadata", None)
                    result = LLMResult(
                        text=resp.text,
                        parsed=parsed,
                        model=model,
                        attempts=attempts,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                        fallback_used=idx > 0,
                    )
                    self._emit(
                        "llm_call", model=model, attempts=attempts, ms=result.latency_ms,
                        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                        thought_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                        fallback_used=result.fallback_used, slow=result.latency_ms > 10_000,
                    )
                    return result

        raise LLMUnavailableError(
            f"all models failed after {attempts} attempts; last error: {type(last_exc).__name__ if last_exc else 'none'}",
            attempts,
        )

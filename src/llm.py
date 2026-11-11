"""Thin LLM layer around google-genai.

Responsibilities (kept out of the graph nodes on purpose):
  * retries with exponential backoff + jitter on transient errors (429/5xx/timeouts/empty output)
  * ordered model fallback chain (e.g. flash -> flash-lite)
  * per-model circuit breaker, so a down model is skipped instead of burning latency
  * structured output (pydantic schema) and token/latency telemetry on every call

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


class LLMUnavailableError(RuntimeError):
    """Every model in the chain failed after retries."""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class LLMPermanentError(RuntimeError):
    """Non-retryable failure (invalid request, auth)."""


class _EmptyResponse(Exception):
    """Model returned no text (blocked or empty). Treated as transient."""


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
        self._sleep = sleep

    @classmethod
    def from_env(cls, tier: str = "fast") -> "LLMClient":
        """tier='fast' uses GEMINI_MODEL_FAST, tier='strong' uses GEMINI_MODEL_STRONG.
        GEMINI_FALLBACKS is a comma-separated list tried in order after the primary."""
        from google import genai

        primary = os.environ[f"GEMINI_MODEL_{tier.upper()}"]
        fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "").split(",") if m.strip()]
        chain = [primary] + [m for m in fallbacks if m != primary]
        return cls(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), chain, breaker=_SHARED_BREAKER,
                   timeout_s=float(os.getenv("LLM_TIMEOUT_S", "40")))

    # ------------------------------------------------------------------
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
                continue

            for try_no in range(self.max_retries + 1):
                attempts += 1
                try:
                    resp = self._call(model, prompt, system, schema, temperature, max_output_tokens)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    code = self._error_code(exc)
                    if code in SKIP_MODEL_CODES:
                        log.warning("llm.model_unavailable model=%s code=%s", model, code)
                        break
                    if code == 429:
                        # Quota/rate limit: retrying the same model just burns time. Park it for the
                        # server-suggested delay and fail over to the next model right away.
                        cooldown = self._retry_delay(exc)
                        self.breaker.trip(model, cooldown)
                        log.warning("llm.rate_limited model=%s cooldown_s=%.0f", model, cooldown)
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
                    parsed = None
                    if schema is not None:
                        parsed = getattr(resp, "parsed", None) or schema.model_validate_json(resp.text)
                    return LLMResult(
                        text=resp.text,
                        parsed=parsed,
                        model=model,
                        attempts=attempts,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                        fallback_used=idx > 0,
                    )

        raise LLMUnavailableError(
            f"all models failed after {attempts} attempts; last error: {type(last_exc).__name__ if last_exc else 'none'}",
            attempts,
        )

"""Probe which Gemini models actually respond right now, with latency.

Usage (from repo root):
    python scripts/probe_models.py
    python scripts/probe_models.py gemini-2.5-flash gemini-3.5-flash gemini-flash-latest

Use the result to choose GEMINI_MODEL_FAST / GEMINI_MODEL_STRONG / GEMINI_FALLBACKS in .env.
A model that answers a trivial prompt reliably 3/3 times is a good primary; a fast, cheap one is a good fallback.
"""
import os
import sys
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
]
ROUNDS = 3


def main() -> None:
    candidates = sys.argv[1:] or DEFAULT_CANDIDATES
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"{'model':36} {'ok':>5} {'avg ms':>8}  last error")
    for model in candidates:
        ok, times, last_err = 0, [], ""
        for _ in range(ROUNDS):
            t0 = time.monotonic()
            try:
                r = client.models.generate_content(model=model, contents="Reply with exactly one word: pong")
                if r.text:
                    ok += 1
                    times.append((time.monotonic() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__} {getattr(e, 'code', '')}"
            time.sleep(1)
        avg = f"{sum(times) / len(times):.0f}" if times else "-"
        print(f"{model:36} {ok}/{ROUNDS:<3} {avg:>8}  {last_err}")


if __name__ == "__main__":
    main()

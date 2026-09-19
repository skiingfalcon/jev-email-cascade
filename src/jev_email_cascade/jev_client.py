"""Client for Jev, TypeSafe's decisions model -- reachable via OpenRouter's Decisions endpoint
or TypeSafe's own API, never via chat completions.

Both routes take the same ``{model, state, questions}`` body and return the same ``{model,
answers, usage}`` shape, so one client covers both; TypeSafe direct takes precedence when both
keys are set, matching every other community client for this API (an OPENROUTER_API_KEY left in
the shell by another tool must not silently reroute and re-bill).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.config import Settings
from jev_email_cascade.questions import Question, questions_json

RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}

# Published price, output free. Used only when a response omits usage.cost.
JEV_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000


class MissingKeyError(RuntimeError):
    """Neither OPENROUTER_API_KEY nor TYPESAFE_API_KEY is set."""


class JevClient:
    """Talks to one of Jev's two routes. Construct via :meth:`from_settings`."""

    name = "jev"

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        route: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        retries: int = 5,
        retry_delay_s: float = 1.0,
        max_retry_delay_s: float = 30.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise MissingKeyError("empty API key")
        self.url = url
        self.model = model
        self.route = route
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.max_retry_delay_s = max_retry_delay_s
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport, headers=headers
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.BaseTransport | None = None
    ) -> JevClient:
        if settings.typesafe_api_key:
            return cls(
                url=settings.typesafe_systemone_url,
                api_key=settings.typesafe_api_key,
                model=settings.typesafe_model,
                route="typesafe",
                timeout_s=settings.timeout_s,
                transport=transport,
            )
        if settings.openrouter_api_key:
            return cls(
                url=settings.jev_decisions_url,
                api_key=settings.openrouter_api_key,
                model=settings.jev_model,
                route="openrouter",
                timeout_s=settings.timeout_s,
                transport=transport,
                # Attribution headers OpenRouter recognises; harmless if ignored.
                extra_headers={
                    "HTTP-Referer": "https://github.com/jev-email-cascade",
                    "X-Title": "jev-email-cascade",
                },
            )
        raise MissingKeyError(
            "set OPENROUTER_API_KEY (https://openrouter.ai/keys) or TYPESAFE_API_KEY "
            "(https://console.typesafe.ai/) in .env"
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def decide(self, state: dict[str, Any], questions: dict[str, Question]) -> DecisionResult:
        body = {"model": self.model, "state": state, "questions": questions_json(questions)}
        delay = self.retry_delay_s
        t0 = time.perf_counter()
        last_error = "no attempt made"
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(self.url, json=body)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt >= self.retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, self.max_retry_delay_s)
                continue
            if r.status_code == 200:
                latency_ms = (time.perf_counter() - t0) * 1000
                return self._parse(r.json(), latency_ms)
            last_error = f"http {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRYABLE_STATUS or attempt >= self.retries:
                break
            wait = self._retry_after(r) or delay
            time.sleep(wait)
            delay = min(delay * 2, self.max_retry_delay_s)
        return DecisionResult(
            answers={}, latency_ms=(time.perf_counter() - t0) * 1000, error=last_error
        )

    def _parse(self, raw: dict, latency_ms: float) -> DecisionResult:
        answers = {qid: Answer.from_json(a) for qid, a in raw.get("answers", {}).items()}
        usage = raw.get("usage", {}) or {}
        input_tokens = usage.get("input_tokens")
        cost = usage.get("cost")
        cost_estimated = cost is None
        if cost is None and input_tokens is not None:
            cost = input_tokens * JEV_PRICE_PER_INPUT_TOKEN
        return DecisionResult(
            answers=answers,
            model=raw.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=usage.get("output_tokens"),
            cost_usd=cost,
            cost_estimated=cost_estimated,
            latency_ms=latency_ms,
            calls=1,
        )

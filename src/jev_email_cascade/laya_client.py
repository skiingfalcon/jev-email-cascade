"""Laya (Convai Innovations' open-weight decision model) served locally by ``laya-serve``.

``laya-serve`` speaks Jev's own ``POST /v1/systemone`` wire protocol, so the request body and the
``answers`` / ``usage`` it returns are the shapes ``JevClient`` already handles. The differences:
there is no API key unless the server was started with ``LAYA_API_KEY``, no per-token price
($0 marginal cost), and an optional ``model`` naming one of the server's resident checkpoints
(``english`` / ``typed-decisions`` / ``multilingual``) instead of a hosted model id. The server is
the sibling ``laya-host`` project (``C:\\Users\\kghosh\\projects\\laya-host``).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.config import Settings
from jev_email_cascade.gliner_backend import render_text
from jev_email_cascade.jev_client import RETRYABLE_STATUS
from jev_email_cascade.questions import Question, questions_json

STATE_MODES = ("raw", "rendered")


class LayaClient:
    """Talks to a local ``laya-serve``. Construct via :meth:`from_settings`."""

    name = "laya"

    def __init__(
        self,
        *,
        url: str,
        model: str | None = None,
        api_key: str | None = None,
        state_mode: str = "raw",
        max_len: int | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        retries: int = 3,
        retry_delay_s: float = 0.5,
        max_retry_delay_s: float = 10.0,
    ) -> None:
        if state_mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}")
        self.url = url
        self.model = model
        self.state_mode = state_mode
        self.max_len = max_len
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.max_retry_delay_s = max_retry_delay_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport, headers=headers
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.BaseTransport | None = None
    ) -> LayaClient:
        return cls(
            url=settings.laya_url,
            model=settings.laya_model,
            api_key=settings.laya_api_key,
            state_mode=settings.laya_state_mode,
            max_len=settings.laya_max_len,
            timeout_s=settings.timeout_s,
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self.url.split("/v1/", 1)[0]

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict | None:
        try:
            r = self._client.get(f"{self.base_url}/health")
        except httpx.HTTPError:
            return None
        return r.json() if r.status_code == 200 else None

    def healthy(self) -> bool:
        return self.health() is not None

    def backend_info(self) -> dict:
        """Recorded in run.json: where the server says each checkpoint computes, at which SHA."""
        info: dict[str, Any] = {
            "url": self.url,
            "model": self.model,
            "state_mode": self.state_mode,
            "max_len": self.max_len,
        }
        health = self.health()
        if health is not None:
            info["server"] = health
        return info

    def encode_state(self, state: dict[str, str]) -> dict[str, str]:
        if self.state_mode == "rendered":
            return {"body": render_text(state)}
        return state

    def request_body(self, state: dict[str, str], questions: dict[str, Question]) -> dict:
        body: dict[str, Any] = {
            "state": self.encode_state(state),
            "questions": questions_json(questions),
        }
        if self.model:
            body["model"] = self.model
        if self.max_len:
            body["max_len"] = self.max_len
        return body

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        body = self.request_body(state, questions)
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
                return self._parse(r.json(), (time.perf_counter() - t0) * 1000)
            last_error = f"http {r.status_code}: {r.text[:300]}"
            # 503 is also laya-serve's LAYA_MAX_CONCURRENT overflow, so it is worth a retry.
            if r.status_code not in RETRYABLE_STATUS or attempt >= self.retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, self.max_retry_delay_s)
        return DecisionResult(
            answers={}, latency_ms=(time.perf_counter() - t0) * 1000, error=last_error
        )

    def _parse(self, raw: dict, latency_ms: float) -> DecisionResult:
        answers = {qid: Answer.from_json(a) for qid, a in raw.get("answers", {}).items()}
        usage = raw.get("usage", {}) or {}
        routing = raw.get("routing") or {}
        return DecisionResult(
            answers=answers,
            model=routing.get("model") or raw.get("model") or self.model,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=0.0,
            cost_estimated=False,
            latency_ms=latency_ms,
            calls=1,
        )

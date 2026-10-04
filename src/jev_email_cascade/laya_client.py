"""Open-weight decision models served locally over Jev's own ``POST /v1/systemone`` protocol.

Both servers live in the sibling ``local-decision-model`` project
(``C:\\Users\\kghosh\\projects\\local-decision-model``):

- ``laya`` -- Convai Innovations' Laya (421M encoder) under ``laya-serve``. An optional ``model``
  names a resident checkpoint (``english`` / ``typed-decisions`` / ``multilingual``).
- ``rune`` -- Invergent's Rune 26B-A4B v3 (Gemma 4 MoE) under llama.cpp's ``llama-server``. The
  GGUF's metadata makes llama-server render the surogate decision prompt and read the option
  logits, so the request is the same body.

Both speak the protocol ``JevClient`` already handles, so the request body and the ``answers`` /
``usage`` they return parse the same way. The differences: no API key unless the server was
started with one, and no per-token price ($0 marginal cost).
"""

from __future__ import annotations

import ntpath
import time
from typing import Any

import httpx

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.config import Settings
from jev_email_cascade.gliner_backend import render_text
from jev_email_cascade.jev_client import RETRYABLE_STATUS
from jev_email_cascade.questions import Question, questions_json

STATE_MODES = ("raw", "rendered")
# Backend name -> the Settings field prefix that configures it.
SERVERS = ("laya", "rune")


class LayaClient:
    """Talks to a local System One server (``laya-serve`` or ``llama-server``). Construct via
    :meth:`from_settings`."""

    def __init__(
        self,
        *,
        url: str,
        name: str = "laya",
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
        self.name = name
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
        cls,
        settings: Settings,
        *,
        server: str = "laya",
        transport: httpx.BaseTransport | None = None,
    ) -> LayaClient:
        if server not in SERVERS:
            raise ValueError(f"server must be one of {SERVERS}")

        def field(suffix: str):
            return getattr(settings, f"{server}_{suffix}")

        return cls(
            url=field("url"),
            name=server,
            model=field("model"),
            api_key=field("api_key"),
            state_mode=field("state_mode"),
            max_len=field("max_len"),
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

    def _get_json(self, path: str) -> dict | None:
        try:
            r = self._client.get(f"{self.base_url}{path}")
        except httpx.HTTPError:
            return None
        return r.json() if r.status_code == 200 else None

    def backend_info(self) -> dict:
        """Recorded in run.json: what the server says it is running. laya-serve's /health names
        each checkpoint's device and SHA; llama-server's /props names the GGUF and the build."""
        info: dict[str, Any] = {
            "url": self.url,
            "model": self.model,
            "state_mode": self.state_mode,
            "max_len": self.max_len,
        }
        health = self.health()
        if health is not None:
            info["server"] = health
        props = self._get_json("/props")
        if props is not None:
            info["props"] = {
                k: props[k]
                for k in ("model_path", "build_info", "n_ctx", "total_slots")
                if k in props
            }
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
            # llama-server reports the GGUF's full path as `model`; keep just the file name.
            model=ntpath.basename(routing.get("model") or raw.get("model") or "")
            or self.model
            or self.name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=0.0,
            cost_estimated=False,
            latency_ms=latency_ms,
            calls=1,
        )

"""The optional generative hook: an OpenAI-compatible chat/completions endpoint (e.g. gpt-oss-120b
on the Spark's llama-server) that handles the band Jev's policy escalates -- an uncertain
category or priority, or anything flagged ``needs_decision`` / ``dissatisfied``.

Unset ``LLM_BASE_URL`` and this client is never constructed; the pipeline queues those items for
human review instead.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from jev_email_cascade.config import Settings
from jev_email_cascade.policy import Decision
from jev_email_cascade.questions import CATEGORIES

SYSTEM_PROMPT = (
    "You triage a business email that an automated classifier could not confidently place. "
    "Return ONLY a JSON object, no prose, no code fences, with exactly these keys: "
    f'"category" (one of {list(CATEGORIES)}), '
    '"priority" (integer 0-3, 0 = no action needed, 3 = act within two days), '
    '"rationale" (one sentence, why you chose the category and priority), '
    '"summary" (one sentence a human reviewer can read instead of the email), '
    '"entities" (an object with "deadline", "amount", and "party" -- null for any not present).'
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(content: str) -> dict[str, Any] | None:
    text = _FENCE_RE.sub("", content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class LlmResult:
    category: str | None = None
    priority: int | None = None
    rationale: str | None = None
    summary: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "priority": self.priority,
            "rationale": self.rationale,
            "summary": self.summary,
            "entities": self.entities,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def _decision_evidence(decision: Decision) -> dict:
    return {
        "category_guess": decision.category,
        "category_confidence": decision.category_conf,
        "priority_guess": decision.priority,
        "priority_confidence": decision.priority_conf,
        "flags": decision.flags,
        "why_escalated": decision.reasons,
    }


class LlmClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport, headers=headers
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.BaseTransport | None = None
    ) -> LlmClient | None:
        if not settings.llm_base_url:
            return None
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_s=settings.timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def healthy(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/models")
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    def triage(self, state: dict[str, str], decision: Decision) -> LlmResult:
        user = json.dumps({"email": state, "classifier_evidence": _decision_evidence(decision)})
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return LlmResult(error=str(exc), latency_ms=(time.perf_counter() - t0) * 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return LlmResult(error=f"http {r.status_code}: {r.text[:300]}", latency_ms=latency_ms)
        data = r.json()
        usage = data.get("usage", {}) or {}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return LlmResult(error="no message content in response", latency_ms=latency_ms)
        parsed = _parse_json_object(content or "")
        if parsed is None:
            return LlmResult(error="could not parse JSON from response", latency_ms=latency_ms)
        return LlmResult(
            category=parsed.get("category"),
            priority=parsed.get("priority"),
            rationale=parsed.get("rationale"),
            summary=parsed.get("summary"),
            entities=parsed.get("entities") or {},
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )

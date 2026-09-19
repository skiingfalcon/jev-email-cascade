"""The optional generative hook: an OpenAI-compatible chat/completions endpoint (e.g. gpt-oss-120b
on the Spark's llama-server) that handles the band Jev's policy escalates -- an uncertain
category or priority, or anything flagged ``needs_decision`` / ``dissatisfied``.

Unset ``LLM_BASE_URL`` and this client is never constructed; the pipeline queues those items for
human review instead. ``LLM_PROVIDER=frontier`` points the same hook at the hosted frontier model
instead, so "Jev + gpt-oss" and "Jev + Terra" on the escalated band are directly comparable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from jev_email_cascade.config import Settings
from jev_email_cascade.generative import (
    Pricing,
    _ChatClient,
    frontier_chat_client,
    local_chat_client,
)
from jev_email_cascade.policy import Decision
from jev_email_cascade.questions import CATEGORIES

PROVIDERS = ("local", "frontier")

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
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float = 0.0
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
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
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
    """The escalation hook. Thin wrapper over the same ``_ChatClient`` the generative backends
    use, so the local/openai request conventions are defined exactly once."""

    max_tokens_by_profile = {"local": 400, "openai": 3000}

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        profile: str = "local",
        reasoning_effort: str | None = None,
        pricing: Pricing | None = None,
        provider: str = "local",
    ) -> None:
        self._chat = _ChatClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
            profile=profile,
            reasoning_effort=reasoning_effort,
        )
        self.pricing = pricing or Pricing()
        self.provider = provider

    @property
    def base_url(self) -> str:
        return self._chat.base_url

    @property
    def model(self) -> str:
        return self._chat.model

    @property
    def profile(self) -> str:
        return self._chat.profile

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        reasoning_effort: str | None = None,
    ) -> LlmClient | None:
        """``None`` when the configured provider has nothing to call (local without
        ``LLM_BASE_URL``). Frontier without ``OPENAI_API_KEY`` raises, because asking for the
        hosted model and silently getting a review queue instead is the wrong kind of quiet."""
        provider = settings.llm_provider
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        if provider == "frontier":
            chat = frontier_chat_client(
                settings, transport=transport, reasoning_effort=reasoning_effort
            )
            pricing = Pricing.frontier(settings)
        else:
            chat = local_chat_client(
                settings, transport=transport, reasoning_effort=reasoning_effort
            )
            if chat is None:
                return None
            pricing = Pricing()
        return cls._wrap(chat, pricing=pricing, provider=provider)

    @classmethod
    def _wrap(cls, chat: _ChatClient, *, pricing: Pricing, provider: str) -> LlmClient:
        self = cls.__new__(cls)  # adopt an already-built chat client instead of re-deriving one
        self._chat = chat
        self.pricing = pricing
        self.provider = provider
        return self

    def close(self) -> None:
        self._chat.close()

    def healthy(self) -> bool:
        return self._chat.healthy()

    def triage(self, state: dict[str, str], decision: Decision) -> LlmResult:
        user = json.dumps({"email": state, "classifier_evidence": _decision_evidence(decision)})
        call = self._chat.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens_by_profile[self._chat.profile],
            response_format={"type": "json_object"},
        )
        usage = {
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "cached_input_tokens": call.cached_tokens,
            "reasoning_tokens": call.reasoning_tokens,
            "cost_usd": self.pricing.cost(
                call.prompt_tokens, call.completion_tokens, call.cached_tokens
            ),
            "latency_ms": call.latency_ms,
        }
        if call.error:
            return LlmResult(error=call.error, **usage)
        parsed = _parse_json_object(call.content or "")
        if parsed is None:
            return LlmResult(error="could not parse JSON from response", **usage)
        return LlmResult(
            category=parsed.get("category"),
            priority=parsed.get("priority"),
            rationale=parsed.get("rationale"),
            summary=parsed.get("summary"),
            entities=parsed.get("entities") or {},
            **usage,
        )

"""The shared answer schema and backend protocol.

Every backend -- the live Jev client, the offline mock, and the two generative modes -- produces
the same shapes, so the policy, pipeline, and report never need to know which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jev_email_cascade.questions import Question


@dataclass
class Answer:
    """One answer, in Jev's own response shape (``POST /v1/systemone`` / OpenRouter Decisions)."""

    type: str  # "noul" | "choice" | "score"
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None
    error: str | None = None

    @classmethod
    def from_json(cls, raw: dict) -> Answer:
        probs = raw.get("probabilities")
        if probs is not None:
            # TypeSafe's own docs note probability keys can arrive as strings even for a Score's
            # numeric levels; normalise to str uniformly and let callers cast where they need to.
            probs = {str(k): float(v) for k, v in probs.items()}
        return cls(
            type=raw.get("type", ""),
            noul=raw.get("noul"),
            choice=raw.get("choice"),
            score=raw.get("score"),
            probabilities=probs,
            confidence=raw.get("confidence"),
            legend=raw.get("legend"),
        )


@dataclass
class DecisionResult:
    """The outcome of asking one backend all questions about one email."""

    answers: dict[str, Answer]
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_estimated: bool = False
    latency_ms: float = 0.0
    calls: int = 1
    error: str | None = None
    # Generative backends only; Jev has neither.
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "answers": {
                qid: {k: v for k, v in vars(a).items() if v is not None}
                for qid, a in self.answers.items()
            },
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "cost_estimated": self.cost_estimated,
            "latency_ms": self.latency_ms,
            "calls": self.calls,
            "error": self.error,
        }


class DecisionBackend(Protocol):
    name: str

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult: ...

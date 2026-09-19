"""A deterministic stand-in for Jev, so the cascade runs end to end with no API key.

Same protocol, same answer schema as the live client. Scoring is plain keyword/regex counting
over the prepared state -- crude on purpose, so genuinely ambiguous or signal-free emails come
out with low confidence or a flat distribution by construction, rather than by hand-picking IDs.
That is also the seam a real local decision model would plug into if a hosted API is not an
option for the data in question.
"""

from __future__ import annotations

import re
import time

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.questions import CATEGORIES, PRIORITY_LEVELS, Question

_CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    "support": (
        "not working",
        "isn't working",
        "doesn't work",
        "bug",
        "error",
        "broken",
        "trouble",
        "failing",
        "crash",
        "can't log in",
        "cannot access",
        "reset my password",
        "how do i",
        "how to use",
        "keeps failing",
    ),
    "sales": (
        "pricing",
        "price of",
        "quote for",
        "a demo",
        "purchase",
        "planning to buy",
        "plan comparison",
        "enterprise plan",
        "free trial",
        "upgrade options",
        "cost of",
        "interested in your product",
    ),
    "billing": (
        "invoice",
        "charge",
        "charged",
        "refund",
        "payment",
        "billed",
        "billing",
        "overcharged",
        "credit card",
        "subscription fee",
        "receipt",
        "payment terms",
    ),
    "hr": (
        "job application",
        "resume",
        "cv attached",
        "payroll",
        "benefits enrollment",
        "onboarding",
        "leave request",
        "vacation days",
        "sick leave",
        "interview",
        "candidate",
        "offer letter",
        "background check",
    ),
    "vendor": (
        "our product line",
        "we supply",
        "our services include",
        "wholesale",
        "purchase order",
        "delivery schedule",
        "our catalog",
        "become a vendor",
        "as your supplier",
        "shipment",
        "we offer",
    ),
    "internal": (
        "team meeting",
        "stand-up",
        "sprint",
        "let's sync",
        "quarterly review",
        "hey team",
        "internal reminder",
    ),
    "spam": (
        "congratulations you",
        "click here",
        "limited time offer",
        "act now",
        "you've won",
        "free gift",
        "unsubscribe",
        "casino",
        "lottery",
        "crypto investment opportunity",
    ),
}

_AWAITING_REPLY = (
    "please let me know",
    "could you",
    "can you",
    "please advise",
    "look forward to your reply",
    "please respond",
    "waiting for your reply",
    "let us know",
)
_DEADLINE_RE = re.compile(
    r"\bby (mon|tue|wed|thu|fri|sat|sun|eod|cob)[a-z]*\b|"
    r"\bwithin \d+ (hour|day)s?\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b",
    re.IGNORECASE,
)
_NO_DEADLINE_RE = re.compile(r"\bno (specific )?deadline\b", re.IGNORECASE)
_DISSATISFIED = (
    "frustrated",
    "disappointed",
    "unacceptable",
    "not happy",
    "poor service",
    "terrible",
    "worst",
    "annoyed",
    "nobody reads these",
    "no one reads these",
    "outraged",
    "fourth time i",
)
_NEEDS_DECISION = (
    "please approve",
    "can you approve",
    "need your decision",
    "choose between",
    "which option",
    "authorize",
    "sign off",
    "decide whether",
)
_OPPORTUNITY = (
    "interested in purchasing",
    "want to buy",
    "upgrade to",
    "expand our",
    "add more seats",
    "add additional seats",
    "quote for additional",
    "referral for",
)
_INJECTION = (
    "ignore previous instructions",
    "ignore prior instructions",
    "disregard the above",
    "classify this as",
    "flag this as",
    "mark this as priority",
    "as an ai you must",
)

# Priority signal keywords, one tuple per level (index-aligned with PRIORITY_LEVELS).
_PRIORITY_TERMS: tuple[tuple[str, ...], ...] = (
    (),  # level 0 has no positive keywords; it wins by default when nothing else matches
    _AWAITING_REPLY,
    ("in the next few days", "later this week") + _DISSATISFIED,
    (
        "within 24 hours",
        "within 48 hours",
        "immediately",
        "asap",
        "urgent",
        "escalate",
        "legal action",
        "outage",
        "down for all users",
    ),
)


def _hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


def _normalise(scores: dict[str, float]) -> tuple[dict[str, float], float]:
    """Turn raw hit counts into a probability distribution and a confidence (its max)."""
    total = sum(scores.values())
    n = len(scores)
    if total <= 0:
        uniform = 1.0 / n
        return {k: uniform for k in scores}, uniform
    probs = {k: v / total for k, v in scores.items()}
    return probs, max(probs.values())


def _category_answer(text: str) -> Answer:
    scores = {c: float(_hits(text, _CATEGORY_TERMS.get(c, ()))) for c in CATEGORIES if c != "other"}
    if sum(scores.values()) == 0:
        # No keyword fired at all: genuinely can't tell which of eight options applies.
        # Spread near-uniformly, "other" a touch ahead, so confidence stays low rather than
        # reading as a confident vote for "other".
        n = len(CATEGORIES)
        base = 1.0 / (n + 0.5)
        probs = {c: base for c in scores}
        probs["other"] = base * 1.5
        total = sum(probs.values())
        probs = {c: p / total for c, p in probs.items()}
    else:
        scores["other"] = 0.0
        probs, _ = _normalise(scores)
    confidence = max(probs.values())
    choice = max(probs, key=probs.get)
    return Answer(type="choice", choice=choice, probabilities=probs, confidence=confidence)


def _priority_answer(text: str, category_confidence: float) -> Answer:
    scores = {str(i): float(_hits(text, terms)) for i, terms in enumerate(_PRIORITY_TERMS)}
    if sum(scores.values()) == 0:
        if category_confidence < 0.3:
            # The email's purpose itself could not be pinned down; its urgency is equally
            # opaque. A genuinely flat distribution, confidence 0 -- per TypeSafe's own guidance
            # on a Score with confidence 0.00, this must not be acted on.
            probs = dict.fromkeys(scores, 0.25)
            confidence = 0.0
        else:
            # No urgency language, but the email's purpose is otherwise clear: "no action
            # needed" read with confidence, not "cannot tell" -- most business email is exactly
            # this, and treating every quiet email as unscoreable would defeat the point of
            # asking for a priority at all.
            probs = {k: (1.0 if k == "0" else 0.0) for k in scores}
            confidence = 0.9
    else:
        probs, confidence = _normalise(scores)
    expectation = sum(int(k) * p for k, p in probs.items())
    legend = {str(i): desc for i, desc in enumerate(PRIORITY_LEVELS)}
    return Answer(
        type="score", score=expectation, probabilities=probs, confidence=confidence, legend=legend
    )


def _noul_answer(
    text: str, positive_terms: tuple[str, ...], *, deadline_guard: bool = False
) -> Answer:
    if deadline_guard:
        if _NO_DEADLINE_RE.search(text):
            p = 0.05
        elif _DEADLINE_RE.search(text):
            p = 0.92
        else:
            p = 0.08
    else:
        n = _hits(text, positive_terms)
        p = min(0.95, 0.08 + 0.42 * n) if n else 0.08
    return Answer(type="noul", noul=p, probabilities={"true": p, "false": 1 - p})


def _strong_noul_answer(text: str, positive_terms: tuple[str, ...]) -> Answer:
    """For phrases that are either clearly present or clearly not (e.g. an injection attempt),
    rather than a fuzzy signal that should land near 0.5 on a single weak hit."""
    p = 0.95 if _hits(text, positive_terms) else 0.05
    return Answer(type="noul", noul=p, probabilities={"true": p, "false": 1 - p})


class MockJev:
    name = "mock-jev"

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        t0 = time.perf_counter()
        text = f"{state.get('subject', '')}\n{state.get('body', '')}".lower()
        answers: dict[str, Answer] = {}
        # Priority's "no signal" case reads differently depending on whether the email's purpose
        # is otherwise clear, so category is always scored first even if not asked for.
        category_answer = _category_answer(text)
        for qid in questions:
            if qid == "category":
                answers[qid] = category_answer
            elif qid == "priority":
                answers[qid] = _priority_answer(text, category_answer.confidence or 0.0)
            elif qid == "deadline_present":
                answers[qid] = _noul_answer(text, (), deadline_guard=True)
            elif qid == "dissatisfied":
                answers[qid] = _noul_answer(text, _DISSATISFIED)
            elif qid == "needs_decision":
                answers[qid] = _noul_answer(text, _NEEDS_DECISION)
            elif qid == "opportunity":
                answers[qid] = _noul_answer(text, _OPPORTUNITY)
            elif qid == "injection_suspected":
                answers[qid] = _strong_noul_answer(text, _INJECTION)
            elif qid == "awaiting_reply":
                answers[qid] = _noul_answer(text, _AWAITING_REPLY)
        input_tokens = max(1, len(text) // 4)
        return DecisionResult(
            answers=answers,
            model="mock-jev-0",
            input_tokens=input_tokens,
            output_tokens=0,
            cost_usd=0.0,
            cost_estimated=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            calls=1,
        )

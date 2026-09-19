"""The routing policy: thresholds are decisions, not model output.

Tune these against a labelled sample, not by intuition; the numbers here are a starting point,
not a law. A single global confidence cutoff is the common mistake -- a branch that can issue a
refund needs more certainty than one that only moves a ticket to a different queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from jev_email_cascade.backends import Answer, DecisionResult

THRESHOLDS: dict[str, object] = {
    # Per-category confidence needed to accept it; "*" is the default for any category not
    # listed. Billing can trigger a refund workflow downstream, so it needs more certainty.
    "category_conf": {"billing": 0.85, "*": 0.60},
    "priority_conf": 0.60,
    # A Noul above this is treated as true, below its mirror (1 - noul_act) as false; the middle
    # band is "cannot tell" and must be escalated, not rounded.
    "noul_act": 0.80,
    # A Score confidence at or below this is a flat distribution: act on nothing it says.
    "flat_score_conf": 0.0,
}

NOUL_QUESTIONS = (
    "awaiting_reply",
    "deadline_present",
    "dissatisfied",
    "needs_decision",
    "opportunity",
    "injection_suspected",
)


class Route(StrEnum):
    AUTO = "auto"
    REVIEW = "review"
    LLM = "llm"


@dataclass
class Decision:
    category: str | None
    category_conf: float | None
    priority: float | None
    priority_conf: float | None
    flags: dict[str, bool | None]
    uncertain: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    route: Route = Route.AUTO


def _category_threshold(category: str | None) -> float:
    table = THRESHOLDS["category_conf"]
    assert isinstance(table, dict)
    if category is not None and category in table:
        return float(table[category])
    return float(table["*"])


def _noul_flag(answer: Answer | None, act: float) -> tuple[bool | None, bool]:
    """Returns (flag, is_uncertain). ``flag`` is None when the answer sits in the "can't tell"
    band between 1 - act and act -- a Noul at exactly 0.5 means that literally, not "maybe"."""
    if answer is None or answer.noul is None:
        return None, True
    p = answer.noul
    if p >= act:
        return True, False
    if p <= 1 - act:
        return False, False
    return None, True


def decide(answers: dict[str, Answer], *, llm_available: bool) -> Decision:
    uncertain: list[str] = []
    reasons: list[str] = []

    priority_answer = answers.get("priority")
    priority_conf = priority_answer.confidence if priority_answer else None
    priority_val = priority_answer.score if priority_answer else None
    if priority_conf is not None and priority_conf <= float(THRESHOLDS["flat_score_conf"]):
        uncertain.append("priority_flat")
        priority_val = None
    elif priority_conf is None or priority_conf < float(THRESHOLDS["priority_conf"]):
        uncertain.append("priority_low_confidence")

    category_answer = answers.get("category")
    category = category_answer.choice if category_answer else None
    category_conf = category_answer.confidence if category_answer else None
    threshold = _category_threshold(category)
    if (
        category is None
        or category == "other"
        or category_conf is None
        or category_conf < threshold
    ):
        uncertain.append("category_uncertain")

    flags: dict[str, bool | None] = {}
    act = float(THRESHOLDS["noul_act"])
    for qid in NOUL_QUESTIONS:
        flag, is_uncertain = _noul_flag(answers.get(qid), act)
        flags[qid] = flag
        if is_uncertain:
            uncertain.append(qid)

    if flags.get("injection_suspected"):
        reasons.append("injection_suspected: forced to review")
        route = Route.REVIEW
    else:
        escalate = bool(uncertain) or flags.get("needs_decision") or flags.get("dissatisfied")
        if not escalate:
            route = Route.AUTO
        elif llm_available:
            route = Route.LLM
        else:
            route = Route.REVIEW
        if uncertain:
            reasons.append(f"uncertain: {', '.join(uncertain)}")
        if flags.get("needs_decision"):
            reasons.append("needs_decision")
        if flags.get("dissatisfied"):
            reasons.append("dissatisfied")

    return Decision(
        category=category,
        category_conf=category_conf,
        priority=priority_val,
        priority_conf=priority_conf,
        flags=flags,
        uncertain=uncertain,
        reasons=reasons,
        route=route,
    )


def decide_result(result: DecisionResult, *, llm_available: bool) -> Decision:
    if not result.ok:
        return Decision(
            category=None,
            category_conf=None,
            priority=None,
            priority_conf=None,
            flags=dict.fromkeys(NOUL_QUESTIONS),
            uncertain=["backend_error"],
            reasons=[result.error or "backend error"],
            route=Route.REVIEW,
        )
    return decide(result.answers, llm_available=llm_available)

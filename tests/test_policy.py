from __future__ import annotations

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.policy import Route, decide, decide_result


def _answers(**kw: Answer) -> dict[str, Answer]:
    base = {
        "category": Answer(type="choice", choice="support", confidence=0.9),
        "priority": Answer(type="score", score=1.0, confidence=0.8),
        "awaiting_reply": Answer(type="noul", noul=0.9),
        "deadline_present": Answer(type="noul", noul=0.1),
        "dissatisfied": Answer(type="noul", noul=0.1),
        "needs_decision": Answer(type="noul", noul=0.1),
        "opportunity": Answer(type="noul", noul=0.1),
        "injection_suspected": Answer(type="noul", noul=0.05),
    }
    base.update(kw)
    return base


def test_clear_case_routes_auto() -> None:
    d = decide(_answers(), llm_available=False)
    assert d.route is Route.AUTO
    assert d.category == "support"
    assert d.uncertain == []


def test_support_at_low_confidence_is_uncertain_but_billing_needs_more() -> None:
    d_support = decide(
        _answers(category=Answer(type="choice", choice="support", confidence=0.65)),
        llm_available=False,
    )
    assert "category_uncertain" not in d_support.uncertain  # 0.65 clears the 0.60 default

    d_billing = decide(
        _answers(category=Answer(type="choice", choice="billing", confidence=0.80)),
        llm_available=False,
    )
    assert "category_uncertain" in d_billing.uncertain  # 0.80 does not clear billing's 0.85
    assert d_billing.route is Route.REVIEW


def test_flat_score_marks_priority_uncertain_and_none() -> None:
    d = decide(
        _answers(priority=Answer(type="score", score=1.5, confidence=0.0)), llm_available=False
    )
    assert "priority_flat" in d.uncertain
    assert d.priority is None
    assert d.route is Route.REVIEW


def test_noul_at_half_is_cannot_tell_not_a_guess() -> None:
    d = decide(_answers(dissatisfied=Answer(type="noul", noul=0.5)), llm_available=False)
    assert d.flags["dissatisfied"] is None
    assert "dissatisfied" in d.uncertain


def test_injection_forces_review_even_with_llm_available() -> None:
    d = decide(_answers(injection_suspected=Answer(type="noul", noul=0.95)), llm_available=True)
    assert d.flags["injection_suspected"] is True
    assert d.route is Route.REVIEW


def test_needs_decision_escalates_to_llm_when_available() -> None:
    d = decide(_answers(needs_decision=Answer(type="noul", noul=0.9)), llm_available=True)
    assert d.route is Route.LLM
    d_no_llm = decide(_answers(needs_decision=Answer(type="noul", noul=0.9)), llm_available=False)
    assert d_no_llm.route is Route.REVIEW


def test_dissatisfied_escalates() -> None:
    d = decide(_answers(dissatisfied=Answer(type="noul", noul=0.9)), llm_available=True)
    assert d.route is Route.LLM
    assert "dissatisfied" in d.reasons


def test_backend_error_routes_review() -> None:
    result = DecisionResult(answers={}, error="http 500: boom")
    d = decide_result(result, llm_available=True)
    assert d.route is Route.REVIEW
    assert "backend_error" in d.uncertain
    assert d.category is None

from __future__ import annotations

import pytest

from jev_email_cascade.report import (
    NOUL_QUESTIONS,
    category_class_metrics,
    noul_probability_metrics,
    priority_metrics,
    selective_accuracy,
)


def _row(
    true_cat: str,
    pred_cat: str,
    conf: float,
    *,
    prio=(0, 0.0),
    p_true: float = 0.5,
    noul_label: bool = False,
) -> dict:
    labels = {"category": true_cat, "priority": prio[0]}
    answers = {
        "category": {"type": "choice", "choice": pred_cat, "confidence": conf},
        "priority": {"type": "score", "score": prio[1]},
    }
    for q in NOUL_QUESTIONS:
        labels[q] = noul_label
        answers[q] = {"type": "noul", "noul": p_true}
    return {"labels": labels, "answers": answers}


def test_per_class_f1_and_macro() -> None:
    rows = [
        _row("billing", "billing", 0.9),
        _row("billing", "support", 0.6),  # billing FN, support FP
        _row("support", "support", 0.8),
        _row("hr", "hr", 0.95),
    ]
    m = category_class_metrics(rows)
    billing = m["per_class"]["billing"]
    assert billing["precision"] == 1.0 and billing["recall"] == 0.5
    assert billing["f1"] == pytest.approx(2 / 3)
    support = m["per_class"]["support"]
    assert support["precision"] == 0.5 and support["recall"] == 1.0
    assert m["per_class"]["hr"]["f1"] == 1.0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)


def test_selective_accuracy_keeps_most_confident() -> None:
    rows = [
        _row("a", "a", 0.99),
        _row("a", "a", 0.95),
        _row("a", "a", 0.90),
        _row("a", "b", 0.40),  # the only miss is the least confident
        _row("a", "(error)", None),
    ]
    rows[-1]["answers"]["category"].pop("confidence")
    sel = selective_accuracy(rows, coverages=(0.6, 1.0))
    assert sel["60%"]["n"] == 3 and sel["60%"]["accuracy"] == 1.0
    assert sel["60%"]["threshold"] == 0.90
    assert sel["100%"]["accuracy"] == 3 / 5


def test_priority_adjacent_vs_jump_errors() -> None:
    rows = [
        _row("a", "a", 1, prio=(0, 0.2)),  # exact
        _row("a", "a", 1, prio=(1, 2.0)),  # adjacent
        _row("a", "a", 1, prio=(0, 2.6)),  # jump of 3
    ]
    m = priority_metrics(rows)
    assert m["adjacent_errors"] == 1
    assert m["jump_errors"] == 1


def test_brier_and_ece_hand_computed() -> None:
    # Every noul says P(true)=0.8. Half the rows are true, so each question has
    # Brier = (0.2^2 + 0.8^2) / 2 = 0.34 and ECE = |0.8 - 0.5| = 0.3 (one occupied bin).
    rows = [
        _row("a", "a", 1, p_true=0.8, noul_label=True),
        _row("a", "a", 1, p_true=0.8, noul_label=False),
    ]
    m = noul_probability_metrics(rows)
    q = m["per_question"]["awaiting_reply"]
    assert q["brier"] == pytest.approx(0.34)
    assert q["ece"] == pytest.approx(0.3)
    assert m["pooled_n"] == 2 * len(NOUL_QUESTIONS)
    assert m["pooled_ece"] == pytest.approx(0.3)
    occupied = [b for b in m["pooled_bins"] if b["n"]]
    assert len(occupied) == 1 and occupied[0]["bin"] == "0.8-0.9"


def test_perfect_probabilities_score_zero() -> None:
    rows = [
        _row("a", "a", 1, p_true=1.0, noul_label=True),
        _row("a", "a", 1, p_true=0.0, noul_label=False),
    ]
    m = noul_probability_metrics(rows)
    assert m["pooled_brier"] == 0.0
    assert m["pooled_ece"] == 0.0

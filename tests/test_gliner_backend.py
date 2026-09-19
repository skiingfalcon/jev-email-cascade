"""No torch here: GlinerBackend is tested against a fake in-process model that mimics the
documented GLiNER2 schema/extract API, so these tests run without the `gliner` extra installed.
"""

from __future__ import annotations

import sys

import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.gliner_backend import (
    GlinerBackend,
    labels_for,
    render_text,
)
from jev_email_cascade.questions import CATEGORIES, QUESTIONS


def test_labels_for_choice_reuses_criteria_including_other() -> None:
    labels = labels_for(QUESTIONS["category"])
    assert set(labels) == set(CATEGORIES)
    assert "other" in labels
    assert labels["billing"] == QUESTIONS["category"].criteria["billing"]


def test_labels_for_score_uses_string_indices() -> None:
    labels = labels_for(QUESTIONS["priority"])
    assert set(labels) == {"0", "1", "2", "3"}
    assert labels["3"] == QUESTIONS["priority"].criteria[3]


def test_labels_for_noul_uses_true_false_with_instructions_prefixed() -> None:
    q = QUESTIONS["injection_suspected"]
    labels = labels_for(q)
    assert set(labels) == {"true", "false"}
    assert q.instructions in labels["true"]
    assert q.criteria["true"] in labels["true"]


def test_render_text_folds_header_into_one_string() -> None:
    text = render_text({"subject": "Hi", "from": "a@b.com", "body": "Body text."})
    assert "Subject: Hi" in text
    assert "From: a@b.com" in text
    assert text.endswith("Body text.")


def test_render_text_with_no_header_fields_is_just_the_body() -> None:
    assert render_text({"body": "just body"}) == "just body"


# --- fake model: mimics the documented schema()/extract() API without torch -------------------


class _FakeSchema:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    def classification(self, name: str, labels, multi_label: bool = False) -> _FakeSchema:
        self.tasks[name] = {"labels": labels, "multi_label": multi_label}
        return self


class _FakeModel:
    def __init__(self, replies: dict[str, dict] | None = None) -> None:
        self.replies = replies or {}
        self.extract_calls = 0

    def create_schema(self) -> _FakeSchema:
        return _FakeSchema()

    def extract(self, text: str, schema: _FakeSchema, include_confidence: bool = True) -> dict:
        self.extract_calls += 1
        if self.replies:
            return self.replies
        # Default: pick the first label of every task as a confident winner.
        return {
            name: {"label": next(iter(t["labels"])), "confidence": 0.99}
            for name, t in schema.tasks.items()
        }


def _backend(replies: dict[str, dict] | None = None) -> GlinerBackend:
    model = _FakeModel(replies)
    from jev_email_cascade.gliner_backend import _build_schema

    schema = _build_schema(model, QUESTIONS)
    return GlinerBackend(model, schema, "fastino/gliner2.5-multi-v1", "cpu")


def test_build_schema_creates_one_task_per_question() -> None:
    model = _FakeModel()
    from jev_email_cascade.gliner_backend import _build_schema

    schema = _build_schema(model, QUESTIONS)
    assert set(schema.tasks) == set(QUESTIONS)
    assert schema.tasks["category"]["multi_label"] is False


def test_decide_maps_winner_and_confidence_into_answer_schema() -> None:
    replies = {
        "category": {"label": "billing", "confidence": 0.9},
        "priority": {"label": "2", "confidence": 0.7},
        "awaiting_reply": {"label": "true", "confidence": 0.85},
        "deadline_present": {"label": "false", "confidence": 0.95},
        "dissatisfied": {"label": "false", "confidence": 0.9},
        "needs_decision": {"label": "false", "confidence": 0.6},
        "opportunity": {"label": "false", "confidence": 0.9},
        "injection_suspected": {"label": "false", "confidence": 0.99},
    }
    backend = _backend(replies)
    result = backend.decide({"subject": "s", "body": "b"}, QUESTIONS)

    assert result.ok
    assert result.model == "fastino/gliner2.5-multi-v1"
    assert result.cost_usd == 0.0
    assert result.cost_estimated is False
    assert result.calls == 1
    assert set(result.answers) == set(QUESTIONS)
    assert result.answers["category"].choice == "billing"
    assert result.answers["category"].confidence == pytest.approx(0.9)
    # winner "2" at 0.7, the other 3 levels share the remaining 0.3 uniformly (0.1 each):
    # expectation = 0*0.1 + 1*0.1 + 2*0.7 + 3*0.1 = 1.8
    assert result.answers["priority"].score == pytest.approx(1.8)
    assert result.answers["awaiting_reply"].noul == pytest.approx(0.85)
    assert result.answers["deadline_present"].noul == pytest.approx(0.05)  # 1 - 0.95


def test_noul_reconstruction_is_exact_two_label_split() -> None:
    """A 2-label task's reconstructed distribution is exact, not an approximation."""
    backend = _backend({"injection_suspected": {"label": "true", "confidence": 0.8}})
    result = backend.decide(
        {"body": "b"}, {"injection_suspected": QUESTIONS["injection_suspected"]}
    )
    assert result.answers["injection_suspected"].noul == pytest.approx(0.8)
    assert result.probabilities_reconstructed is False


def test_wide_choice_reconstruction_is_flagged_as_reconstructed() -> None:
    """An 8-way category task built from a winner-only confidence sets the honesty flag."""
    replies = {"category": {"label": "billing", "confidence": 0.9}}
    backend = _backend(replies)
    result = backend.decide({"body": "b"}, {"category": QUESTIONS["category"]})
    assert result.answers["category"].choice == "billing"
    assert result.probabilities_reconstructed is True


def test_missing_task_in_response_is_a_per_question_error_not_a_crash() -> None:
    backend = _backend(
        {"category": {"label": "billing", "confidence": 0.9}}
    )  # every other key missing
    result = backend.decide({"body": "b"}, QUESTIONS)
    assert result.answers["category"].error is None
    assert result.answers["priority"].error == "missing task in GLiNER2 response"
    assert not result.ok


def test_extract_exception_is_recorded_as_a_row_error() -> None:
    class _Boom(_FakeModel):
        def extract(self, *a, **k):
            raise RuntimeError("boom")

    model = _Boom()
    from jev_email_cascade.gliner_backend import _build_schema

    schema = _build_schema(model, QUESTIONS)
    backend = GlinerBackend(model, schema, "m", "cpu")
    result = backend.decide({"body": "b"}, QUESTIONS)
    assert not result.ok
    assert "boom" in (result.error or "")
    assert result.answers == {}


def test_full_distribution_response_is_used_directly_and_marked_exact() -> None:
    """If the installed library ever returns the full distribution (not just the winner), use
    it as-is instead of reconstructing, and don't set the approximation flag."""
    replies = {
        "injection_suspected": {"true": 0.1, "false": 0.9},
    }
    backend = _backend(replies)
    result = backend.decide(
        {"body": "b"}, {"injection_suspected": QUESTIONS["injection_suspected"]}
    )
    assert result.answers["injection_suspected"].noul == pytest.approx(0.1)
    assert result.probabilities_reconstructed is False


def test_backend_info_reports_device() -> None:
    backend = _backend()
    assert backend.backend_info() == {"device": "cpu"}


def test_from_settings_without_gliner2_installed_raises_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", None)
    with pytest.raises(RuntimeError, match="uv sync --extra gliner"):
        GlinerBackend.from_settings(Settings())


def test_build_backend_gliner_listed_in_backend_names() -> None:
    from jev_email_cascade.pipeline import BACKEND_NAMES

    assert "gliner" in BACKEND_NAMES

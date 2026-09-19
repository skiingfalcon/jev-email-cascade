"""An open-weight, self-hosted decision-model backend: GLiNER2.5
(`fastino/gliner2.5-multi-v1`), a bidirectional encoder that reads text plus a schema of typed
labels and returns labels with scores in one forward pass -- the same shape as Jev's answers, at
$0 marginal cost, because the weights are Apache-2.0 and run on our own hardware.

This is the seam the README's data-hosting caveat points at: a local decision model behind the
same :class:`~jev_email_cascade.backends.DecisionBackend` protocol :class:`MockJev` exercises.
Not installed by default -- `gliner2[local]` pulls torch (~2 GB) -- so this module imports
``gliner2`` lazily inside :meth:`GlinerBackend.from_settings`; ``uv sync --extra gliner`` first.

The question<->schema mapping is derived once from ``questions.py`` so the two backends answer
literally the same contract:

- ``Choice``  -> one single-label classification task; its ``criteria`` dict becomes the labels'
  descriptions (``category``, 8 labels including the ``other`` escape option).
- ``Score``   -> one single-label classification task over ``"0".."N-1"``, the level text as the
  description of each label (``priority``, 4 labels); the numeric score is the same
  probability-weighted expectation :func:`generative.answer_from_token_probs` computes for
  ``gen-logprob``.
- ``Noul``    -> one single-label classification task over the literal labels ``"true"`` /
  ``"false"``, with the instructions sentence prefixed to both criteria as the description --
  scope and negation have nowhere else to go once the instruction can't be sent as free text.

GLiNER2's classification API returns each task's winning label and its confidence
(``include_confidence=True``); it does not (as far as this client can tell without a live model
against which to check) expose the full distribution over the losing labels. For a 2-label Noul
task that is no loss: P(false) = 1 - P(true) is exact regardless. For the 8-way category and
4-way priority tasks it is an approximation -- the remaining mass is spread uniformly over the
other labels, because that is the least-committal assumption, not because it is likely true.
Rows where this happened set ``probabilities_reconstructed=True`` so the report never treats the
reconstructed spread as a measured distribution.
"""

from __future__ import annotations

import time
from typing import Any

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.config import Settings
from jev_email_cascade.generative import answer_from_token_probs
from jev_email_cascade.questions import QUESTIONS, Choice, Noul, Question, Score

INSTALL_HINT = (
    "gliner2 is not installed; run `uv sync --extra gliner` on a machine with room for torch "
    "(~2 GB) before using --backend gliner"
)


def _qtype(q: Question) -> str:
    if isinstance(q, Noul):
        return "noul"
    if isinstance(q, Choice):
        return "choice"
    if isinstance(q, Score):
        return "score"
    raise TypeError(type(q))


def labels_for(q: Question) -> dict[str, str]:
    """{label: description} for one question -- the GLiNER2 schema task's label set. The label
    strings are exactly the answer-schema keys downstream code expects: category/option names
    for Choice, "0".."N-1" for Score, "true"/"false" for Noul."""
    if isinstance(q, Choice):
        return dict(q.criteria)
    if isinstance(q, Score):
        return {str(i): desc for i, desc in enumerate(q.criteria)}
    if isinstance(q, Noul):
        true_desc = (q.criteria or {}).get("true", "")
        false_desc = (q.criteria or {}).get("false", "")
        return {
            "true": f"{q.instructions} {true_desc}".strip(),
            "false": f"{q.instructions} {false_desc}".strip(),
        }
    raise TypeError(type(q))


def render_text(state: dict[str, str]) -> str:
    """GLiNER2 takes one string, not the {subject, from, received, body} dict Jev's ``state``
    is; fold it into a single block the way a human would read the message."""
    header = "\n".join(
        f"{label}: {state[key]}"
        for label, key in (("Subject", "subject"), ("From", "from"), ("Received", "received"))
        if state.get(key)
    )
    body = state.get("body", "")
    return f"{header}\n\n{body}" if header else body


def _label_probs(labels: list[str], raw: Any) -> tuple[dict[str, float], bool]:
    """Map one task's GLiNER2 result into a normalised {label: probability} distribution, plus
    whether it is exact (True) or reconstructed from a winner-only confidence (False, except
    when there are only 2 labels, where reconstruction is exact by construction)."""
    # A full distribution, if the installed version ever returns one: {label: score, ...}.
    if (
        isinstance(raw, dict)
        and set(raw) & set(labels)
        and all(isinstance(v, int | float) for v in raw.values())
    ):
        total = sum(raw.get(label, 0.0) for label in labels) or 1.0
        return {label: raw.get(label, 0.0) / total for label in labels}, True
    # The documented shape: {"label": ..., "confidence": ...}.
    if isinstance(raw, dict) and "label" in raw:
        winner = str(raw["label"])
        confidence = float(raw.get("confidence", 1.0))
    else:
        winner, confidence = str(raw), 1.0
    if winner not in labels:
        return {}, False
    remaining = [label for label in labels if label != winner]
    if not remaining:
        return {winner: confidence}, True
    other_each = (1.0 - confidence) / len(remaining)
    probs = {winner: confidence, **dict.fromkeys(remaining, other_each)}
    return probs, len(labels) <= 2


def _build_schema(model: Any, questions: dict[str, Question]):
    schema = model.create_schema()
    for qid, q in questions.items():
        labels = labels_for(q)
        try:
            schema = schema.classification(qid, labels, multi_label=False)
        except TypeError:
            # Older/newer API: labels as a bare list, descriptions dropped. Schema-build only
            # happens once per process, so this fallback never turns into a per-email crash.
            schema = schema.classification(qid, list(labels), multi_label=False)
    return schema


class GlinerBackend:
    """A GLiNER2.5 model wrapped as a :class:`DecisionBackend`. $0 marginal cost, self-hosted."""

    name = "gliner"

    def __init__(self, model: Any, schema: Any, model_id: str, device: str) -> None:
        self.model = model
        self.schema = schema
        self.model_id = model_id
        self.device = device

    @classmethod
    def from_settings(cls, settings: Settings) -> GlinerBackend:
        try:
            from gliner2 import AutoExtractor
        except ImportError as exc:
            raise RuntimeError(INSTALL_HINT) from exc
        model = AutoExtractor.from_pretrained(
            settings.gliner_model, map_location=settings.gliner_device
        )
        if settings.gliner_fp16:
            model.quantize()
        schema = _build_schema(model, QUESTIONS)
        backend = cls(model, schema, settings.gliner_model, settings.gliner_device)
        backend._warm_up()
        return backend

    def _warm_up(self) -> None:
        # So the first scored email's latency isn't model-load / lazy-compile time.
        try:
            self.model.extract("Warm-up.", self.schema, include_confidence=True)
        except Exception:
            pass  # best-effort; a real problem surfaces on the first real call instead

    def backend_info(self) -> dict:
        info: dict = {"device": self.device}
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return info

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        text = render_text(state)
        t0 = time.perf_counter()
        try:
            raw = self.model.extract(text, self.schema, include_confidence=True)
        except Exception as exc:  # noqa: BLE001 -- recorded as a row error, not a crash
            return DecisionResult(
                answers={},
                model=self.model_id,
                input_tokens=max(1, len(text) // 4),
                output_tokens=0,
                cost_usd=0.0,
                cost_estimated=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                calls=1,
                error=str(exc),
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answers: dict[str, Answer] = {}
        errors: list[str] = []
        reconstructed = False
        for qid, q in questions.items():
            task_raw = raw.get(qid) if isinstance(raw, dict) else None
            if task_raw is None:
                answers[qid] = Answer(type=_qtype(q), error="missing task in GLiNER2 response")
                errors.append(f"{qid}: {answers[qid].error}")
                continue
            labels = list(labels_for(q))
            probs, exact = _label_probs(labels, task_raw)
            if not exact:
                reconstructed = True
            answers[qid] = answer_from_token_probs(q, probs)
            if answers[qid].error:
                errors.append(f"{qid}: {answers[qid].error}")

        return DecisionResult(
            answers=answers,
            model=self.model_id,
            input_tokens=max(1, len(text) // 4),
            output_tokens=0,
            cost_usd=0.0,
            cost_estimated=False,
            latency_ms=latency_ms,
            calls=1,
            error="; ".join(errors) or None,
            probabilities_reconstructed=reconstructed,
        )

"""Generative backends for the comparison scenario: the same eight questions, answered by a
chat model instead of Jev, in two shapes.

``gen-json`` is the common pattern: one prompt lists every question, the model returns one JSON
object with a self-reported confidence per answer. It is the shape the cascade pitch says
degrades as questions pile into a single call -- this module measures whether it does, on our
own data, instead of asserting it.

``gen-logprob`` is the fairer comparison on "probability": one call per question, the answer
forced to a single token, and the probability read from the token's own log-probability rather
than a number the model typed. Costs eight times the calls; the report is where that trade
becomes visible.

Both map onto the same :class:`~jev_email_cascade.backends.Answer` schema Jev uses, so nothing
downstream needs to know which backend ran. ``MockGenerative`` exercises the same mapping code
with synthetic model output, so the offline demo is not a separate implementation to trust.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass

import httpx

from jev_email_cascade.backends import Answer, DecisionResult
from jev_email_cascade.config import Settings
from jev_email_cascade.mock_jev import MockJev
from jev_email_cascade.questions import Choice, Noul, Question, Score

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(content: str) -> dict | None:
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


def _qtype(q: Question) -> str:
    if isinstance(q, Noul):
        return "noul"
    if isinstance(q, Choice):
        return "choice"
    if isinstance(q, Score):
        return "score"
    raise TypeError(type(q))


def question_prompt(qid: str, q: Question) -> str:
    if isinstance(q, Noul):
        crit = ""
        if q.criteria:
            true_desc = q.criteria.get("true", "")
            false_desc = q.criteria.get("false", "")
            crit = f" True means: {true_desc} False means: {false_desc}"
        return f'"{qid}" (yes/no): {q.instructions}{crit}'
    if isinstance(q, Choice):
        options = "; ".join(f"{k} = {v}" for k, v in q.criteria.items())
        return f'"{qid}" (choose one key): {q.instructions} Options: {options}'
    if isinstance(q, Score):
        levels = "; ".join(f"{i} = {d}" for i, d in enumerate(q.criteria))
        return f'"{qid}" (choose one level index): {q.instructions} Levels: {levels}'
    raise TypeError(type(q))


def candidate_tokens(q: Question) -> dict[str, str]:
    """Token text -> answer key, for the single-token-answer (gen-logprob) prompt."""
    if isinstance(q, Noul):
        return {"yes": "true", "no": "false"}
    if isinstance(q, Choice):
        return {k: k for k in q.criteria}
    if isinstance(q, Score):
        return {str(i): str(i) for i in range(len(q.criteria))}
    raise TypeError(type(q))


def map_json_answer(q: Question, raw: object) -> Answer:
    """Map one question's slice of a parsed gen-json response into the Answer schema."""
    qtype = _qtype(q)
    if not isinstance(raw, dict):
        return Answer(type=qtype, error=f"expected an object, got {type(raw).__name__}")
    conf_raw = raw.get("confidence")
    confidence = float(conf_raw) if isinstance(conf_raw, int | float) else 0.5
    ans = raw.get("answer")
    if isinstance(q, Noul):
        truthy = str(ans).strip().lower() in ("true", "yes", "1")
        p = confidence if truthy else 1 - confidence
        return Answer(
            type="noul", noul=p, confidence=confidence, probabilities={"true": p, "false": 1 - p}
        )
    if isinstance(q, Choice):
        choice = str(ans) if ans is not None else None
        if choice not in q.criteria:
            return Answer(type="choice", error=f"unknown option {choice!r}")
        return Answer(
            type="choice", choice=choice, confidence=confidence, probabilities={choice: confidence}
        )
    if isinstance(q, Score):
        try:
            level = int(ans)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return Answer(type="score", error=f"unparseable level {ans!r}")
        if not 0 <= level < len(q.criteria):
            return Answer(type="score", error=f"level {level} out of range")
        return Answer(
            type="score",
            score=float(level),
            confidence=confidence,
            probabilities={str(level): confidence},
        )
    raise TypeError(type(q))


def answer_from_token_probs(q: Question, probs: dict[str, float]) -> Answer:
    """Map a normalised {answer-key: probability} distribution (from top_logprobs, live or
    synthetic) into the Answer schema. Pure and network-free, so both the live gen-logprob
    backend and the offline mock go through the same code."""
    qtype = _qtype(q)
    if not probs:
        return Answer(type=qtype, error="no candidate token found in the model's top choices")
    if isinstance(q, Noul):
        p_yes = probs.get("true", 0.0)
        return Answer(type="noul", noul=p_yes, probabilities=probs)
    if isinstance(q, Choice):
        best = max(probs, key=probs.get)
        return Answer(type="choice", choice=best, probabilities=probs, confidence=probs[best])
    if isinstance(q, Score):
        expectation = sum(int(k) * v for k, v in probs.items())
        legend = {str(i): d for i, d in enumerate(q.criteria)}
        return Answer(
            type="score",
            score=expectation,
            probabilities=probs,
            confidence=max(probs.values()),
            legend=legend,
        )
    raise TypeError(type(q))


@dataclass
class _ChatCall:
    content: str | None
    top_logprobs: list[dict] | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    latency_ms: float
    error: str | None = None


class _ChatClient:
    """The one seam that talks to an OpenAI-compatible chat/completions endpoint."""

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

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: int = 10,
    ) -> _ChatCall:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return _ChatCall(
                None, None, None, None, None, (time.perf_counter() - t0) * 1000, str(exc)
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return _ChatCall(
                None, None, None, None, None, latency_ms, f"http {r.status_code}: {r.text[:300]}"
            )
        data = r.json()
        usage = data.get("usage", {}) or {}
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError):
            return _ChatCall(None, None, None, None, None, latency_ms, "no message content")
        top: list[dict] | None = None
        lp = (choice.get("logprobs") or {}).get("content")
        if lp:
            top = lp[0].get("top_logprobs")
        return _ChatCall(
            content=content,
            top_logprobs=top,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            latency_ms=latency_ms,
        )


GEN_JSON_SYSTEM = (
    "You will answer several questions about one email in a single JSON object. For each "
    'question id return an object: {"answer": ..., "confidence": 0-1}. "answer" is true/false '
    "for a yes/no question, one option key for a choose-one question, or an integer level index "
    "for a level question. Read every instruction literally. Return ONLY the JSON object, no "
    "prose, no code fences."
)


class GenJsonBackend:
    """One chat completion per email; every question answered in the same JSON object."""

    name = "gen-json"

    def __init__(self, client: _ChatClient, price_per_mtok: float = 0.0) -> None:
        self.client = client
        self.price_per_mtok = price_per_mtok

    @classmethod
    def from_settings(cls, settings: Settings, *, price_per_mtok: float = 0.0, transport=None):
        if not settings.llm_base_url:
            return None
        client = _ChatClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_s=settings.timeout_s,
            transport=transport,
        )
        return cls(client, price_per_mtok=price_per_mtok)

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        lines = [f"- {question_prompt(qid, q)}" for qid, q in questions.items()]
        user = "Email:\n" + json.dumps(state) + "\n\nQuestions:\n" + "\n".join(lines)
        call = self.client.complete(
            [{"role": "system", "content": GEN_JSON_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=600,
        )
        if call.error or call.content is None:
            return DecisionResult(answers={}, error=call.error, latency_ms=call.latency_ms, calls=1)
        parsed = parse_json_object(call.content)
        answers: dict[str, Answer] = {}
        error = None
        if parsed is None:
            error = "could not parse a JSON object from the response"
        else:
            for qid, q in questions.items():
                if qid not in parsed:
                    answers[qid] = Answer(type=_qtype(q), error="missing key in generative JSON")
                else:
                    answers[qid] = map_json_answer(q, parsed[qid])
        return DecisionResult(
            answers=answers,
            model=self.client.model,
            input_tokens=call.prompt_tokens,
            output_tokens=call.completion_tokens,
            cost_usd=_hosted_cost(call.prompt_tokens, call.completion_tokens, self.price_per_mtok),
            cost_estimated=True,
            latency_ms=call.latency_ms,
            calls=1,
            error=error,
        )


class GenLogprobBackend:
    """One constrained, single-token call per question; the probability comes from the model's
    own log-probabilities rather than a number it typed."""

    name = "gen-logprob"

    def __init__(self, client: _ChatClient, price_per_mtok: float = 0.0) -> None:
        self.client = client
        self.price_per_mtok = price_per_mtok

    @classmethod
    def from_settings(cls, settings: Settings, *, price_per_mtok: float = 0.0, transport=None):
        if not settings.llm_base_url:
            return None
        client = _ChatClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_s=settings.timeout_s,
            transport=transport,
        )
        return cls(client, price_per_mtok=price_per_mtok)

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        answers: dict[str, Answer] = {}
        errors: list[str] = []
        prompt_tokens = completion_tokens = 0
        latency_ms = 0.0
        for qid, q in questions.items():
            tokens = candidate_tokens(q)
            instruction = (
                f"Answer with exactly one token, one of: {', '.join(tokens)}. No other text."
            )
            user = (
                f"Email:\n{json.dumps(state)}\n\nQuestion: {question_prompt(qid, q)}\n{instruction}"
            )
            call = self.client.complete(
                [{"role": "user", "content": user}], max_tokens=1, logprobs=True
            )
            latency_ms += call.latency_ms
            prompt_tokens += call.prompt_tokens or 0
            completion_tokens += call.completion_tokens or 0
            if call.error or call.top_logprobs is None:
                answers[qid] = Answer(type=_qtype(q), error=call.error or "no logprobs in response")
                errors.append(f"{qid}: {answers[qid].error}")
                continue
            probs = _extract_token_probs(call.top_logprobs, tokens)
            answers[qid] = answer_from_token_probs(q, probs)
            if answers[qid].error:
                errors.append(f"{qid}: {answers[qid].error}")
        return DecisionResult(
            answers=answers,
            model=self.client.model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=_hosted_cost(prompt_tokens, completion_tokens, self.price_per_mtok),
            cost_estimated=True,
            latency_ms=latency_ms,
            calls=len(questions),
            error="; ".join(errors) or None,
        )


def _extract_token_probs(top_logprobs: list[dict], tokens: dict[str, str]) -> dict[str, float]:
    """llama-server's top_logprobs entries -> {answer-key: probability}, summing tokens that
    map to the same answer key (e.g. "Yes" and "yes") and renormalising over only the keys we
    recognise, since the raw distribution also spends mass on tokens we did not ask for."""
    mass: dict[str, float] = {}
    for entry in top_logprobs:
        tok = str(entry.get("token", "")).strip().lower()
        logprob = entry.get("logprob")
        if logprob is None:
            continue
        for token_text, answer_key in tokens.items():
            if tok == token_text.lower() or tok == token_text.lower()[:1]:
                mass[answer_key] = mass.get(answer_key, 0.0) + math.exp(logprob)
    total = sum(mass.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in mass.items()}


def _hosted_cost(
    prompt_tokens: int | None, completion_tokens: int | None, price_per_mtok: float
) -> float:
    if not price_per_mtok:
        return 0.0
    return ((prompt_tokens or 0) + (completion_tokens or 0)) * price_per_mtok / 1_000_000


class MockGenerative:
    """Offline stand-in for both generative modes: no network, but the parsing and mapping code
    it exercises (``parse_json_object``, ``map_json_answer``, ``answer_from_token_probs``) is the
    same code the live backends use.

    Scored with the same keyword heuristics as :class:`~jev_email_cascade.mock_jev.MockJev`, so
    accuracy differences in a mock-only run come from the generative *shape* (one JSON object
    for eight questions vs. one constrained token each), not from a weaker scorer.
    """

    def __init__(self, mode: str = "json", price_per_mtok: float = 0.0) -> None:
        if mode not in ("json", "logprob"):
            raise ValueError(mode)
        self.mode = mode
        self.price_per_mtok = price_per_mtok
        self.name = f"mock-gen-{mode}"
        self._jev = MockJev()

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        t0 = time.perf_counter()
        jev_result = self._jev.decide(state, questions)
        if self.mode == "json":
            return self._as_json_mode(jev_result, questions, t0)
        return self._as_logprob_mode(jev_result, questions, t0)

    def _as_json_mode(self, jev_result: DecisionResult, questions: dict[str, Question], t0: float):
        # Render the mock Jev's own answers as the JSON text a generative model might have typed.
        # On a deterministic ~1-in-5 subset of emails, drop one question's key entirely -- a
        # stand-in for a model losing track of one of several questions asked in the same
        # prompt, rather than asserting it happens on every single call.
        obj: dict = {}
        for qid, q in questions.items():
            a = jev_result.answers.get(qid)
            if a is None:
                continue
            if isinstance(q, Noul):
                obj[qid] = {
                    "answer": (a.noul or 0) >= 0.5,
                    "confidence": round(a.confidence or 0.6, 3),
                }
            elif isinstance(q, Choice):
                obj[qid] = {"answer": a.choice, "confidence": round(a.confidence or 0.5, 3)}
            elif isinstance(q, Score):
                obj[qid] = {
                    "answer": int(round(a.score or 0)),
                    "confidence": round(a.confidence or 0.5, 3),
                }
        digest = int(hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(), 16)
        if "opportunity" in obj and digest % 5 == 0:
            del obj["opportunity"]
        content = "```json\n" + json.dumps(obj) + "\n```"
        parsed = parse_json_object(content)
        answers = {}
        for qid, q in questions.items():
            if parsed is not None and qid in parsed:
                answers[qid] = map_json_answer(q, parsed[qid])
            else:
                answers[qid] = Answer(type=_qtype(q), error="missing key in generative JSON")
        return DecisionResult(
            answers=answers,
            model="mock-gen-json-0",
            input_tokens=jev_result.input_tokens,
            output_tokens=len(questions) * 12,
            cost_usd=_hosted_cost(
                jev_result.input_tokens, len(questions) * 12, self.price_per_mtok
            ),
            cost_estimated=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            calls=1,
        )

    def _as_logprob_mode(
        self, jev_result: DecisionResult, questions: dict[str, Question], t0: float
    ):
        answers = {}
        for qid, q in questions.items():
            a = jev_result.answers.get(qid)
            probs = _synthetic_token_probs(q, a)
            answers[qid] = answer_from_token_probs(q, probs)
        # One call per question, each re-sending the whole email as its prompt -- unlike
        # gen-json's single shared call, the input tokens are paid len(questions) times over.
        n = len(questions)
        input_tokens = (jev_result.input_tokens or 0) * n
        return DecisionResult(
            answers=answers,
            model="mock-gen-logprob-0",
            input_tokens=input_tokens,
            output_tokens=n,
            cost_usd=_hosted_cost(input_tokens, n, self.price_per_mtok),
            cost_estimated=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            calls=n,
        )


def _synthetic_token_probs(q: Question, a: Answer | None) -> dict[str, float]:
    if a is None:
        return {}
    if isinstance(q, Noul):
        p = a.noul if a.noul is not None else 0.5
        return {"true": p, "false": 1 - p}
    if isinstance(q, Choice):
        return dict(a.probabilities or ({a.choice: a.confidence or 0.5} if a.choice else {}))
    if isinstance(q, Score):
        return dict(a.probabilities or {})
    raise TypeError(type(q))

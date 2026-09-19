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

``frontier`` is ``gen-json`` pointed at a hosted frontier model (OpenAI ``gpt-5.6-terra`` by
default) -- the reference row the whole comparison is priced against. Hosted reasoning models
reject ``logprobs``, so the token-probability shape is not available there; only the JSON shape
is compared, at list price.

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

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
PROFILES = ("local", "openai")


class MissingFrontierKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pricing:
    """$ per million tokens. ``flat(x)`` reproduces the old single-price estimate."""

    input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0

    @classmethod
    def flat(cls, price_per_mtok: float) -> Pricing:
        return cls(price_per_mtok, price_per_mtok, price_per_mtok)

    @classmethod
    def frontier(cls, settings: Settings) -> Pricing:
        return cls(
            settings.frontier_price_input_per_mtok,
            settings.frontier_price_cached_input_per_mtok,
            settings.frontier_price_output_per_mtok,
        )

    @property
    def zero(self) -> bool:
        return not (self.input or self.cached_input or self.output)

    def cost(
        self, prompt_tokens: int | None, completion_tokens: int | None, cached_tokens: int | None
    ) -> float:
        if self.zero:
            return 0.0
        prompt = prompt_tokens or 0
        cached = min(cached_tokens or 0, prompt)
        return (
            (prompt - cached) * self.input
            + cached * self.cached_input
            + (completion_tokens or 0) * self.output
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {"input": self.input, "cached_input": self.cached_input, "output": self.output}


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
    content: str | None = None
    top_logprobs: list[dict] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None


class _ChatClient:
    """The one seam that talks to an OpenAI-compatible chat/completions endpoint.

    ``profile="local"`` is llama-server (temperature 0, ``max_tokens``, ``logprobs`` allowed,
    ``reasoning_effort`` via ``chat_template_kwargs``). ``profile="openai"`` is a hosted frontier
    reasoning model: ``max_completion_tokens`` instead of ``max_tokens``, no ``temperature`` or
    ``seed`` (not consistently accepted), ``reasoning_effort`` as a top-level field, and
    ``logprobs`` refused up front because the model does not return them -- the same conventions
    llama-cpp-spark's ``OpenAIEndpoint`` settled on against the same API.
    """

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
        retries: int = 5,
        retry_delay_s: float = 1.0,
        max_retry_delay_s: float = 30.0,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.profile = profile
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.max_retry_delay_s = max_retry_delay_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport, headers=headers
        )

    def close(self) -> None:
        self._client.close()

    def healthy(self) -> bool:
        """``GET /models/{model}`` for a hosted provider (listing all models needs broader
        permissions than reading one), ``GET /models`` for llama-server."""
        url = f"{self.base_url}/models"
        if self.profile == "openai":
            url += f"/{self.model}"
        try:
            r = self._client.get(url)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    def build_body(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: int = 10,
        response_format: dict | None = None,
    ) -> dict:
        body: dict = {"model": self.model, "messages": messages}
        if self.profile == "openai":
            body["max_completion_tokens"] = max_tokens
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = 0
            body["max_tokens"] = max_tokens
            if self.reasoning_effort:
                body["chat_template_kwargs"] = {"reasoning_effort": self.reasoning_effort}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        if response_format is not None:
            body["response_format"] = response_format
        return body

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: int = 10,
        response_format: dict | None = None,
    ) -> _ChatCall:
        if logprobs and self.profile == "openai":
            return _ChatCall(error="logprobs not supported by this provider")
        body = self.build_body(
            messages,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            response_format=response_format,
        )
        t0 = time.perf_counter()
        delay = self.retry_delay_s
        last_error = "no attempt made"
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt >= self.retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, self.max_retry_delay_s)
                continue
            if r.status_code == 200:
                return self._parse(r.json(), (time.perf_counter() - t0) * 1000)
            last_error = f"http {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRYABLE_STATUS or attempt >= self.retries:
                break
            time.sleep(retry_after_seconds(r) or delay)
            delay = min(delay * 2, self.max_retry_delay_s)
        return _ChatCall(latency_ms=(time.perf_counter() - t0) * 1000, error=last_error)

    @staticmethod
    def _parse(data: dict, latency_ms: float) -> _ChatCall:
        usage = data.get("usage", {}) or {}
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return _ChatCall(latency_ms=latency_ms, error="no message content")
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
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            latency_ms=latency_ms,
        )


def local_chat_client(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    reasoning_effort: str | None = None,
) -> _ChatClient | None:
    if not settings.llm_base_url:
        return None
    return _ChatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout_s=settings.timeout_s,
        transport=transport,
        profile="local",
        reasoning_effort=reasoning_effort,
    )


def frontier_chat_client(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    reasoning_effort: str | None = None,
) -> _ChatClient:
    if not settings.openai_api_key:
        raise MissingFrontierKeyError(
            "OPENAI_API_KEY is not set; the frontier backend needs it (put it in .env, the same "
            "variable llama-cpp-spark reads). FRONTIER_MODEL and OPENAI_BASE_URL are optional."
        )
    return _ChatClient(
        base_url=settings.frontier_base_url,
        model=settings.frontier_model,
        api_key=settings.openai_api_key,
        timeout_s=max(settings.timeout_s, 180.0),  # reasoning models think before they answer
        transport=transport,
        profile="openai",
        reasoning_effort=reasoning_effort,
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
    # Local gpt-oss answers in ~150 tokens; a hosted reasoning model spends its budget thinking
    # first and the answer is cut off if the cap is too tight, so the frontier profile gets room.
    max_tokens_by_profile = {"local": 600, "openai": 4000}

    def __init__(self, client: _ChatClient, pricing: Pricing | None = None) -> None:
        self.client = client
        self.pricing = pricing or Pricing()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        pricing: Pricing | None = None,
        transport=None,
        reasoning_effort: str | None = None,
    ):
        client = local_chat_client(settings, transport=transport, reasoning_effort=reasoning_effort)
        return None if client is None else cls(client, pricing=pricing)

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        lines = [f"- {question_prompt(qid, q)}" for qid, q in questions.items()]
        user = "Email:\n" + json.dumps(state) + "\n\nQuestions:\n" + "\n".join(lines)
        call = self.client.complete(
            [{"role": "system", "content": GEN_JSON_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=self.max_tokens_by_profile[self.client.profile],
        )
        if call.error or call.content is None:
            return DecisionResult(
                answers={},
                model=self.client.model,
                error=call.error or "empty message content",
                latency_ms=call.latency_ms,
                calls=1,
                input_tokens=call.prompt_tokens,
                output_tokens=call.completion_tokens,
                cached_input_tokens=call.cached_tokens,
                reasoning_tokens=call.reasoning_tokens,
                cost_usd=self.pricing.cost(
                    call.prompt_tokens, call.completion_tokens, call.cached_tokens
                ),
                cost_estimated=True,
            )
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
            cached_input_tokens=call.cached_tokens,
            reasoning_tokens=call.reasoning_tokens,
            cost_usd=self.pricing.cost(
                call.prompt_tokens, call.completion_tokens, call.cached_tokens
            ),
            cost_estimated=True,
            latency_ms=call.latency_ms,
            calls=1,
            error=error,
        )


class FrontierBackend(GenJsonBackend):
    """``gen-json`` against a hosted frontier model, at list price. Only the JSON shape exists
    here: hosted reasoning models do not return ``logprobs``, so there is no ``frontier-logprob``.
    """

    name = "frontier"

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        pricing: Pricing | None = None,
        transport=None,
        reasoning_effort: str | None = None,
    ):
        client = frontier_chat_client(
            settings, transport=transport, reasoning_effort=reasoning_effort
        )
        return cls(client, pricing=pricing or Pricing.frontier(settings))


class GenLogprobBackend:
    """One constrained, single-token call per question; the probability comes from the model's
    own log-probabilities rather than a number it typed."""

    name = "gen-logprob"

    def __init__(self, client: _ChatClient, pricing: Pricing | None = None) -> None:
        self.client = client
        self.pricing = pricing or Pricing()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        pricing: Pricing | None = None,
        transport=None,
        reasoning_effort: str | None = None,
    ):
        client = local_chat_client(settings, transport=transport, reasoning_effort=reasoning_effort)
        return None if client is None else cls(client, pricing=pricing)

    def decide(self, state: dict[str, str], questions: dict[str, Question]) -> DecisionResult:
        answers: dict[str, Answer] = {}
        errors: list[str] = []
        prompt_tokens = completion_tokens = cached_tokens = 0
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
            cached_tokens += call.cached_tokens or 0
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
            cached_input_tokens=cached_tokens,
            cost_usd=self.pricing.cost(prompt_tokens, completion_tokens, cached_tokens),
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


class MockGenerative:
    """Offline stand-in for both generative modes: no network, but the parsing and mapping code
    it exercises (``parse_json_object``, ``map_json_answer``, ``answer_from_token_probs``) is the
    same code the live backends use.

    Scored with the same keyword heuristics as :class:`~jev_email_cascade.mock_jev.MockJev`, so
    accuracy differences in a mock-only run come from the generative *shape* (one JSON object
    for eight questions vs. one constrained token each), not from a weaker scorer.
    """

    def __init__(self, mode: str = "json", pricing: Pricing | None = None) -> None:
        if mode not in ("json", "logprob"):
            raise ValueError(mode)
        self.mode = mode
        self.pricing = pricing or Pricing()
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
            cost_usd=self.pricing.cost(jev_result.input_tokens, len(questions) * 12, 0),
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
            cost_usd=self.pricing.cost(input_tokens, n, 0),
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

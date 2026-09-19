from __future__ import annotations

import json
import math

import httpx
import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.generative import (
    FrontierBackend,
    GenJsonBackend,
    GenLogprobBackend,
    MissingFrontierKeyError,
    MockGenerative,
    Pricing,
    _ChatClient,
    answer_from_token_probs,
    map_json_answer,
    parse_json_object,
)
from jev_email_cascade.questions import QUESTIONS, Choice, Noul, Score


def _ok_response(content: str = "{}", **usage) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}], "usage": usage}
    )


def test_parse_json_object_strips_fences() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_map_json_answer_noul() -> None:
    q = Noul(instructions="?")
    a = map_json_answer(q, {"answer": True, "confidence": 0.8})
    assert a.type == "noul"
    assert a.noul == 0.8
    a2 = map_json_answer(q, {"answer": "false", "confidence": 0.9})
    assert a2.noul == pytest.approx(0.1)


def test_map_json_answer_choice_rejects_unknown_option() -> None:
    q = Choice(instructions="?", criteria={"a": "x", "b": "y"})
    a = map_json_answer(q, {"answer": "c", "confidence": 0.9})
    assert a.error is not None
    a_ok = map_json_answer(q, {"answer": "a", "confidence": 0.9})
    assert a_ok.choice == "a" and a_ok.error is None


def test_map_json_answer_score_range_checked() -> None:
    q = Score(instructions="?", criteria=["l0", "l1", "l2"])
    a = map_json_answer(q, {"answer": 5, "confidence": 0.5})
    assert a.error is not None
    a_ok = map_json_answer(q, {"answer": 1, "confidence": 0.5})
    assert a_ok.score == 1.0


def test_answer_from_token_probs_noul() -> None:
    a = answer_from_token_probs(Noul(instructions="?"), {"true": 0.9, "false": 0.1})
    assert a.noul == 0.9


def test_answer_from_token_probs_empty_is_error() -> None:
    a = answer_from_token_probs(Noul(instructions="?"), {})
    assert a.error is not None


def test_gen_json_backend_request_contract_and_mapping() -> None:
    reply = {
        qid: {
            "answer": True
            if isinstance(q, Noul)
            else (next(iter(q.criteria)) if isinstance(q, Choice) else 0),
            "confidence": 0.9,
        }
        for qid, q in QUESTIONS.items()
    }

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(reply)}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 120},
            },
        )

    client = _ChatClient(
        base_url="http://x/v1", model="gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    backend = GenJsonBackend(client)
    result = backend.decide({"subject": "s"}, QUESTIONS)

    assert captured["body"]["messages"][1]["content"].startswith("Email:")
    assert result.ok
    assert result.calls == 1
    assert set(result.answers) == set(QUESTIONS)
    assert result.input_tokens == 900


def test_gen_json_backend_records_missing_key_as_per_question_error() -> None:
    reply = {"category": {"answer": "support", "confidence": 0.9}}  # every other key missing

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(reply)}}], "usage": {}}
        )

    client = _ChatClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    backend = GenJsonBackend(client)
    result = backend.decide({}, QUESTIONS)
    assert result.answers["category"].error is None
    assert result.answers["priority"].error == "missing key in generative JSON"


def test_gen_logprob_backend_issues_one_call_per_question() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "yes"},
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": math.log(0.9)},
                                        {"token": "no", "logprob": math.log(0.1)},
                                    ]
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 1},
            },
        )

    client = _ChatClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    backend = GenLogprobBackend(client)
    result = backend.decide({"subject": "s"}, QUESTIONS)

    assert len(calls) == len(QUESTIONS)
    assert result.calls == len(QUESTIONS)
    assert all(c["max_tokens"] == 1 for c in calls)
    assert all(c["logprobs"] is True for c in calls)
    # Every question type maps onto the fixed {"yes","no"} token set here, so only the Noul
    # answers are meaningful; the point of this test is the call count and request shape.
    assert result.answers["awaiting_reply"].noul == 0.9


def test_gen_logprob_missing_token_is_a_per_question_error_not_a_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "maybe"},
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "maybe", "logprob": math.log(0.99)},
                                    ]
                                }
                            ]
                        },
                    }
                ],
                "usage": {},
            },
        )

    client = _ChatClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    backend = GenLogprobBackend(client)
    result = backend.decide({}, {"awaiting_reply": QUESTIONS["awaiting_reply"]})
    assert result.answers["awaiting_reply"].error is not None
    assert not result.ok


def test_mock_generative_json_mode_produces_all_answers() -> None:
    backend = MockGenerative(mode="json")
    result = backend.decide(
        {"subject": "invoice overcharge", "body": "This charge is unacceptable, refund please."},
        QUESTIONS,
    )
    assert set(result.answers) == set(QUESTIONS)
    assert result.calls == 1


def test_mock_generative_logprob_mode_costs_one_call_per_question() -> None:
    backend = MockGenerative(mode="logprob")
    result = backend.decide({"subject": "s", "body": "b"}, QUESTIONS)
    assert result.calls == len(QUESTIONS)


# --- profiles: local llama-server vs hosted OpenAI reasoning model -------------------------


def test_openai_profile_request_body_follows_frontier_conventions() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return _ok_response("{}", prompt_tokens=10, completion_tokens=5)

    client = _ChatClient(
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-terra",
        api_key="sk-test",
        profile="openai",
        reasoning_effort="high",
        transport=httpx.MockTransport(handler),
    )
    client.complete([{"role": "user", "content": "hi"}], max_tokens=123)

    body = captured["body"]
    assert body["model"] == "gpt-5.6-terra"
    assert body["max_completion_tokens"] == 123
    assert "max_tokens" not in body
    assert "temperature" not in body and "seed" not in body
    assert body["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in body
    assert captured["headers"]["authorization"] == "Bearer sk-test"


def test_local_profile_keeps_llama_server_body_and_maps_reasoning_effort() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response("{}")

    client = _ChatClient(
        base_url="http://127.0.0.1:8082/v1",
        model="gpt-oss-120b",
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
    )
    client.complete([{"role": "user", "content": "hi"}], max_tokens=50)

    body = captured["body"]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 50
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_local_profile_without_reasoning_effort_sends_no_template_kwargs() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response("{}")

    client = _ChatClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert "chat_template_kwargs" not in captured["body"]


def test_openai_profile_refuses_logprobs_without_an_http_call() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok_response("yes")

    client = _ChatClient(
        base_url="http://x/v1", model="m", profile="openai", transport=httpx.MockTransport(handler)
    )
    backend = GenLogprobBackend(client)
    result = backend.decide({"subject": "s"}, QUESTIONS)

    assert calls == []
    assert not result.ok
    assert all("logprobs not supported" in (a.error or "") for a in result.answers.values())


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError):
        _ChatClient(base_url="http://x/v1", model="m", profile="anthropic")


def test_healthy_uses_per_model_path_for_openai_and_list_for_local() -> None:
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    _ChatClient(
        base_url="http://x/v1",
        model="gpt-5.6-terra",
        profile="openai",
        transport=httpx.MockTransport(handler),
    ).healthy()
    _ChatClient(
        base_url="http://x/v1", model="gpt-oss-120b", transport=httpx.MockTransport(handler)
    ).healthy()
    assert paths == ["/v1/models/gpt-5.6-terra", "/v1/models"]


# --- retries, usage parsing, pricing ----------------------------------------------------------


def test_complete_retries_on_429_honouring_retry_after() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0"})
        return _ok_response('{"a": 1}', prompt_tokens=3, completion_tokens=2)

    client = _ChatClient(
        base_url="http://x/v1", model="m", retry_delay_s=0, transport=httpx.MockTransport(handler)
    )
    call = client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert len(attempts) == 2
    assert call.error is None
    assert call.content == '{"a": 1}'


def test_complete_does_not_retry_on_4xx_other_than_429() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, text="bad request")

    client = _ChatClient(
        base_url="http://x/v1", model="m", retry_delay_s=0, transport=httpx.MockTransport(handler)
    )
    call = client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert len(attempts) == 1
    assert "400" in (call.error or "")


def test_complete_gives_up_after_retries_exhausted() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, text="unavailable")

    client = _ChatClient(
        base_url="http://x/v1",
        model="m",
        retries=2,
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )
    call = client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert len(attempts) == 3
    assert "503" in (call.error or "")


def test_complete_parses_cached_and_reasoning_token_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 400,
                    "prompt_tokens_details": {"cached_tokens": 600},
                    "completion_tokens_details": {"reasoning_tokens": 350},
                },
            },
        )

    client = _ChatClient(
        base_url="http://x/v1", model="m", profile="openai", transport=httpx.MockTransport(handler)
    )
    call = client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert call.prompt_tokens == 1000
    assert call.completion_tokens == 400
    assert call.cached_tokens == 600
    assert call.reasoning_tokens == 350


def test_pricing_math_splits_cached_input_and_output() -> None:
    p = Pricing(input=2.0, cached_input=0.20, output=12.0)
    # 1000 prompt of which 600 cached, 400 completion:
    #   400 * 2 + 600 * 0.2 + 400 * 12 = 800 + 120 + 4800 = 5720 per M -> $0.00572
    assert p.cost(1000, 400, 600) == pytest.approx(0.00572)
    assert p.cost(None, None, None) == 0.0
    # cached can never exceed prompt (defensive against odd usage payloads)
    assert p.cost(100, 0, 500) == pytest.approx(100 * 0.20 / 1e6)


def test_pricing_flat_matches_old_single_price_estimate() -> None:
    p = Pricing.flat(1.5)
    assert p.cost(900, 100, 0) == pytest.approx(1000 * 1.5 / 1e6)
    assert Pricing().zero and not p.zero


def test_frontier_backend_requires_openai_api_key() -> None:
    with pytest.raises(MissingFrontierKeyError, match="OPENAI_API_KEY"):
        FrontierBackend.from_settings(Settings(openai_api_key=None))


def test_frontier_backend_from_settings_uses_openai_profile_and_list_prices() -> None:
    settings = Settings(
        openai_api_key="sk-test",
        frontier_model="gpt-5.6-terra",
        frontier_price_input_per_mtok=2.0,
        frontier_price_cached_input_per_mtok=0.2,
        frontier_price_output_per_mtok=12.0,
    )
    backend = FrontierBackend.from_settings(settings, reasoning_effort="medium")
    assert backend.name == "frontier"
    assert backend.client.profile == "openai"
    assert backend.client.model == "gpt-5.6-terra"
    assert backend.client.reasoning_effort == "medium"
    assert backend.pricing == Pricing(2.0, 0.2, 12.0)


def test_frontier_backend_decide_records_reasoning_tokens_and_list_price_cost() -> None:
    reply = {
        qid: {
            "answer": True
            if isinstance(q, Noul)
            else (next(iter(q.criteria)) if isinstance(q, Choice) else 0),
            "confidence": 0.9,
        }
        for qid, q in QUESTIONS.items()
    }
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(reply)}}],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 300,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 250},
                },
            },
        )

    settings = Settings(openai_api_key="sk-test")
    backend = FrontierBackend.from_settings(settings, transport=httpx.MockTransport(handler))
    result = backend.decide({"subject": "s"}, QUESTIONS)

    assert result.ok
    assert result.model == "gpt-5.6-terra"
    assert result.reasoning_tokens == 250
    assert result.cached_input_tokens == 0
    assert result.cost_usd == pytest.approx((900 * 2.0 + 300 * 12.0) / 1e6)
    assert result.cost_estimated is True
    assert (
        captured["body"]["max_completion_tokens"] == GenJsonBackend.max_tokens_by_profile["openai"]
    )
    assert set(result.answers) == set(QUESTIONS)


def test_gen_json_backend_keeps_usage_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _ChatClient(
        base_url="http://x/v1", model="m", retries=0, transport=httpx.MockTransport(handler)
    )
    result = GenJsonBackend(client, pricing=Pricing.flat(1.0)).decide({}, QUESTIONS)
    assert not result.ok
    assert result.calls == 1
    assert result.cost_usd == 0.0

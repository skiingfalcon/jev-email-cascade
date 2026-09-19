from __future__ import annotations

import json
import math

import httpx
import pytest

from jev_email_cascade.generative import (
    GenJsonBackend,
    GenLogprobBackend,
    MockGenerative,
    _ChatClient,
    answer_from_token_probs,
    map_json_answer,
    parse_json_object,
)
from jev_email_cascade.questions import QUESTIONS, Choice, Noul, Score


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

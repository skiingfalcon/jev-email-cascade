from __future__ import annotations

import json

import httpx
import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.laya_client import LayaClient
from jev_email_cascade.pipeline import build_backend
from jev_email_cascade.questions import Choice, Noul, Score

URL = "http://127.0.0.1:8000/v1/systemone"
STATE = {"subject": "Duplicate charge", "from": "a@b.c", "received": "2026-09-01", "body": "refund"}


def _questions() -> dict:
    return {
        "is_urgent": Noul(instructions="urgent?"),
        "department": Choice(instructions="which team?", criteria={"billing": "b", "other": "o"}),
        "priority": Score(instructions="how urgent?", criteria=["low", "high"]),
    }


def _laya_response() -> dict:
    # Trimmed from a real laya-serve 0.3.26 reply: Jev-shaped answers plus Laya's `routing`.
    return {
        "model": "convaiinnovations/laya",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.81, "confidence": 0.62},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.93, "other": 0.07},
                "confidence": 0.86,
            },
            "priority": {"type": "score", "score": 0.7, "probabilities": {"0": 0.3, "1": 0.7}},
        },
        "usage": {"input_tokens": 212, "output_tokens": 0},
        "routing": {"model": "english", "reason": "default"},
    }


def _settings(**kw) -> Settings:
    base = {"runs_dir": "runs", "laya_api_key": None, "laya_model": None}
    base.update(kw)
    return Settings(**base)


def test_request_contract_raw_state_no_auth() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_laya_response())

    client = LayaClient(url=URL, transport=httpx.MockTransport(handler))
    result = client.decide(STATE, _questions())

    assert captured["url"] == URL
    assert captured["auth"] is None
    body = captured["body"]
    assert body["state"] == STATE
    assert "model" not in body and "max_len" not in body
    assert body["questions"]["priority"]["criteria"] == ["low", "high"]
    assert body["questions"]["department"]["criteria"] == {"billing": "b", "other": "o"}

    assert result.ok
    assert result.model == "english"
    assert result.answers["department"].choice == "billing"
    assert result.answers["department"].probabilities == {"billing": 0.93, "other": 0.07}
    assert result.answers["priority"].probabilities == {"0": 0.3, "1": 0.7}
    assert result.answers["is_urgent"].noul == 0.81
    assert result.input_tokens == 212
    assert result.cost_usd == 0.0


def test_rendered_state_model_max_len_and_key() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_laya_response())

    client = LayaClient(
        url=URL,
        model="typed-decisions",
        api_key="secret",
        state_mode="rendered",
        max_len=1024,
        transport=httpx.MockTransport(handler),
    )
    client.decide(STATE, _questions())

    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["model"] == "typed-decisions"
    assert captured["body"]["max_len"] == 1024
    rendered = captured["body"]["state"]["body"]
    assert rendered.startswith("Subject: Duplicate charge\nFrom: a@b.c\nReceived: 2026-09-01")
    assert rendered.endswith("\n\nrefund")


def test_retries_503_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"detail": "server busy"})
        return httpx.Response(200, json=_laya_response())

    client = LayaClient(url=URL, transport=httpx.MockTransport(handler), retry_delay_s=0)
    result = client.decide(STATE, _questions())
    assert result.ok
    assert calls["n"] == 2


def test_422_is_not_retried_and_reported() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "question 'priority': bad criteria"})

    client = LayaClient(url=URL, transport=httpx.MockTransport(handler), retry_delay_s=0)
    result = client.decide(STATE, _questions())
    assert not result.ok
    assert result.error.startswith("http 422")
    assert calls["n"] == 1


def test_health_and_backend_info() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/health":
            return httpx.Response(404)  # laya-serve has no /props
        return httpx.Response(200, json={"status": "ok", "device": "cpu"})

    client = LayaClient(url=URL, transport=httpx.MockTransport(handler))
    assert client.healthy()
    info = client.backend_info()
    assert info["server"]["device"] == "cpu"
    assert "props" not in info
    assert info["state_mode"] == "raw"


def test_unhealthy_when_server_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = LayaClient(url=URL, transport=httpx.MockTransport(handler))
    assert not client.healthy()


def test_bad_state_mode_rejected() -> None:
    with pytest.raises(ValueError):
        LayaClient(url=URL, state_mode="json")


def test_build_backend_from_settings() -> None:
    backend = build_backend(
        "laya", _settings(laya_model="multilingual", laya_state_mode="rendered")
    )
    assert isinstance(backend, LayaClient)
    assert backend.model == "multilingual"
    assert backend.state_mode == "rendered"
    assert backend.base_url == "http://127.0.0.1:8000"


def test_build_rune_backend_from_settings() -> None:
    backend = build_backend("rune", _settings(rune_api_key=None))
    assert isinstance(backend, LayaClient)
    assert backend.name == "rune"
    assert backend.url == "http://127.0.0.1:8001/v1/systemone"
    assert backend.base_url == "http://127.0.0.1:8001"
    assert backend.model is None


def test_llama_server_response_without_routing_or_model() -> None:
    # llama-server's /v1/systemone (b11382) returns answers and usage only.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "model" not in body
        return httpx.Response(
            200,
            json={
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.98, "other": 0.02},
                        "confidence": 0.96,
                    },
                    "is_urgent": {"type": "noul", "noul": 0.12},
                },
                "usage": {"input_tokens": 619, "output_tokens": 0},
            },
        )

    client = LayaClient(
        url="http://127.0.0.1:8001/v1/systemone",
        name="rune",
        transport=httpx.MockTransport(handler),
    )
    result = client.decide(STATE, _questions())
    assert result.ok
    assert result.model == "rune"
    assert result.answers["department"].choice == "billing"
    assert result.input_tokens == 619


def test_backend_info_records_llama_server_props() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "model_path": "Rune-26B-A4B-v3-Q8_0.gguf",
                    "build_info": "b11382-11fe0215",
                    "total_slots": 1,
                    "chat_template": "...long...",
                },
            )
        return httpx.Response(404)

    client = LayaClient(
        url="http://127.0.0.1:8001/v1/systemone",
        name="rune",
        transport=httpx.MockTransport(handler),
    )
    info = client.backend_info()
    assert info["props"] == {
        "model_path": "Rune-26B-A4B-v3-Q8_0.gguf",
        "build_info": "b11382-11fe0215",
        "total_slots": 1,
    }


def test_unknown_server_rejected() -> None:
    with pytest.raises(ValueError):
        LayaClient.from_settings(_settings(), server="jev")

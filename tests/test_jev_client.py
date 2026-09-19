from __future__ import annotations

import json

import httpx
import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.jev_client import JevClient, MissingKeyError
from jev_email_cascade.questions import Choice, Noul, Score


def _questions() -> dict:
    return {
        "is_urgent": Noul(instructions="urgent?"),
        "department": Choice(instructions="which team?", criteria={"billing": "b", "other": "o"}),
        "priority": Score(instructions="how urgent?", criteria=["low", "high"]),
    }


def _settings(**kw) -> Settings:
    # Explicit None for both keys unless overridden, so a real .env some day cannot leak into
    # these tests: pydantic-settings gives constructor kwargs priority over the dotenv source.
    base = {"runs_dir": "runs", "openrouter_api_key": None, "typesafe_api_key": None}
    base.update(kw)
    return Settings(**base)


def test_from_settings_requires_a_key() -> None:
    with pytest.raises(MissingKeyError):
        JevClient.from_settings(_settings())


def test_typesafe_takes_precedence_when_both_keys_set() -> None:
    settings = _settings(openrouter_api_key="or-key", typesafe_api_key="ts-key")
    client = JevClient.from_settings(settings)
    assert client.route == "typesafe"
    assert client.url == settings.typesafe_systemone_url
    assert client.model == settings.typesafe_model


def test_openrouter_used_when_only_that_key_set() -> None:
    settings = _settings(openrouter_api_key="or-key")
    client = JevClient.from_settings(settings)
    assert client.route == "openrouter"
    assert client.url == settings.jev_decisions_url


def test_request_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"is_urgent": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 4.2e-6},
            },
        )

    transport = httpx.MockTransport(handler)
    client = JevClient(
        url="https://openrouter.ai/api/alpha/decisions",
        api_key="k",
        model="typesafe/jev-1.13",
        route="openrouter",
        transport=transport,
    )
    result = client.decide({"subject": "hi"}, _questions())

    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "typesafe/jev-1.13"
    assert captured["body"]["state"] == {"subject": "hi"}
    assert set(captured["body"]["questions"]) == {"is_urgent", "department", "priority"}
    assert captured["body"]["questions"]["department"]["type"] == "choice"
    assert captured["body"]["questions"]["department"]["criteria"] == {"billing": "b", "other": "o"}

    assert result.ok
    assert result.model == "typesafe/jev-1.13-20260917"
    assert result.answers["is_urgent"].noul == 0.9
    assert result.input_tokens == 100
    assert result.cost_usd == 4.2e-6
    assert result.cost_estimated is False


def test_parses_string_keyed_probabilities_and_missing_confidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13",
                "answers": {
                    "is_urgent": {
                        "type": "noul",
                        "noul": 0.7,
                        "probabilities": {"true": 0.7, "false": 0.3},
                    }
                },
                "usage": {"input_tokens": 50, "output_tokens": 3},
            },
        )

    client = JevClient(
        url="https://x/decisions",
        api_key="k",
        model="m",
        route="openrouter",
        transport=httpx.MockTransport(handler),
    )
    result = client.decide({}, {"is_urgent": Noul(instructions="?")})
    answer = result.answers["is_urgent"]
    assert answer.confidence is None
    assert answer.probabilities == {"true": 0.7, "false": 0.3}
    # cost.usd absent from usage -> estimated from the published $0.042/M rate
    assert result.cost_estimated is True
    assert result.cost_usd == pytest.approx(50 * 0.042 / 1_000_000)


def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "slow down"})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {"is_urgent": {"type": "noul", "noul": 0.5}},
                "usage": {},
            },
        )

    client = JevClient(
        url="https://x/decisions",
        api_key="k",
        model="m",
        route="openrouter",
        transport=httpx.MockTransport(handler),
        retry_delay_s=0.01,
    )
    result = client.decide({}, {"is_urgent": Noul(instructions="?")})
    assert len(calls) == 2
    assert sleeps == [2.0]
    assert result.ok


def test_gives_up_after_retries_exhausted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = JevClient(
        url="https://x/decisions",
        api_key="k",
        model="m",
        route="openrouter",
        transport=httpx.MockTransport(handler),
        retries=1,
        retry_delay_s=0.0,
    )
    result = client.decide({}, {"is_urgent": Noul(instructions="?")})
    assert not result.ok
    assert "http 500" in (result.error or "")


def test_non_retryable_status_stops_immediately() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422, json={"error": "bad question"})

    client = JevClient(
        url="https://x/decisions",
        api_key="k",
        model="m",
        route="openrouter",
        transport=httpx.MockTransport(handler),
        retries=3,
        retry_delay_s=0.0,
    )
    result = client.decide({}, {"is_urgent": Noul(instructions="?")})
    assert len(calls) == 1
    assert not result.ok
    assert "422" in (result.error or "")

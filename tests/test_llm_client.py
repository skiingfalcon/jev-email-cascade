from __future__ import annotations

import json

import httpx
import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.generative import MissingFrontierKeyError
from jev_email_cascade.llm_client import LlmClient, _parse_json_object
from jev_email_cascade.policy import Decision, Route


def _decision() -> Decision:
    return Decision(
        category=None,
        category_conf=0.4,
        priority=1,
        priority_conf=0.7,
        flags={"needs_decision": True},
        uncertain=["category_uncertain"],
        reasons=["needs_decision"],
        route=Route.LLM,
    )


def test_from_settings_none_when_unset() -> None:
    assert LlmClient.from_settings(Settings(llm_base_url=None)) is None


def test_from_settings_builds_client_when_set() -> None:
    client = LlmClient.from_settings(Settings(llm_base_url="http://spark:8082/v1"))
    assert client is not None
    assert client.base_url == "http://spark:8082/v1"


def test_healthy_true_on_200() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    client = LlmClient(base_url="http://x/v1", model="m", transport=transport)
    assert client.healthy() is True


def test_healthy_false_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = LlmClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    assert client.healthy() is False


def test_triage_parses_plain_json() -> None:
    payload = {
        "category": "billing",
        "priority": 2,
        "rationale": "r",
        "summary": "s",
        "entities": {"deadline": None, "amount": "100", "party": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-oss-120b"
        assert body["temperature"] == 0
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 40},
            },
        )

    client = LlmClient(
        base_url="http://x/v1", model="gpt-oss-120b", transport=httpx.MockTransport(handler)
    )
    result = client.triage({"subject": "s", "body": "b"}, _decision())
    assert result.ok
    assert result.category == "billing"
    assert result.priority == 2
    assert result.prompt_tokens == 300


def test_triage_parses_fenced_json() -> None:
    content = "```json\n" + json.dumps({"category": "support", "priority": 1}) + "\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {}}
        )

    client = LlmClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    result = client.triage({}, _decision())
    assert result.ok
    assert result.category == "support"


def test_triage_error_on_unparseable_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]})

    client = LlmClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    result = client.triage({}, _decision())
    assert not result.ok


def test_triage_http_error_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = LlmClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))
    result = client.triage({}, _decision())
    assert not result.ok
    assert "500" in (result.error or "")


def test_parse_json_object_strips_fences() -> None:
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_object_finds_embedded_object() -> None:
    assert _parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}


def test_parse_json_object_none_on_garbage() -> None:
    assert _parse_json_object("no object here") is None


# --- LLM_PROVIDER=frontier ----------------------------------------------------------------------


def test_from_settings_frontier_provider_builds_openai_profile_client() -> None:
    settings = Settings(
        llm_provider="frontier",
        openai_api_key="sk-test",
        frontier_model="gpt-5.6-terra",
        llm_base_url=None,  # the local URL is irrelevant to the frontier provider
    )
    client = LlmClient.from_settings(settings, reasoning_effort="low")
    assert client is not None
    assert client.provider == "frontier"
    assert client.profile == "openai"
    assert client.model == "gpt-5.6-terra"
    assert client.base_url == "https://api.openai.com/v1"
    assert client.pricing.input == 2.0


def test_from_settings_frontier_provider_without_key_raises() -> None:
    with pytest.raises(MissingFrontierKeyError):
        LlmClient.from_settings(Settings(llm_provider="frontier", openai_api_key=None))


def test_from_settings_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LlmClient.from_settings(Settings(llm_provider="anthropic"))


def test_frontier_triage_body_health_path_and_cost() -> None:
    seen = []
    payload = {"category": "billing", "priority": 2, "rationale": "r", "summary": "s"}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "gpt-5.6-terra"})
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == LlmClient.max_tokens_by_profile["openai"]
        assert "temperature" not in body and "max_tokens" not in body
        assert body["reasoning_effort"] == "high"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "completion_tokens_details": {"reasoning_tokens": 420},
                },
            },
        )

    settings = Settings(llm_provider="frontier", openai_api_key="sk-test")
    client = LlmClient.from_settings(
        settings, transport=httpx.MockTransport(handler), reasoning_effort="high"
    )
    assert client is not None
    assert client.healthy() is True
    result = client.triage({"subject": "s"}, _decision())

    assert seen[0] == ("GET", "/v1/models/gpt-5.6-terra")
    assert result.ok
    assert result.category == "billing"
    assert result.reasoning_tokens == 420
    assert result.cost_usd == pytest.approx((1000 * 2.0 + 500 * 12.0) / 1e6)
    assert result.as_dict()["cost_usd"] == result.cost_usd


def test_local_triage_cost_is_zero_and_body_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0 and body["max_tokens"] == 400
        assert "reasoning_effort" not in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"category": "hr"})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    client = LlmClient.from_settings(
        Settings(llm_base_url="http://127.0.0.1:8082/v1"), transport=httpx.MockTransport(handler)
    )
    assert client is not None and client.provider == "local"
    result = client.triage({}, _decision())
    assert result.ok and result.cost_usd == 0.0 and result.reasoning_tokens is None

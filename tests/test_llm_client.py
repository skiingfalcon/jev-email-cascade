from __future__ import annotations

import json

import httpx

from jev_email_cascade.config import Settings
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

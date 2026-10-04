"""Against the real local decision-model servers (local-decision-model: laya-serve on LAYA_URL,
Rune's llama-server on RUNE_URL). Deselected by default; run with `make test-live`. Each server
is skipped if it is not answering."""

from __future__ import annotations

import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.laya_client import SERVERS, LayaClient
from jev_email_cascade.policy import decide_result
from jev_email_cascade.questions import QUESTIONS

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", params=SERVERS)
def client(request: pytest.FixtureRequest) -> LayaClient:
    c = LayaClient.from_settings(Settings(), server=request.param)
    if not c.healthy():
        pytest.skip(f"{request.param} server not answering at {c.base_url}/health")
    return c


def test_health(client: LayaClient) -> None:
    health = client.health()
    assert health["status"] == "ok"
    if client.name == "laya":
        assert health["loaded"]
        assert health["checkpoint_devices"]


def test_full_question_set_round_trip(client: LayaClient) -> None:
    state = {
        "subject": "Invoice INV-2231 charged twice",
        "from": "ap@customer.example",
        "received": "2026-09-30T09:12:00Z",
        "body": "Hi, we were billed twice for invoice INV-2231 this month. Please refund the "
        "duplicate charge by Friday or we will have to escalate.",
    }
    result = client.decide(state, QUESTIONS)
    assert result.ok, result.error
    assert set(result.answers) == set(QUESTIONS)
    assert result.answers["category"].choice == "billing"
    assert 0.0 <= result.answers["priority"].score <= 3.0
    for qid, answer in result.answers.items():
        if answer.type == "noul":
            assert 0.0 <= answer.noul <= 1.0, qid
    # The policy layer accepts the answers unchanged.
    decision = decide_result(result, llm_available=False)
    assert decision.category == "billing"

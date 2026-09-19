from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from jev_email_cascade.config import Settings
from jev_email_cascade.pipeline import run
from jev_email_cascade.policy import Route
from jev_email_cascade.report import compute_metrics, load_run

DATA = Path(__file__).resolve().parents[1] / "data" / "emails.jsonl"


def _settings(tmp_path: Path, **kw) -> Settings:
    base = {
        "runs_dir": tmp_path / "runs",
        "openrouter_api_key": None,
        "typesafe_api_key": None,
        "llm_base_url": None,
        "openai_api_key": None,
    }
    base.update(kw)
    return Settings(**base)


def _healthy_llm_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": []})
    payload = {
        "category": "other",
        "priority": 0,
        "rationale": "r",
        "summary": "s",
        "entities": {"deadline": None, "amount": None, "party": None},
    }
    return httpx.Response(
        200, json={"choices": [{"message": {"content": json.dumps(payload)}}], "usage": {}}
    )


def test_mock_jev_end_to_end_writes_artifacts_and_scores_well(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_dir = run(settings, source=DATA, backend_name="mock-jev")

    assert (run_dir / "run.json").is_file()
    assert (run_dir / "results.jsonl").is_file()
    run_json, rows = load_run(run_dir)
    assert run_json["n"] == len(rows) > 0
    assert run_json["backend"] == "mock-jev"

    metrics = compute_metrics(run_json, rows)
    assert metrics["category"]["accuracy"] > 0.8


def test_without_llm_configured_every_escalation_goes_to_review(tmp_path: Path) -> None:
    settings = _settings(tmp_path)  # llm_base_url unset
    run_dir = run(settings, source=DATA, backend_name="mock-jev", use_llm=True)
    _, rows = load_run(run_dir)

    routes = {r["decision"]["route"] for r in rows}
    assert routes <= {Route.AUTO.value, Route.REVIEW.value}
    assert Route.REVIEW.value in routes  # the injection specials always escalate
    assert all(r["llm"] is None for r in rows)


def test_with_llm_configured_all_three_routes_appear(tmp_path: Path) -> None:
    settings = _settings(tmp_path, llm_base_url="http://spark:8082/v1")
    run_dir = run(
        settings,
        source=DATA,
        backend_name="mock-jev",
        use_llm=True,
        llm_transport=httpx.MockTransport(_healthy_llm_handler),
    )
    _, rows = load_run(run_dir)

    routes = {r["decision"]["route"] for r in rows}
    assert routes == {Route.AUTO.value, Route.REVIEW.value, Route.LLM.value}

    llm_rows = [r for r in rows if r["decision"]["route"] == Route.LLM.value]
    assert llm_rows and all(r["llm"] is not None and r["llm"]["error"] is None for r in llm_rows)

    # injection_suspected forces review even though the LLM is healthy and available
    review_rows = [r for r in rows if r["decision"]["route"] == Route.REVIEW.value]
    assert review_rows and all(r["decision"]["flags"]["injection_suspected"] for r in review_rows)


def test_unhealthy_llm_degrades_every_escalation_to_review(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    settings = _settings(tmp_path, llm_base_url="http://unreachable:8082/v1")
    run_dir = run(
        settings,
        source=DATA,
        backend_name="mock-jev",
        use_llm=True,
        llm_transport=httpx.MockTransport(handler),
    )
    _, rows = load_run(run_dir)
    routes = {r["decision"]["route"] for r in rows}
    assert Route.LLM.value not in routes
    assert all(r["llm"] is None for r in rows)


def test_limit_and_parallel_options(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_dir = run(settings, source=DATA, backend_name="mock-jev", limit=5, parallel=3)
    _, rows = load_run(run_dir)
    assert len(rows) == 5


def test_run_dirs_do_not_collide(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    d1 = run(settings, source=DATA, backend_name="mock-jev", limit=1)
    d2 = run(settings, source=DATA, backend_name="mock-jev", limit=1)
    assert d1 != d2


# --- frontier backend ---------------------------------------------------------------------------


def _frontier_reply() -> dict:
    from jev_email_cascade.questions import QUESTIONS, Choice, Noul

    return {
        qid: {
            "answer": False if isinstance(q, Noul) else ("other" if isinstance(q, Choice) else 0),
            "confidence": 0.95,
        }
        for qid, q in QUESTIONS.items()
    }


def _frontier_handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        assert request.url.path == "/v1/models/gpt-5.6-terra"
        return httpx.Response(200, json={"id": "gpt-5.6-terra"})
    body = json.loads(request.content)
    assert body["reasoning_effort"] == "low"
    assert "max_completion_tokens" in body and "temperature" not in body
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(_frontier_reply())}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 100},
                "completion_tokens_details": {"reasoning_tokens": 150},
            },
        },
    )


def test_frontier_without_key_is_a_clean_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run(settings, source=DATA, backend_name="frontier", limit=1)


def test_frontier_preflight_failure_stops_before_any_completion(tmp_path: Path) -> None:
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    settings = _settings(tmp_path, openai_api_key="sk-bad")
    with pytest.raises(RuntimeError, match="preflight"):
        run(
            settings,
            source=DATA,
            backend_name="frontier",
            limit=2,
            transport=httpx.MockTransport(handler),
        )
    assert posts == []


def test_frontier_end_to_end_records_pricing_tokens_and_effort(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="sk-test")
    run_dir = run(
        settings,
        source=DATA,
        backend_name="frontier",
        limit=3,
        reasoning_effort="low",
        transport=httpx.MockTransport(_frontier_handler),
    )
    run_json, rows = load_run(run_dir)

    assert run_json["backend"] == "frontier"
    assert run_json["model"] == "gpt-5.6-terra"
    assert run_json["reasoning_effort"] == "low"
    assert run_json["pricing"] == {"input": 2.0, "cached_input": 0.2, "output": 12.0}
    assert run_json["input_tokens"] == 3000
    assert run_json["output_tokens"] == 600
    assert run_json["cached_input_tokens"] == 300
    assert run_json["reasoning_tokens"] == 450
    per_email = (900 * 2.0 + 100 * 0.2 + 200 * 12.0) / 1e6
    assert run_json["cost_usd"] == pytest.approx(3 * per_email, abs=1e-6)
    assert run_json["calls_per_email"] == 1.0
    assert run_json["llm_provider"] is None
    assert all(r["reasoning_tokens"] == 150 for r in rows)
    assert all(r["cost_estimated"] for r in rows)


def test_invalid_reasoning_effort_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        run(_settings(tmp_path), source=DATA, backend_name="mock-jev", reasoning_effort="max")


def test_llm_provider_frontier_hook_records_provider_and_cost(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/v1/models/gpt-5.6-terra"
            return httpx.Response(200, json={})
        payload = {"category": "other", "priority": 0, "rationale": "r", "summary": "s"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 80},
                },
            },
        )

    settings = _settings(tmp_path, llm_provider="frontier", openai_api_key="sk-test")
    run_dir = run(
        settings,
        source=DATA,
        backend_name="mock-jev",
        use_llm=True,
        llm_transport=httpx.MockTransport(handler),
    )
    run_json, rows = load_run(run_dir)

    llm_rows = [r for r in rows if r["decision"]["route"] == Route.LLM.value]
    assert llm_rows
    assert run_json["llm_provider"] == "frontier"
    assert run_json["llm_model"] == "gpt-5.6-terra"
    assert run_json["llm_calls"] == len(llm_rows)
    assert run_json["llm_reasoning_tokens"] == 80 * len(llm_rows)
    assert run_json["llm_cost_usd"] == pytest.approx(
        len(llm_rows) * (500 * 2.0 + 100 * 12.0) / 1e6, abs=1e-6
    )
    assert all(r["llm"]["cost_usd"] > 0 for r in llm_rows)


def test_run_json_for_local_backends_has_null_pricing_and_provider(tmp_path: Path) -> None:
    run_dir = run(_settings(tmp_path), source=DATA, backend_name="mock-gen", limit=2)
    run_json, _ = load_run(run_dir)
    assert run_json["pricing"] is None
    assert run_json["reasoning_effort"] is None
    assert run_json["llm_provider"] is None
    assert run_json["reasoning_tokens"] is None

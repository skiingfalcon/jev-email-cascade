from __future__ import annotations

import json
from pathlib import Path

import httpx

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

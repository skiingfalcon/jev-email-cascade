"""Wires one backend to the email set and writes runs/<stamp>/{run.json,results.jsonl}."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from jev_email_cascade.backends import DecisionBackend
from jev_email_cascade.config import Settings
from jev_email_cascade.generative import GenJsonBackend, GenLogprobBackend, MockGenerative
from jev_email_cascade.jev_client import JevClient
from jev_email_cascade.llm_client import LlmClient
from jev_email_cascade.mock_jev import MockJev
from jev_email_cascade.policy import Route, decide_result
from jev_email_cascade.prepare import Prepared, load_emails, prepare
from jev_email_cascade.questions import QUESTIONS, questions_json
from jev_email_cascade.stats import percentile

BACKEND_NAMES = ("jev", "mock-jev", "gen-json", "gen-logprob", "mock-gen")


def build_backend(
    name: str,
    settings: Settings,
    *,
    price_per_mtok: float = 0.0,
    transport: httpx.BaseTransport | None = None,
) -> DecisionBackend:
    if name == "jev":
        return JevClient.from_settings(settings, transport=transport)
    if name == "mock-jev":
        return MockJev()
    if name == "gen-json":
        backend = GenJsonBackend.from_settings(
            settings, price_per_mtok=price_per_mtok, transport=transport
        )
        if backend is None:
            raise RuntimeError("LLM_BASE_URL is not set; cannot use --backend gen-json")
        return backend
    if name == "gen-logprob":
        backend = GenLogprobBackend.from_settings(
            settings, price_per_mtok=price_per_mtok, transport=transport
        )
        if backend is None:
            raise RuntimeError("LLM_BASE_URL is not set; cannot use --backend gen-logprob")
        return backend
    if name == "mock-gen":
        return MockGenerative(mode="json", price_per_mtok=price_per_mtok)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")


def run_parallel(items: list, fn, parallel: int = 1) -> list:
    if parallel <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(fn, items))


def _new_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = runs_dir / stamp
    d, n = base, 1
    while d.exists():
        n += 1
        d = base.with_name(f"{base.name}-{n}")
    d.mkdir(parents=True)
    return d


def _questions_hash() -> str:
    blob = json.dumps(questions_json(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _process_one(
    prepared: Prepared, backend: DecisionBackend, llm_client, llm_available: bool
) -> dict:
    result = backend.decide(prepared.state, QUESTIONS)
    decision = decide_result(result, llm_available=llm_available)
    llm_result = None
    if decision.route is Route.LLM and llm_client is not None:
        llm_result = llm_client.triage(prepared.state, decision)
        if not llm_result.ok:
            decision.route = Route.REVIEW
            decision.reasons.append(f"llm_error: {llm_result.error}")
    return {
        "id": prepared.email.id,
        "labels": prepared.email.labels,
        "state_chars": len(prepared.state.get("body", "")),
        "truncated": prepared.truncated,
        **result.as_dict(),
        "decision": {
            "category": decision.category,
            "category_conf": decision.category_conf,
            "priority": decision.priority,
            "priority_conf": decision.priority_conf,
            "flags": decision.flags,
            "uncertain": decision.uncertain,
            "reasons": decision.reasons,
            "route": decision.route.value,
        },
        "llm": llm_result.as_dict() if llm_result is not None else None,
    }


def _summarise(
    rows: list[dict],
    *,
    backend_name: str,
    settings: Settings,
    llm_available: bool,
    started: datetime,
    finished: datetime,
) -> dict:
    n = len(rows)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    calls = [r.get("calls", 1) for r in rows]
    latencies = [r.get("latency_ms", 0.0) for r in rows]
    llm_latencies = [r["llm"]["latency_ms"] for r in rows if r.get("llm")]
    routes = Counter(r["decision"]["route"] for r in rows)
    errors = sum(1 for r in rows if r.get("error"))
    models = sorted({r["model"] for r in rows if r.get("model")})
    return {
        "backend": backend_name,
        "model": models[0] if len(models) == 1 else models,
        "llm_model": settings.llm_model if llm_available else None,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "wall_s": round((finished - started).total_seconds(), 3),
        "n": n,
        "cost_usd": round(sum(costs), 6) if costs else 0.0,
        "cost_per_1k_emails": round(sum(costs) / n * 1000, 4) if costs and n else None,
        "calls_per_email": round(sum(calls) / n, 3) if n else None,
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "llm_latency_ms_p50": percentile(llm_latencies, 0.5) if llm_latencies else None,
        "llm_latency_ms_p95": percentile(llm_latencies, 0.95) if llm_latencies else None,
        "route_counts": dict(routes),
        "error_count": errors,
        "questions_hash": _questions_hash(),
    }


def run(
    settings: Settings,
    *,
    source: Path,
    backend_name: str,
    limit: int | None = None,
    parallel: int = 1,
    use_llm: bool = False,
    price_per_mtok: float = 0.0,
    transport: httpx.BaseTransport | None = None,
    llm_transport: httpx.BaseTransport | None = None,
) -> Path:
    emails = load_emails(source)
    if limit:
        emails = emails[:limit]
    prepared_emails = [prepare(e) for e in emails]

    backend = build_backend(
        backend_name, settings, price_per_mtok=price_per_mtok, transport=transport
    )
    llm_client = LlmClient.from_settings(settings, transport=llm_transport) if use_llm else None
    llm_available = False
    if llm_client is not None:
        llm_available = llm_client.healthy()
        if not llm_available:
            print(
                f"warning: LLM_BASE_URL {settings.llm_base_url!r} did not respond to a health "
                "check; escalations will be queued for review instead of called",
                file=sys.stderr,
            )

    started = datetime.now(UTC)
    rows = run_parallel(
        prepared_emails,
        lambda p: _process_one(p, backend, llm_client, llm_available),
        parallel,
    )
    finished = datetime.now(UTC)

    run_dir = _new_run_dir(settings.runs_dir)
    with (run_dir / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = _summarise(
        rows,
        backend_name=backend_name,
        settings=settings,
        llm_available=llm_available,
        started=started,
        finished=finished,
    )
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir

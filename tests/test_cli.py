from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jev_email_cascade.cli import app


def _write_run(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "backend": "jev",
                "model": "typesafe/jev-1.13-20260917",
                "n": len(rows),
                "cost_usd": 0.001,
                "cost_per_1k_emails": 0.01,
                "calls_per_email": 1.0,
                "latency_ms_p50": 100.0,
                "latency_ms_p95": 200.0,
                "error_count": 0,
            }
        )
    )
    with (run_dir / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(category: str) -> dict:
    return {
        "id": "x",
        "labels": {
            "category": category,
            "priority": 0,
            "awaiting_reply": True,
            "deadline_present": False,
            "dissatisfied": False,
            "needs_decision": False,
            "opportunity": False,
            "injection_suspected": False,
        },
        "answers": {
            "category": {"type": "choice", "choice": category, "confidence": 0.9},
            "priority": {"type": "score", "score": 0.0, "confidence": 0.8},
            "awaiting_reply": {"type": "noul", "noul": 0.9},
            "deadline_present": {"type": "noul", "noul": 0.1},
            "dissatisfied": {"type": "noul", "noul": 0.1},
            "needs_decision": {"type": "noul", "noul": 0.1},
            "opportunity": {"type": "noul", "noul": 0.1},
            "injection_suspected": {"type": "noul", "noul": 0.05},
        },
        "decision": {
            "route": "auto",
            "flags": {
                "awaiting_reply": True,
                "deadline_present": False,
                "dissatisfied": False,
                "needs_decision": False,
                "opportunity": False,
                "injection_suspected": False,
            },
        },
    }


def test_report_markdown_confusion_row_is_not_wrapped(tmp_path: Path) -> None:
    """Regression test: `out.print()` (Rich) soft-wraps long lines when stdout isn't a wide
    TTY, splitting a wide table row across two lines and corrupting the Markdown. `report
    --markdown` must emit each table row as a single line regardless of terminal width."""
    categories = ["billing", "hr", "internal", "other", "sales", "spam", "support", "vendor"]
    run_dir = tmp_path / "runs" / "r1"
    _write_run(run_dir, [_row(c) for c in categories])

    runner = CliRunner()
    # A narrow COLUMNS forces Rich to wrap if the CLI mistakenly routes markdown through it.
    result = runner.invoke(
        app, ["report", str(run_dir), "--markdown"], env={"COLUMNS": "40", "TERM": "dumb"}
    )

    assert result.exit_code == 0
    lines = result.output.splitlines()
    header_lines = [ln for ln in lines if ln.startswith("| true \\ pred")]
    assert len(header_lines) == 1
    header = header_lines[0]
    for c in categories:
        assert c in header
    assert header.endswith("|")


def test_run_help_lists_frontier_backend_and_reasoning_effort() -> None:
    result = CliRunner().invoke(app, ["run", "--help"], env={"COLUMNS": "200", "TERM": "dumb"})
    assert result.exit_code == 0
    assert "frontier" in result.output
    assert "--reasoning-effort" in result.output


def test_run_rejects_bad_reasoning_effort_before_touching_anything() -> None:
    result = CliRunner().invoke(app, ["run", "--backend", "mock-jev", "--reasoning-effort", "max"])
    assert result.exit_code == 2
    assert "reasoning-effort" in result.output


def test_run_frontier_without_key_exits_1_with_guidance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = Path(__file__).resolve().parents[1] / "data" / "emails.jsonl"
    monkeypatch.chdir(tmp_path)  # no .env here, so the key really is unset
    result = CliRunner().invoke(
        app, ["run", "--backend", "frontier", "--limit", "1", "--source", str(source)]
    )
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_report_markdown_shows_tokens_pricing_and_llm_hook(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r2"
    _write_run(run_dir, [_row("billing") for _ in range(6)])
    run_json = json.loads((run_dir / "run.json").read_text())
    run_json.update(
        {
            "backend": "frontier",
            "model": "gpt-5.6-terra",
            "input_tokens": 67120,
            "output_tokens": 18004,
            "cached_input_tokens": 0,
            "reasoning_tokens": 12300,
            "reasoning_effort": "low",
            "pricing": {"input": 2.0, "cached_input": 0.2, "output": 12.0},
            "llm_provider": "frontier",
            "llm_model": "gpt-5.6-terra",
            "llm_calls": 3,
            "llm_cost_usd": 0.0123,
            "llm_reasoning_tokens": 900,
        }
    )
    (run_dir / "run.json").write_text(json.dumps(run_json))
    result = CliRunner().invoke(app, ["report", str(run_dir), "--markdown"])
    assert result.exit_code == 0
    assert "- tokens: in 67,120 (cached 0) / out 18,004 (reasoning 12,300)" in result.output
    assert "- pricing: list price $2.0/M in, $0.2/M cached, $12.0/M out; reasoning_effort=low" in (
        result.output
    )
    assert "- llm hook: frontier gpt-5.6-terra, 3 calls, $0.0123, 900 reasoning tokens" in (
        result.output
    )

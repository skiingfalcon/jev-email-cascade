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

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from jev_email_cascade.report import (
    compare,
    compare_markdown,
    compute_metrics,
    load_run,
    render_markdown,
)


def _write_run(run_dir: Path, *, backend: str, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "model": "m",
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


def _row(
    qid_category: str,
    pred_category: str,
    priority_true: int,
    priority_pred: float,
    route: str = "auto",
) -> dict:
    return {
        "id": "x",
        "labels": {
            "category": qid_category,
            "priority": priority_true,
            "awaiting_reply": True,
            "deadline_present": False,
            "dissatisfied": False,
            "needs_decision": False,
            "opportunity": False,
            "injection_suspected": False,
        },
        "answers": {
            "category": {"type": "choice", "choice": pred_category, "confidence": 0.9},
            "priority": {"type": "score", "score": priority_pred, "confidence": 0.8},
            "awaiting_reply": {"type": "noul", "noul": 0.9},
            "deadline_present": {"type": "noul", "noul": 0.1},
            "dissatisfied": {"type": "noul", "noul": 0.1},
            "needs_decision": {"type": "noul", "noul": 0.1},
            "opportunity": {"type": "noul", "noul": 0.1},
            "injection_suspected": {"type": "noul", "noul": 0.05},
        },
        "decision": {
            "route": route,
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


def test_category_and_priority_metrics(tmp_path: Path) -> None:
    rows = [
        _row("billing", "billing", 2, 2.0),
        _row("billing", "billing", 2, 3.0),  # priority off by 1
        _row("support", "sales", 1, 1.0),  # category wrong
        _row("support", "support", 1, 0.0, route="review"),  # priority off by 1
        _row("hr", "hr", 0, 0.0),
        _row("hr", "hr", 0, 0.0),
    ]
    run_dir = tmp_path / "runs" / "run1"
    _write_run(run_dir, backend="mock-jev", rows=rows)

    run_json, loaded_rows = load_run(run_dir)
    assert len(loaded_rows) == 6
    metrics = compute_metrics(run_json, loaded_rows)

    assert metrics["category"]["accuracy"] == 5 / 6
    assert metrics["category"]["confusion"]["support"]["sales"] == 1
    assert metrics["priority"]["exact"] == 4 / 6
    assert metrics["priority"]["within1"] == 1.0
    assert metrics["routes"] == {"auto": 5, "review": 1}


def test_noul_calibration_bands(tmp_path: Path) -> None:
    rows = [_row("billing", "billing", 0, 0.0) for _ in range(6)]
    run_dir = tmp_path / "runs" / "run2"
    _write_run(run_dir, backend="mock-jev", rows=rows)
    run_json, loaded_rows = load_run(run_dir)
    metrics = compute_metrics(run_json, loaded_rows)
    band = metrics["nouls"]["awaiting_reply"]["calibration"]["0.9-1.0"]
    assert band["n"] == 6
    assert band["positive_rate"] == 1.0


def test_bootstrap_ci_present_at_min_n_and_none_below(tmp_path: Path) -> None:
    few = [_row("billing", "billing", 0, 0.0) for _ in range(3)]
    many = [_row("billing", "billing", 0, 0.0) for _ in range(6)]
    d1 = tmp_path / "runs" / "few"
    d2 = tmp_path / "runs" / "many"
    _write_run(d1, backend="mock-jev", rows=few)
    _write_run(d2, backend="mock-jev", rows=many)
    m1 = compute_metrics(*load_run(d1))
    m2 = compute_metrics(*load_run(d2))
    assert m1["category"]["ci"] is None
    assert m2["category"]["ci"] is not None


def test_compare_runs_no_crash(tmp_path: Path) -> None:
    rows = [_row("billing", "billing", 0, 0.0) for _ in range(6)]
    d1 = tmp_path / "runs" / "a"
    d2 = tmp_path / "runs" / "b"
    _write_run(d1, backend="mock-jev", rows=rows)
    _write_run(d2, backend="gen-json", rows=rows)
    console = Console(record=True, width=200)
    compare([d1, d2], console)
    text = console.export_text()
    assert "mock-jev" in text
    assert "gen-json" in text


def test_compare_markdown_one_line_per_run_with_tokens_and_hook_cost(tmp_path: Path) -> None:
    rows = [_row("billing", "billing", 0, 0.0) for _ in range(6)]
    d1 = tmp_path / "runs" / "jev"
    d2 = tmp_path / "runs" / "frontier"
    _write_run(d1, backend="jev", rows=rows)
    _write_run(d2, backend="frontier", rows=rows)
    run_json = json.loads((d2 / "run.json").read_text())
    run_json.update(
        {
            "model": "gpt-5.6-terra",
            "input_tokens": 67120,
            "output_tokens": 18004,
            "reasoning_tokens": 12300,
            "llm_cost_usd": 0.05,
        }
    )
    (d2 / "run.json").write_text(json.dumps(run_json))

    text = compare_markdown([d1, d2])
    lines = text.splitlines()
    assert len(lines) == 4  # header, separator, two runs
    assert lines[0].startswith("| run | backend | model |")
    assert "| jev | m |" in lines[2] and "| - |" in lines[2]  # no token counts recorded
    assert "| frontier | gpt-5.6-terra |" in lines[3]
    assert "67,120 / 18,004 (12,300)" in lines[3]
    assert "| 0.05 |" in lines[3]


def test_render_markdown_includes_confusion_and_calibration(tmp_path: Path) -> None:
    rows = [
        _row("billing", "billing", 2, 2.0),
        _row("support", "sales", 1, 1.0),
    ]
    run_dir = tmp_path / "runs" / "md"
    _write_run(run_dir, backend="jev", rows=rows)
    metrics = compute_metrics(*load_run(run_dir))
    text = render_markdown(metrics, run_dir)

    assert "category accuracy" in text
    assert "Category confusion" in text
    assert "| true \\ pred |" in text
    assert "billing" in text and "support" in text and "sales" in text
    assert "Noul calibration" in text
    assert "awaiting_reply" in text

"""Score one run's results.jsonl against its labels, and compare runs side by side."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from jev_email_cascade.stats import bootstrap_ci

NOUL_QUESTIONS = (
    "awaiting_reply",
    "deadline_present",
    "dissatisfied",
    "needs_decision",
    "opportunity",
    "injection_suspected",
)

BANDS: list[tuple[float, float]] = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.001)]


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    run_json = json.loads((run_dir / "run.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return run_json, rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def category_metrics(rows: list[dict]) -> dict:
    flags: list[bool] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        true = row["labels"]["category"]
        a = row.get("answers", {}).get("category") or {}
        pred = a.get("choice") or "(error)"
        flags.append(pred == true)
        confusion[true][pred] += 1
    n = len(flags)
    return {
        "accuracy": sum(flags) / n if n else None,
        "ci": bootstrap_ci(flags),
        "n": n,
        "confusion": confusion,
    }


def priority_metrics(rows: list[dict]) -> dict:
    exact: list[bool] = []
    within1: list[bool] = []
    for row in rows:
        true = row["labels"]["priority"]
        a = row.get("answers", {}).get("priority") or {}
        score = a.get("score")
        if score is None:
            exact.append(False)
            within1.append(False)
            continue
        pred = round(score)
        exact.append(pred == true)
        within1.append(abs(pred - true) <= 1)
    n = len(exact)
    return {
        "exact": sum(exact) / n if n else None,
        "exact_ci": bootstrap_ci(exact),
        "within1": sum(within1) / n if n else None,
        "within1_ci": bootstrap_ci(within1),
        "n": n,
    }


def noul_metrics(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for qid in NOUL_QUESTIONS:
        raw_flags: list[bool] = []
        acted_flags: list[bool] = []
        wrong_confidences: list[float] = []
        band_counts = {b: [0, 0] for b in BANDS}
        for row in rows:
            true = bool(row["labels"][qid])
            a = row.get("answers", {}).get(qid) or {}
            p = a.get("noul")
            if p is None:
                raw_flags.append(False)
                continue
            pred = p >= 0.5
            raw_flags.append(pred == true)
            if pred != true:
                wrong_confidences.append(max(p, 1 - p))
            for band in BANDS:
                if band[0] <= p < band[1]:
                    band_counts[band][0] += 1
                    band_counts[band][1] += int(true)
                    break
            decision_flag = row.get("decision", {}).get("flags", {}).get(qid)
            if decision_flag is not None:
                acted_flags.append(decision_flag == true)
        n_raw = len(raw_flags)
        out[qid] = {
            "raw_accuracy": sum(raw_flags) / n_raw if n_raw else None,
            "raw_ci": bootstrap_ci(raw_flags),
            "acted_accuracy": sum(acted_flags) / len(acted_flags) if acted_flags else None,
            "acted_n": len(acted_flags),
            "n": n_raw,
            "mean_confidence_on_wrong": (
                sum(wrong_confidences) / len(wrong_confidences) if wrong_confidences else None
            ),
            "calibration": {
                f"{lo:.1f}-{min(hi, 1.0):.1f}": {
                    "n": c[0],
                    "positive_rate": (c[1] / c[0]) if c[0] else None,
                }
                for (lo, hi), c in band_counts.items()
            },
        }
    return out


def route_metrics(rows: list[dict]) -> dict:
    return dict(Counter(row.get("decision", {}).get("route", "?") for row in rows))


def compute_metrics(run_json: dict, rows: list[dict]) -> dict:
    return {
        "backend": run_json.get("backend"),
        "model": run_json.get("model"),
        "n": len(rows),
        "category": category_metrics(rows),
        "priority": priority_metrics(rows),
        "nouls": noul_metrics(rows),
        "routes": route_metrics(rows),
        "cost_usd": run_json.get("cost_usd"),
        "cost_per_1k_emails": run_json.get("cost_per_1k_emails"),
        "calls_per_email": run_json.get("calls_per_email"),
        "latency_ms_p50": run_json.get("latency_ms_p50"),
        "latency_ms_p95": run_json.get("latency_ms_p95"),
        "error_count": run_json.get("error_count"),
        "wall_s": run_json.get("wall_s"),
        "input_tokens": run_json.get("input_tokens"),
        "output_tokens": run_json.get("output_tokens"),
        "cached_input_tokens": run_json.get("cached_input_tokens"),
        "reasoning_tokens": run_json.get("reasoning_tokens"),
        "reasoning_effort": run_json.get("reasoning_effort"),
        "pricing": run_json.get("pricing"),
        "llm_provider": run_json.get("llm_provider"),
        "llm_model": run_json.get("llm_model"),
        "llm_calls": run_json.get("llm_calls"),
        "llm_cost_usd": run_json.get("llm_cost_usd"),
        "llm_reasoning_tokens": run_json.get("llm_reasoning_tokens"),
    }


def _tokens_text(metrics: dict) -> str | None:
    """'in 67,120 (cached 0) / out 18,004 (reasoning 12,300)' -- None when the run recorded no
    token counts at all (older run.json files)."""
    if metrics.get("input_tokens") is None and metrics.get("output_tokens") is None:
        return None
    parts = [f"in {metrics.get('input_tokens') or 0:,}"]
    if metrics.get("cached_input_tokens") is not None:
        parts[-1] += f" (cached {metrics['cached_input_tokens']:,})"
    parts.append(f"out {metrics.get('output_tokens') or 0:,}")
    if metrics.get("reasoning_tokens") is not None:
        parts[-1] += f" (reasoning {metrics['reasoning_tokens']:,})"
    return " / ".join(parts)


def _pricing_text(metrics: dict) -> str | None:
    p = metrics.get("pricing")
    if not p:
        return None
    text = f"list price ${p['input']}/M in, ${p['cached_input']}/M cached, ${p['output']}/M out"
    if metrics.get("reasoning_effort"):
        text += f"; reasoning_effort={metrics['reasoning_effort']}"
    return text


def _llm_text(metrics: dict) -> str | None:
    if not metrics.get("llm_model"):
        return None
    text = f"{metrics.get('llm_provider') or 'local'} {metrics['llm_model']}"
    if metrics.get("llm_calls") is not None:
        text += f", {metrics['llm_calls']} calls"
    if metrics.get("llm_cost_usd") is not None:
        text += f", ${_fmt(metrics['llm_cost_usd'])}"
    if metrics.get("llm_reasoning_tokens") is not None:
        text += f", {metrics['llm_reasoning_tokens']:,} reasoning tokens"
    return text


def render_rich(metrics: dict, run_dir: Path, console: Console | None = None) -> None:
    console = console or Console()
    console.print(
        f"[bold]{run_dir}[/bold]  backend={metrics['backend']}  model={metrics['model']}  "
        f"n={metrics['n']}"
    )

    cat = metrics["category"]
    console.print(f"category accuracy: {_fmt(cat['accuracy'])}  95% CI {ci_text_from(cat['ci'])}")
    conf_table = Table(title="category confusion (rows = true, cols = predicted)")
    labels = sorted(cat["confusion"])
    all_preds = sorted({p for row in cat["confusion"].values() for p in row})
    conf_table.add_column("true \\ pred")
    for p in all_preds:
        conf_table.add_column(p)
    for true in labels:
        conf_table.add_row(true, *(_fmt(cat["confusion"][true].get(p, 0)) for p in all_preds))
    console.print(conf_table)

    pri = metrics["priority"]
    console.print(
        f"priority exact: {_fmt(pri['exact'])} (CI {ci_text_from(pri['exact_ci'])})  "
        f"within-1: {_fmt(pri['within1'])} (CI {ci_text_from(pri['within1_ci'])})"
    )

    noul_table = Table(title="noul questions")
    for col in ("question", "raw acc", "95% CI", "acted acc (n)", "mean conf on wrong"):
        noul_table.add_column(col)
    for qid, m in metrics["nouls"].items():
        noul_table.add_row(
            qid,
            _fmt(m["raw_accuracy"]),
            ci_text_from(m["raw_ci"]),
            f"{_fmt(m['acted_accuracy'])} ({m['acted_n']}/{m['n']})",
            _fmt(m["mean_confidence_on_wrong"]),
        )
    console.print(noul_table)

    cal_table = Table(
        title="noul calibration (observed positive rate by predicted-probability band)"
    )
    cal_table.add_column("question")
    bands = list(next(iter(metrics["nouls"].values()))["calibration"]) if metrics["nouls"] else []
    for b in bands:
        cal_table.add_column(b)
    for qid, m in metrics["nouls"].items():
        cal_table.add_row(
            qid,
            *(
                f"{_fmt(m['calibration'][b]['positive_rate'])} (n={m['calibration'][b]['n']})"
                for b in bands
            ),
        )
    console.print(cal_table)

    console.print(f"routes: {metrics['routes']}")
    console.print(
        f"cost: ${_fmt(metrics['cost_usd'])} total, ${_fmt(metrics['cost_per_1k_emails'])} "
        f"/1K emails  |  calls/email: {_fmt(metrics['calls_per_email'])}  |  "
        f"latency p50/p95: {_fmt(metrics['latency_ms_p50'])}/{_fmt(metrics['latency_ms_p95'])} ms  "
        f"|  errors: {metrics['error_count']}"
    )
    for label, text in (
        ("tokens", _tokens_text(metrics)),
        ("pricing", _pricing_text(metrics)),
        ("llm hook", _llm_text(metrics)),
    ):
        if text:
            console.print(f"{label}: {text}")


def ci_text_from(ci: tuple[float, float] | None) -> str:
    return "-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"


def render_markdown(metrics: dict, run_dir: Path) -> str:
    lines = [
        f"### {run_dir}",
        f"backend=`{metrics['backend']}` model=`{metrics['model']}` n={metrics['n']}",
        "",
        f"- category accuracy: {_fmt(metrics['category']['accuracy'])} "
        f"(CI {ci_text_from(metrics['category']['ci'])})",
        f"- priority exact: {_fmt(metrics['priority']['exact'])} "
        f"(CI {ci_text_from(metrics['priority']['exact_ci'])}), "
        f"within-1: {_fmt(metrics['priority']['within1'])} "
        f"(CI {ci_text_from(metrics['priority']['within1_ci'])})",
        f"- routes: {metrics['routes']}",
        f"- cost: ${_fmt(metrics['cost_usd'])} total, "
        f"${_fmt(metrics['cost_per_1k_emails'])}/1K emails, "
        f"{_fmt(metrics['calls_per_email'])} calls/email",
        f"- latency p50/p95: {_fmt(metrics['latency_ms_p50'])}/"
        f"{_fmt(metrics['latency_ms_p95'])} ms",
        f"- errors: {metrics['error_count']}",
    ]
    for label, text in (
        ("tokens", _tokens_text(metrics)),
        ("pricing", _pricing_text(metrics)),
        ("llm hook", _llm_text(metrics)),
    ):
        if text:
            lines.append(f"- {label}: {text}")
    lines += [
        "",
        "| question | raw acc | 95% CI | acted acc (n) | mean conf on wrong |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for qid, m in metrics["nouls"].items():
        lines.append(
            f"| {qid} | {_fmt(m['raw_accuracy'])} | {ci_text_from(m['raw_ci'])} | "
            f"{_fmt(m['acted_accuracy'])} ({m['acted_n']}/{m['n']}) | "
            f"{_fmt(m['mean_confidence_on_wrong'])} |"
        )

    cat = metrics["category"]
    all_preds = sorted({p for row in cat["confusion"].values() for p in row})
    lines += ["", "**Category confusion** (rows = true, cols = predicted)", ""]
    lines.append("| true \\ pred | " + " | ".join(all_preds) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in all_preds) + " |")
    for true in sorted(cat["confusion"]):
        row = cat["confusion"][true]
        lines.append(f"| {true} | " + " | ".join(_fmt(row.get(p, 0)) for p in all_preds) + " |")

    bands = list(next(iter(metrics["nouls"].values()))["calibration"]) if metrics["nouls"] else []
    if bands:
        lines += [
            "",
            "**Noul calibration** (observed positive rate by predicted-probability band)",
            "",
        ]
        lines.append("| question | " + " | ".join(bands) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in bands) + " |")
        for qid, m in metrics["nouls"].items():
            cells = [
                f"{_fmt(m['calibration'][b]['positive_rate'])} (n={m['calibration'][b]['n']})"
                for b in bands
            ]
            lines.append(f"| {qid} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


COMPARE_COLUMNS = (
    "run",
    "backend",
    "model",
    "category acc",
    "priority exact / ±1",
    "calls/email",
    "tokens in / out (reasoning)",
    "$ total",
    "$/1K emails",
    "llm hook $",
    "latency p50 / p95 ms",
    "errors",
)


def _compare_row(run_dir: Path, m: dict) -> list[str]:
    if m.get("input_tokens") is not None or m.get("output_tokens") is not None:
        tokens = f"{m['input_tokens'] or 0:,} / {m['output_tokens'] or 0:,}"
        if m.get("reasoning_tokens") is not None:
            tokens += f" ({m['reasoning_tokens']:,})"
    else:
        tokens = "-"
    return [
        run_dir.name,
        str(m["backend"]),
        str(m["model"]),
        _fmt(m["category"]["accuracy"]),
        f"{_fmt(m['priority']['exact'])} / {_fmt(m['priority']['within1'])}",
        _fmt(m["calls_per_email"]),
        tokens,
        _fmt(m["cost_usd"]),
        _fmt(m["cost_per_1k_emails"]),
        _fmt(m.get("llm_cost_usd")),
        f"{_fmt(m['latency_ms_p50'])} / {_fmt(m['latency_ms_p95'])}",
        str(m["error_count"]),
    ]


def compare_rows(run_dirs: list[Path]) -> list[list[str]]:
    rows = []
    for run_dir in run_dirs:
        run_json, results = load_run(run_dir)
        rows.append(_compare_row(run_dir, compute_metrics(run_json, results)))
    return rows


def compare_markdown(run_dirs: list[Path]) -> str:
    lines = [
        "| " + " | ".join(COMPARE_COLUMNS) + " |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in COMPARE_COLUMNS[3:]) + " |",
    ]
    for row in compare_rows(run_dirs):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compare(run_dirs: list[Path], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="compare")
    for col in COMPARE_COLUMNS:
        table.add_column(col)
    for row in compare_rows(run_dirs):
        table.add_row(*row)
    console.print(table)

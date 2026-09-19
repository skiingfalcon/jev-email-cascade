"""cascade demo | run | report | compare | questions | estimate."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from jev_email_cascade.config import get_settings
from jev_email_cascade.jev_client import JEV_PRICE_PER_INPUT_TOKEN
from jev_email_cascade.pipeline import BACKEND_NAMES
from jev_email_cascade.pipeline import run as run_pipeline
from jev_email_cascade.questions import questions_json
from jev_email_cascade.report import compare as compare_runs
from jev_email_cascade.report import compute_metrics, load_run, render_markdown, render_rich

app = typer.Typer(
    name="cascade", help="Prove the Jev decision-model cascade end to end.", no_args_is_help=True
)
out = Console()
err = Console(stderr=True)

DEFAULT_SOURCE = Path("data/emails.jsonl")


@app.command()
def demo() -> None:
    """Mock Jev vs mock generative (json mode), end to end, on the full sample set. No API key
    or LLM_BASE_URL needed."""
    settings = get_settings()
    jev_dir = run_pipeline(settings, source=DEFAULT_SOURCE, backend_name="mock-jev")
    gen_dir = run_pipeline(settings, source=DEFAULT_SOURCE, backend_name="mock-gen")
    out.print("[bold]mock-jev[/bold] -- parallel typed questions, one call per email")
    render_rich(compute_metrics(*load_run(jev_dir)), jev_dir, out)
    out.print()
    out.print("[bold]mock-gen[/bold] -- one prompt, all questions, one JSON object back")
    render_rich(compute_metrics(*load_run(gen_dir)), gen_dir, out)
    out.print()
    compare_runs([jev_dir, gen_dir], out)


@app.command(name="run")
def run_cmd(
    backend: str = typer.Option(..., "--backend", help=f"One of {', '.join(BACKEND_NAMES)}"),
    source: Path = typer.Option(DEFAULT_SOURCE, "--source", help="emails.jsonl to run"),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N emails"),
    parallel: int = typer.Option(1, "--parallel", help="Concurrent requests"),
    llm: bool = typer.Option(
        False, "--llm/--no-llm", help="Escalate uncertain items to LLM_BASE_URL"
    ),
    gen_price_per_mtok: float = typer.Option(
        0.0,
        "--gen-price-per-mtok",
        help="Hosted-equivalent $/M tokens for the generative backends' cost estimate "
        "(local gpt-oss is $0 marginal; this answers 'what would this cost through an API')",
    ),
) -> None:
    """Run one backend over the email set and write runs/<stamp>/."""
    if backend not in BACKEND_NAMES:
        err.print(f"[red]unknown backend[/red] {backend!r}; choose from {', '.join(BACKEND_NAMES)}")
        raise typer.Exit(2)
    settings = get_settings()
    try:
        run_dir = run_pipeline(
            settings,
            source=source,
            backend_name=backend,
            limit=limit,
            parallel=parallel,
            use_llm=llm,
            price_per_mtok=gen_price_per_mtok,
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not a crash
        err.print(f"[red]error[/red] {exc}")
        raise typer.Exit(1) from exc
    out.print(f"[green]done[/green] {run_dir}")
    render_rich(compute_metrics(*load_run(run_dir)), run_dir, out)


@app.command(name="report")
def report_cmd(
    run_dir: Path = typer.Argument(..., help="A runs/<stamp> directory"),
    markdown: bool = typer.Option(
        False, "--markdown", help="Print as Markdown instead of a rich table"
    ),
) -> None:
    metrics = compute_metrics(*load_run(run_dir))
    if markdown:
        # Plain print, not out.print(): Rich's Console soft-wraps long lines to the terminal
        # width (or a fixed default when output isn't a TTY), which corrupts a wide Markdown
        # table row into two lines. Markdown output must be byte-exact and pasteable.
        print(render_markdown(metrics, run_dir))
    else:
        render_rich(metrics, run_dir, out)


@app.command(name="compare")
def compare_cmd(
    run_dirs: list[Path] = typer.Argument(..., help="Two or more runs/<stamp> directories"),
) -> None:
    compare_runs(run_dirs, out)


@app.command(name="questions")
def questions_cmd() -> None:
    """Print the question set exactly as sent to a decision backend."""
    out.print_json(json.dumps(questions_json()))


@app.command(name="estimate")
def estimate_cmd(
    emails: int = typer.Option(..., "--emails", help="Number of emails"),
    tokens_per_email: int = typer.Option(
        700, "--tokens-per-email", help="Input tokens per email, all questions"
    ),
) -> None:
    """Cost math for Jev at $0.042/M input tokens, output free."""
    total_tokens = emails * tokens_per_email
    cost = total_tokens * JEV_PRICE_PER_INPUT_TOKEN
    out.print(
        f"{emails:,} emails x {tokens_per_email:,} tokens/email = {total_tokens:,} input tokens"
    )
    if emails:
        out.print(f"cost: ${cost:.4f}   (${cost / emails * 1000:.4f} per 1K emails)")

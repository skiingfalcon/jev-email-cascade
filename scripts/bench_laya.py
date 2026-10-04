"""Latency / throughput / memory benchmark for a local laya-serve, using this repo's emails and
question set (the same request bodies `cascade run --backend laya` sends).

Phases:
  first     the first request this process sends (a true cold start needs a fresh server)
  seq       every email, --repeat times, one request at a time: client p50/p95/p99 and the
            server's own X-Inference-Time-Ms
  conc-N    the same requests with N in flight (laya-serve runs one inference at a time, so
            this measures queueing, not parallel speedup)
  batch-B   POST /v1/systemone/batch with B emails per call: per-email amortised latency

Writes runs/bench-<stamp>/bench.json and bench.md.

    uv run python scripts/bench_laya.py [--model english] [--label cpu] [--repeat 2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from jev_email_cascade.config import Settings
from jev_email_cascade.laya_client import LayaClient
from jev_email_cascade.prepare import load_emails, prepare
from jev_email_cascade.questions import QUESTIONS, questions_json
from jev_email_cascade.stats import percentile

REPO = Path(__file__).resolve().parents[1]

_MEM_PS = r"""
$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1
if (-not $c) {{ '{{}}'; exit }}
$procId = $c.OwningProcess
$p = Get-Process -Id $procId
$base = "\GPU Process Memory(pid_${{procId}}_*)"
$ctr = Get-Counter "$base\Dedicated Usage", "$base\Shared Usage" -ErrorAction SilentlyContinue
$samples = $ctr.CounterSamples
$ded = ($samples | Where-Object Path -like '*dedicated usage' | Measure-Object CookedValue -Sum).Sum
$sh = ($samples | Where-Object Path -like '*shared usage' | Measure-Object CookedValue -Sum).Sum
@{{pid=$procId; rss_mb=[math]::Round($p.WorkingSet64/1MB);
  private_mb=[math]::Round($p.PrivateMemorySize64/1MB);
  gpu_dedicated_mb=[math]::Round($ded/1MB);
  gpu_shared_mb=[math]::Round($sh/1MB)}} | ConvertTo-Json -Compress
"""


def server_memory(port: int) -> dict:
    """Server process RAM and GPU memory (Windows perf counters); {} when unavailable."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _MEM_PS.format(port=port)],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        return json.loads(out) if out else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def _stats(client_ms: list[float], server_ms: list[float], wall_s: float, n_emails: int) -> dict:
    return {
        "n_requests": len(client_ms),
        "n_emails": n_emails,
        "wall_s": wall_s,
        "emails_per_s": n_emails / wall_s if wall_s else None,
        "client_ms_p50": percentile(client_ms, 0.5),
        "client_ms_p95": percentile(client_ms, 0.95),
        "client_ms_p99": percentile(client_ms, 0.99),
        "server_ms_p50": percentile(server_ms, 0.5),
        "server_ms_p95": percentile(server_ms, 0.95),
    }


class Bench:
    def __init__(self, laya: LayaClient, http: httpx.Client) -> None:
        self.laya = laya
        self.http = http
        self.errors = 0

    def one(self, state: dict) -> tuple[float, float | None]:
        body = self.laya.request_body(state, QUESTIONS)
        t0 = time.perf_counter()
        r = self.http.post(self.laya.url, json=body)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            self.errors += 1
        server = r.headers.get("x-inference-time-ms")
        return ms, float(server) if server else None

    def run(self, states: list[dict], concurrency: int) -> dict:
        t0 = time.perf_counter()
        if concurrency <= 1:
            results = [self.one(s) for s in states]
        else:
            with ThreadPoolExecutor(concurrency) as pool:
                results = list(pool.map(self.one, states))
        wall = time.perf_counter() - t0
        return _stats(
            [c for c, _ in results], [s for _, s in results if s is not None], wall, len(states)
        )

    def batch(self, states: list[dict], size: int) -> dict:
        url = self.laya.url.rstrip("/") + "/batch"
        client_ms, server_ms = [], []
        t0 = time.perf_counter()
        for i in range(0, len(states), size):
            chunk = [self.laya.encode_state(s) for s in states[i : i + size]]
            body = {"states": chunk, "questions": questions_json(QUESTIONS)}
            if self.laya.model:
                body["model"] = self.laya.model
            t = time.perf_counter()
            r = self.http.post(url, json=body)
            client_ms.append((time.perf_counter() - t) * 1000 / len(chunk))
            if r.status_code != 200:
                self.errors += 1
                print(f"batch-{size}: http {r.status_code} {r.text[:200]}", file=sys.stderr)
            if r.headers.get("x-inference-time-ms"):
                server_ms.append(float(r.headers["x-inference-time-ms"]) / len(chunk))
        wall = time.perf_counter() - t0
        out = _stats(client_ms, server_ms, wall, len(states))
        out["note"] = "latencies are per email (call latency / batch size)"
        return out


def to_markdown(report: dict) -> str:
    devices = report["health"].get("checkpoint_devices") or report["health"].get("device")
    lines = [
        f"# Laya bench `{report['label']}` ({report['started']})",
        "",
        f"- server: `{report['url']}`, model `{report['model'] or 'router default'}`, "
        f"state `{report['state_mode']}`",
        f"- device: {devices}",
        f"- emails: {report['n_emails']} x {report['repeat']} repeat(s); "
        f"errors: {report['errors']}",
        f"- memory idle: {report['memory_idle']}",
        f"- memory peak (after phases): {report['memory_peak']}",
        "",
        "| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    for name, m in report["phases"].items():
        lines.append(
            f"| {name} | {m['n_requests']} | {f(m['emails_per_s'])} | {f(m['client_ms_p50'])} | "
            f"{f(m['client_ms_p95'])} | {f(m['client_ms_p99'])} | {f(m['server_ms_p50'])} | "
            f"{f(m['server_ms_p95'])} |"
        )
    lines += ["", "Latencies in ms. Batch rows are per email (call latency / batch size)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--label", default="laya", help="name for this configuration, e.g. cpu / rocm")
    ap.add_argument("--model", default=None, help="english | typed-decisions | multilingual")
    ap.add_argument("--state-mode", default=None, choices=("raw", "rendered"))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--concurrency", default="2,4,8,16")
    ap.add_argument("--batch-sizes", default="8,32,64")
    ap.add_argument("--source", type=Path, default=REPO / "data" / "emails.jsonl")
    args = ap.parse_args()

    settings = Settings()
    overrides = {"laya_model": args.model} if args.model else {}
    if args.state_mode:
        overrides["laya_state_mode"] = args.state_mode
    laya = LayaClient.from_settings(settings.model_copy(update=overrides))
    health = laya.health()
    if health is None:
        print(f"laya-serve not answering at {laya.base_url}/health", file=sys.stderr)
        return 1
    port = httpx.URL(laya.url).port or 80
    states = [prepare(e).state for e in load_emails(args.source)]
    http = httpx.Client(
        timeout=httpx.Timeout(300.0, connect=5.0),
        headers={"Authorization": f"Bearer {settings.laya_api_key}"}
        if settings.laya_api_key
        else {},
    )
    bench = Bench(laya, http)

    report: dict = {
        "label": args.label,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": laya.url,
        "model": laya.model,
        "state_mode": laya.state_mode,
        "health": health,
        "n_emails": len(states),
        "repeat": args.repeat,
        "memory_idle": server_memory(port),
        "phases": {},
    }
    peak: dict = {}

    def sample() -> None:
        for k, v in server_memory(port).items():
            if isinstance(v, int | float) and k != "pid":
                peak[k] = max(peak.get(k, 0), v)

    phases = report["phases"]
    phases["first"] = bench.run(states[:1], 1)
    print("first", phases["first"]["client_ms_p50"])
    phases["seq"] = bench.run(states * args.repeat, 1)
    sample()
    print("seq", phases["seq"]["client_ms_p50"], phases["seq"]["server_ms_p50"])
    for c in (int(x) for x in args.concurrency.split(",") if x):
        phases[f"conc-{c}"] = bench.run(states, c)
        sample()
        print(f"conc-{c}", phases[f"conc-{c}"]["emails_per_s"])
    for b in (int(x) for x in args.batch_sizes.split(",") if x):
        phases[f"batch-{b}"] = bench.batch(states, b)
        sample()
        print(f"batch-{b}", phases[f"batch-{b}"]["emails_per_s"])

    report["memory_peak"] = peak
    report["errors"] = bench.errors
    out = REPO / "runs" / f"bench-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{args.label}"
    out.mkdir(parents=True)
    (out / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "bench.md").write_text(to_markdown(report))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

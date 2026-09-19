"""Small statistics helpers: latency percentiles for run.json, accuracy percentiles and
bootstrap confidence intervals for the report. Deterministic (seeded) so a report is
reproducible from its results.jsonl.
"""

from __future__ import annotations

import random

MIN_CI_N = 5
BOOTSTRAP_RESAMPLES = 2000


def percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def bootstrap_ci(
    flags: list[bool], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile-bootstrap 95% interval over a list of pass/fail flags. None under
    :data:`MIN_CI_N` items, where a bootstrap interval is not meaningful."""
    n = len(flags)
    if n < MIN_CI_N:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def ci_text(flags: list[bool]) -> str:
    ci = bootstrap_ci(flags)
    return "-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"

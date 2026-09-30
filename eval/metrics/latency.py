"""Latency percentiles (p50/p90/p95/p99, max) with a bootstrap CI on p95.

Percentiles use numpy's default linear interpolation between order statistics
(Hyndman-Fan type 7), stated here so numbers are comparable across reports. The
caller says which clock was measured (e.g. "event end -> alert emit, stream time"
or "decode -> alert, wall clock"); ARCHITECTURE targets p95 < 5 s event -> Alert.
"""

from __future__ import annotations

import numpy as np

from eval.metrics.stats import DEFAULT_B, DEFAULT_SEED, LOW_N, MetricValue, bootstrap_stat

QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}


def latency_summary(
    values: np.ndarray, clock: str, b: int = DEFAULT_B, seed: int = DEFAULT_SEED
) -> dict[str, MetricValue]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: MetricValue.unavailable(f"no latency samples ({clock})") for k in [*QUANTILES, "max"]}
    n = {"samples": int(v.size)}
    out: dict[str, MetricValue] = {}
    for k, q in QUANTILES.items():
        if k == "p95":
            pt, ci = bootstrap_stat(list(v), lambda xs: float(np.quantile(np.array(xs), 0.95)), b, seed)
            out[k] = MetricValue(pt, ci, n, unit="s", reason=clock, low_n=v.size < LOW_N)
        else:
            out[k] = MetricValue(float(np.quantile(v, q)), None, n, unit="s", reason=clock)
    out["max"] = MetricValue(float(v.max()), None, n, unit="s", reason=clock)
    return out

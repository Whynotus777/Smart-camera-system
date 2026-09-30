"""Metric values with sample counts and bootstrap confidence intervals.

docs/EVAL.md: every headline metric carries its sample counts and a 95% CI from a
bootstrap over clips/actors, and anything the labels can't support is reported as
"unavailable" rather than as a number. `MetricValue` is the one type that enforces
that shape, so suites can't emit a bare float by accident.

Bootstrap: resample *units* (clips, cameras, actors, sequences) with replacement and
recompute the metric from the resampled units' sufficient statistics. Events within a
clip are correlated, so resampling events individually would give CIs that are too
narrow. Percentile interval, fixed seed, B=1000 by default.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_B = 1000
DEFAULT_SEED = 20260929
LOW_N = 30  # fewer positives than this: the report says the CI is wide/unstable


@dataclass(frozen=True)
class MetricValue:
    """One reported number. `value is None` iff `status != "ok"`."""

    value: float | None
    ci95: tuple[float, float] | None = None
    n: Mapping[str, int] = field(default_factory=dict)  # e.g. {"events": 43, "clips": 12}
    status: str = "ok"  # ok | unavailable
    reason: str = ""  # why unavailable, or a caveat
    unit: str = ""
    low_n: bool = False  # few samples: say so in the summary

    @classmethod
    def unavailable(cls, reason: str, **n: int) -> MetricValue:
        return cls(value=None, status="unavailable", reason=reason, n=dict(n))

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": _clean(self.value), "status": self.status, "n": dict(self.n)}
        if self.ci95 is not None:
            d["ci95"] = [_clean(self.ci95[0]), _clean(self.ci95[1])]
        if self.reason:
            d["reason"] = self.reason
        if self.unit:
            d["unit"] = self.unit
        if self.low_n:
            d["low_n"] = True
        return d

    def fmt(self, digits: int = 3) -> str:
        if self.status != "ok" or self.value is None:
            return f"unavailable ({self.reason})" if self.reason else "unavailable"
        s = f"{self.value:.{digits}f}"
        if self.ci95 is not None:
            s += f" [{self.ci95[0]:.{digits}f}, {self.ci95[1]:.{digits}f}]"
        if self.n:
            s += " (" + ", ".join(f"n_{k}={v}" for k, v in self.n.items()) + ")"
        if self.low_n:
            s += " LOW-N"
        return s


def _clean(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates, ignoring NaN replicates (e.g. 0/0)."""
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return (float(np.quantile(s, a)), float(np.quantile(s, 1 - a)))


def bootstrap_indices(n_units: int, b: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> np.ndarray:
    """(b, n_units) array of unit indices sampled with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(b, n_units))


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, b: int = DEFAULT_B,
                    seed: int = DEFAULT_SEED) -> tuple[float, tuple[float, float]]:
    """Point estimate and CI of sum(num)/sum(den) with units resampled together.

    `num[i]`, `den[i]` are the sufficient statistics of unit i (e.g. detected events
    and positive events in clip i; false alerts and hours in clip i). A replicate whose
    denominator is 0 is NaN and dropped from the interval.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    tot = den.sum()
    point = float(num.sum() / tot) if tot > 0 else math.nan
    if num.size == 0:
        return point, (math.nan, math.nan)
    idx = bootstrap_indices(num.size, b, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return point, percentile_ci(reps)


def bootstrap_stat(units: list[Any], stat: Callable[[list[Any]], float], b: int = DEFAULT_B,
                   seed: int = DEFAULT_SEED) -> tuple[float, tuple[float, float]]:
    """Generic cluster bootstrap for statistics that aren't ratios of sums (AUC, percentiles)."""
    point = stat(units)
    if not units:
        return point, (math.nan, math.nan)
    idx = bootstrap_indices(len(units), b, seed)
    reps = np.array([stat([units[i] for i in row]) for row in idx], dtype=float)
    return point, percentile_ci(reps)

"""Journey event F1 (checkout_visit, store_exit): the inputs to the strongest cheap rule.

A predicted journey event matches a GT event of the same type on the same unit (camera
clip, or site for global journeys) when |t_pred - t_gt| <= `tol` seconds. Matching is
one-to-one and globally optimal per (unit, type) (Hungarian on time distance), so
duplicate predictions become false positives. If both sides carry a person key (GT
`actor_id` mapped to predicted ids by the caller, e.g. via the IDF1 mapping), pairs
with different keys don't match. F1 = 2TP / (2TP + FP + FN), additive per unit, so
the CI resamples units.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from eval.metrics.stats import DEFAULT_B, DEFAULT_SEED, LOW_N, MetricValue, bootstrap_ratio
from eval.metrics.tracking import linear_sum_assignment

JOURNEY_TYPES = ("checkout_visit", "store_exit")


@dataclass(frozen=True)
class JourneyEvent:
    unit: str
    type: str
    t: float
    person: str | None = None


def journey_f1(
    gt: list[JourneyEvent],
    pred: list[JourneyEvent],
    tol: float = 3.0,
    types: tuple[str, ...] = JOURNEY_TYPES,
    b: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
) -> dict[str, MetricValue]:
    out: dict[str, MetricValue] = {}
    for typ in types:
        g_by, p_by = defaultdict(list), defaultdict(list)
        for e in gt:
            if e.type == typ:
                g_by[e.unit].append(e)
        for e in pred:
            if e.type == typ:
                p_by[e.unit].append(e)
        if not g_by:
            out[typ] = MetricValue.unavailable(f"no GT {typ} labels")
            continue
        units = sorted(set(g_by) | set(p_by))
        tp, fp, fn = (np.zeros(len(units)) for _ in range(3))
        for k, u in enumerate(units):
            gs, ps = g_by.get(u, []), p_by.get(u, [])
            m = 0
            if gs and ps:
                d = np.array([[abs(p.t - g.t) for p in ps] for g in gs])
                ok = d <= tol + 1e-9
                for i, g in enumerate(gs):
                    for j, p in enumerate(ps):
                        if g.person is not None and p.person is not None and g.person != p.person:
                            ok[i, j] = False
                # maximize matches first, then minimize total time error: each valid pair
                # costs d - K with K larger than any possible sum of time errors
                big_k = (tol + 1.0) * (min(d.shape) + 1)
                r, c = linear_sum_assignment(np.where(ok, d - big_k, 1e9))
                m = int(sum(ok[i, j] for i, j in zip(r, c, strict=True)))
            tp[k], fp[k], fn[k] = m, len(ps) - m, len(gs) - m
        v, ci = bootstrap_ratio(2 * tp, 2 * tp + fp + fn, b, seed)
        n_gt = int(tp.sum() + fn.sum())
        out[typ] = MetricValue(
            v, ci, {"gt": n_gt, "pred": int(tp.sum() + fp.sum()), "units": len(units)}, low_n=n_gt < LOW_N
        )
    return out

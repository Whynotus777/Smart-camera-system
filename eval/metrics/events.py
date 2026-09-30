"""Event recall at a false-alert budget, the full recall-vs-FA curve, per-subtype recall.

Definitions (docs/EVAL.md "Primary"; hand-worked examples in tests/eval/test_events.py):

- Matching scope is a *unit*: one clip of one camera. Alerts never match events on
  another camera; a multi-camera alert is split into one prediction per camera.
- GT events have a `role`: `positive` (target type, `visible: observed` for this camera)
  or `ignore` (partially/not observed, or a type the suite neither targets nor
  penalizes). Predictions that overlap only ignore regions are neither true nor false.
- Overlap: `pred.t_start <= gt.t_end + tol` and `pred.t_end >= gt.t_start - tol`.
- One-to-one, greedy by score (COCO-style): predictions are processed by descending
  score (ties: earlier t_start, then input order). Each prediction takes the
  still-unmatched overlapping positive with the highest temporal IoU. So one long
  alert can't claim every event it touches.
- Outcomes: `tp` (matched a positive) · `duplicate` (overlaps positives, all already
  matched: review burden, not a false accusation) · `ignored` (overlaps only ignore
  regions) · `false` (overlaps nothing labeled).
- False-alert rate = (false [+ duplicate]) / hours of *continuous* footage. Duplicates
  count toward the budget by default because the budget is about review load; both
  counts are always reported separately.
- Because greedy decisions depend only on higher-scored predictions, the outcomes of
  the predictions kept at threshold θ are identical to the prefix of one full pass.
  One pass therefore yields the whole curve.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from eval.metrics.stats import DEFAULT_B, DEFAULT_SEED, LOW_N, MetricValue, bootstrap_ratio

Role = str  # "positive" | "ignore"


@dataclass(frozen=True)
class GTEvent:
    unit: str  # clip id on one camera: matching scope and bootstrap unit
    t_start: float
    t_end: float
    role: Role = "positive"
    type: str = ""
    subtype: str | None = None
    event_id: str = ""


@dataclass(frozen=True)
class Pred:
    unit: str
    t_start: float
    t_end: float
    score: float
    t_emit: float | None = None  # stream time the pipeline emitted it (latency)
    pred_id: str = ""


@dataclass
class MatchResult:
    """Outcome of one full greedy pass; slice it at any threshold."""

    preds: list[Pred]  # in processing order (descending score)
    outcome: list[str]  # per pred: tp | duplicate | ignored | false
    matched_gt: list[int]  # per pred: index into gts, or -1
    gts: list[GTEvent]
    unit_hours: dict[str, float] | None  # None = not continuous footage: no FA/h
    units: list[str] = field(default_factory=list)

    def kept(self, threshold: float) -> int:
        """Number of leading predictions with score >= threshold."""
        scores = np.array([p.score for p in self.preds], dtype=float)
        return int(np.searchsorted(-scores, -threshold, side="right"))


def _overlaps(p: Pred, g: GTEvent, tol: float) -> bool:
    return p.t_start <= g.t_end + tol and p.t_end >= g.t_start - tol


def _tiou(p: Pred, g: GTEvent) -> float:
    inter = min(p.t_end, g.t_end) - max(p.t_start, g.t_start)
    union = max(p.t_end, g.t_end) - min(p.t_start, g.t_start)
    if union <= 0:  # both instantaneous at the same time
        return 1.0
    return max(inter, 0.0) / union


def match(
    gts: Sequence[GTEvent],
    preds: Iterable[Pred],
    tol: float = 2.0,
    unit_hours: dict[str, float] | None = None,
) -> MatchResult:
    """Run the greedy pass. `unit_hours` must cover every unit if FA/h is wanted."""
    gts = list(gts)
    order = sorted(enumerate(preds), key=lambda ip: (-ip[1].score, ip[1].t_start, ip[0]))
    by_unit: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gts):
        if g.t_end < g.t_start:
            raise ValueError(f"GT event with t_end < t_start: {g}")
        by_unit[g.unit].append(i)
    taken: set[int] = set()
    out_preds, outcome, matched = [], [], []
    for _, p in order:
        if p.t_end < p.t_start:
            raise ValueError(f"prediction with t_end < t_start: {p}")
        cands = [i for i in by_unit.get(p.unit, ()) if _overlaps(p, gts[i], tol)]
        pos = [i for i in cands if gts[i].role == "positive"]
        free = [i for i in pos if i not in taken]
        if free:
            best = max(free, key=lambda i: (_tiou(p, gts[i]), -gts[i].t_start, -i))
            taken.add(best)
            res, m = "tp", best
        elif pos:
            res, m = "duplicate", -1
        elif cands:
            res, m = "ignored", -1
        else:
            res, m = "false", -1
        out_preds.append(p)
        outcome.append(res)
        matched.append(m)
    units = sorted({g.unit for g in gts} | {p.unit for p in out_preds} | set(unit_hours or {}))
    if unit_hours is not None:
        missing = [u for u in units if u not in unit_hours]
        if missing:
            raise ValueError(f"unit_hours missing units that have events/preds: {missing[:5]}")
    return MatchResult(out_preds, outcome, matched, gts, unit_hours, units)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    recall: MetricValue
    fa_per_hour: MetricValue  # false (+duplicates if counted) per camera-hour
    counts: dict[str, int]


def _unit_stats(r: MatchResult, k: int, dup_as_false: bool) -> dict[str, np.ndarray]:
    idx = {u: j for j, u in enumerate(r.units)}
    n = len(r.units)
    pos, det, fal, dup = (np.zeros(n) for _ in range(4))
    for g in r.gts:
        if g.role == "positive":
            pos[idx[g.unit]] += 1
    for p, o in zip(r.preds[:k], r.outcome[:k], strict=True):
        j = idx[p.unit]
        if o == "tp":
            det[j] += 1
        elif o == "false":
            fal[j] += 1
        elif o == "duplicate":
            dup[j] += 1
    hours = np.array([r.unit_hours[u] for u in r.units]) if r.unit_hours is not None else None
    burden = fal + dup if dup_as_false else fal
    return {
        "pos": pos,
        "det": det,
        "false": fal,
        "dup": dup,
        "burden": burden,
        **({"hours": hours} if hours is not None else {}),
    }


def operating_point(
    r: MatchResult, threshold: float, dup_as_false: bool = True, b: int = DEFAULT_B, seed: int = DEFAULT_SEED
) -> OperatingPoint:
    """Recall and FA/h (with clip-bootstrap CIs) at a fixed threshold."""
    k = r.kept(threshold)
    s = _unit_stats(r, k, dup_as_false)
    n_pos, n_units = int(s["pos"].sum()), len(r.units)
    counts = {
        "positives": n_pos,
        "detected": int(s["det"].sum()),
        "false": int(s["false"].sum()),
        "duplicates": int(s["dup"].sum()),
        "units": n_units,
        "alerts_kept": k,
    }
    if n_pos == 0:
        rec = MetricValue.unavailable("no positive events", units=n_units)
    else:
        v, ci = bootstrap_ratio(s["det"], s["pos"], b, seed)
        rec = MetricValue(v, ci, {"events": n_pos, "units": n_units}, low_n=n_pos < LOW_N)
    if "hours" not in s:
        fa = MetricValue.unavailable("not continuous footage: false alerts/hour undefined")
    else:
        v, ci = bootstrap_ratio(s["burden"], s["hours"], b, seed)
        hours = float(s["hours"].sum())
        fa = MetricValue(
            v,
            ci,
            {"units": n_units, "false_alerts": int(s["burden"].sum())},
            unit="per camera-hour",
            reason=f"{hours:.2f} camera-hours",
        )
    return OperatingPoint(threshold, rec, fa, counts)


def curve(r: MatchResult, dup_as_false: bool = True) -> list[dict[str, float]]:
    """Recall vs FA/h at every distinct score, plus the no-alert point (threshold=+inf)."""
    total_pos = sum(1 for g in r.gts if g.role == "positive")
    hours = sum(r.unit_hours.values()) if r.unit_hours is not None else None
    pts = [
        {
            "threshold": math.inf,
            "recall": 0.0 if total_pos else math.nan,
            "fa_per_hour": 0.0 if hours else math.nan,
            "false": 0,
            "detected": 0,
        }
    ]
    det = fal = 0
    for i, (p, o) in enumerate(zip(r.preds, r.outcome, strict=True)):
        det += o == "tp"
        fal += o == "false" or (dup_as_false and o == "duplicate")
        last_of_score = i + 1 == len(r.preds) or r.preds[i + 1].score != p.score
        if last_of_score:
            pts.append(
                {
                    "threshold": p.score,
                    "recall": det / total_pos if total_pos else math.nan,
                    "fa_per_hour": fal / hours if hours else math.nan,
                    "false": fal,
                    "detected": det,
                }
            )
    return pts


def threshold_at_budget(r: MatchResult, budget_per_hour: float, dup_as_false: bool = True) -> float:
    """Lowest threshold whose FA/h <= budget (= highest recall under budget; both are
    monotone in the threshold). +inf if even the top-scored alert breaks the budget.
    Fit this on val footage and freeze it for test (docs/EVAL.md Rules)."""
    if r.unit_hours is None:
        raise ValueError("threshold_at_budget needs continuous footage (unit_hours)")
    best = math.inf
    for pt in curve(r, dup_as_false):
        if pt["fa_per_hour"] <= budget_per_hour + 1e-12:
            best = pt["threshold"]
    return best


def per_subtype_recall(
    r: MatchResult, threshold: float, b: int = DEFAULT_B, seed: int = DEFAULT_SEED
) -> dict[str, MetricValue]:
    """Recall per positive subtype at `threshold`; unavailable if no positive has a subtype."""
    k = r.kept(threshold)
    hit = {m for m, o in zip(r.matched_gt[:k], r.outcome[:k], strict=True) if o == "tp"}
    pos = [(i, g) for i, g in enumerate(r.gts) if g.role == "positive"]
    if not pos or all(g.subtype is None for _, g in pos):
        return {"_all": MetricValue.unavailable("dataset has no subtype labels")}
    out: dict[str, MetricValue] = {}
    for st in sorted({g.subtype for _, g in pos if g.subtype is not None}):
        units = sorted({g.unit for _, g in pos if g.subtype == st})
        uj = {u: j for j, u in enumerate(units)}
        num, den = np.zeros(len(units)), np.zeros(len(units))
        for i, g in pos:
            if g.subtype == st:
                den[uj[g.unit]] += 1
                num[uj[g.unit]] += i in hit
        v, ci = bootstrap_ratio(num, den, b, seed)
        out[st] = MetricValue(v, ci, {"events": int(den.sum()), "units": len(units)}, low_n=den.sum() < LOW_N)
    n_unlabeled = sum(1 for _, g in pos if g.subtype is None)
    if n_unlabeled:
        out["_no_subtype"] = MetricValue.unavailable("positives without a subtype label", events=n_unlabeled)
    return out


def latencies(r: MatchResult, threshold: float) -> np.ndarray:
    """Emit time minus GT event end, for true positives that carry `t_emit` (seconds)."""
    k = r.kept(threshold)
    out = [
        p.t_emit - r.gts[m].t_end
        for p, o, m in zip(r.preds[:k], r.outcome[:k], r.matched_gt[:k], strict=True)
        if o == "tp" and p.t_emit is not None
    ]
    return np.array(out, dtype=float)

"""Hand-computed toy cases for event recall @ FA budget, the curve, and per-subtype recall.

Each case states the expected numbers worked out by hand in the comments.
"""

import math

import numpy as np
import pytest

from eval.metrics.events import (
    GTEvent,
    Pred,
    curve,
    latencies,
    match,
    operating_point,
    per_subtype_recall,
    threshold_at_budget,
)


def G(unit, a, b, role="positive", subtype=None):  # noqa: N802
    return GTEvent(unit, a, b, role=role, subtype=subtype)


def P(unit, a, b, s, emit=None):  # noqa: N802
    return Pred(unit, a, b, s, t_emit=emit)


# Clip "c1" (1 h): positives A=[10,20], B=[15,25] (overlapping), C=[100,110]; ignore region D=[200,210].
# Clip "c2" (1 h): positive E=[50,60].
GTS = [G("c1", 10, 20), G("c1", 15, 25), G("c1", 100, 110), G("c1", 200, 210, role="ignore"), G("c2", 50, 60)]
HOURS = {"c1": 1.0, "c2": 1.0}
PREDS = [
    P("c1", 12, 22, 0.9),  # overlaps A and B -> takes the higher tIoU: A (8/12) vs B (7/13) -> A. tp
    P("c1", 11, 19, 0.8),  # overlaps A, B; B still free -> tp (B)
    P("c1", 14, 18, 0.7),  # overlaps A, B, both taken -> duplicate
    P("c1", 203, 205, 0.6),  # only the ignore region -> ignored
    P("c1", 300, 310, 0.5),  # nothing -> false
    P("c2", 55, 56, 0.4),  # E -> tp
    P("c2", 58, 59, 0.4),  # E taken -> duplicate (tie on score: earlier t_start processed first)
    P("c1", 104, 105, 0.1),  # C -> tp at a low score
]


def test_greedy_outcomes_overlap_duplicates_ignore():
    r = match(GTS, PREDS, tol=0.0, unit_hours=HOURS)
    got = [(p.score, p.t_start, o) for p, o in zip(r.preds, r.outcome, strict=True)]
    assert got == [
        (0.9, 12, "tp"), (0.8, 11, "tp"), (0.7, 14, "duplicate"), (0.6, 203, "ignored"),
        (0.5, 300, "false"), (0.4, 55, "tp"), (0.4, 58, "duplicate"), (0.1, 104, "tp"),
    ]
    # the 0.9 alert took A (index 0), 0.8 took B (index 1)
    assert r.matched_gt[:2] == [0, 1]


def test_operating_point_counts_and_rates():
    r = match(GTS, PREDS, tol=0.0, unit_hours=HOURS)
    # threshold 0.4: kept = first 7. positives = 4 (A,B,C,E). detected A,B,E = 3.
    # false = 1 (300-310); duplicates = 2 (0.7 and 0.4@58); hours = 2.
    op = operating_point(r, 0.4, dup_as_false=True, b=200)
    assert op.counts == {"positives": 4, "detected": 3, "false": 1, "duplicates": 2, "units": 2,
                         "alerts_kept": 7}
    assert op.recall.value == pytest.approx(3 / 4)
    assert op.fa_per_hour.value == pytest.approx((1 + 2) / 2)
    op2 = operating_point(r, 0.4, dup_as_false=False, b=200)
    assert op2.fa_per_hour.value == pytest.approx(1 / 2)
    lo, hi = op.recall.ci95
    assert 0 <= lo <= op.recall.value <= hi <= 1
    assert op.recall.n == {"events": 4, "units": 2} and op.recall.low_n


def test_tolerance_extends_overlap():
    # alert ends 1.5 s before E starts: miss at tol=0, hit at tol=2
    gts = [G("c2", 50, 60)]
    preds = [P("c2", 40, 48.5, 0.9)]
    assert match(gts, preds, tol=0.0).outcome == ["false"]
    assert match(gts, preds, tol=2.0).outcome == ["tp"]


def test_one_long_alert_cannot_claim_every_event():
    gts = [G("u", 0, 1), G("u", 10, 11), G("u", 20, 21)]
    r = match(gts, [P("u", 0, 21, 0.9)], tol=0, unit_hours={"u": 1})
    assert operating_point(r, 0.5, b=10).counts["detected"] == 1


def test_alerts_never_match_across_units():
    r = match([G("c1", 0, 10)], [P("c2", 0, 10, 0.9)], unit_hours={"c1": 1, "c2": 1})
    assert r.outcome == ["false"]


def test_curve_and_budget_threshold():
    r = match(GTS, PREDS, tol=0.0, unit_hours=HOURS)
    pts = curve(r, dup_as_false=True)
    # (threshold, recall, fa/h) by hand; 4 positives, 2 hours
    expect = [
        (math.inf, 0.0, 0.0),
        (0.9, 1 / 4, 0.0),
        (0.8, 2 / 4, 0.0),
        (0.7, 2 / 4, 0.5),  # duplicate counts toward the budget
        (0.6, 2 / 4, 0.5),  # ignored: no change
        (0.5, 2 / 4, 1.0),
        (0.4, 3 / 4, 1.5),  # tie at 0.4: both preds enter together (tp + duplicate)
        (0.1, 4 / 4, 1.5),
    ]
    assert [(p["threshold"], p["recall"], p["fa_per_hour"]) for p in pts] == pytest.approx(expect)
    assert threshold_at_budget(r, 0.0) == 0.8
    assert threshold_at_budget(r, 1.0) == 0.5
    assert threshold_at_budget(r, 1.5) == 0.1
    # without duplicates in the budget, 0.6 already allows 0.0 FA/h
    assert threshold_at_budget(r, 0.0, dup_as_false=False) == 0.6


def test_budget_unreachable_returns_inf():
    r = match([G("u", 0, 1)], [P("u", 50, 51, 0.9)], unit_hours={"u": 10})
    assert threshold_at_budget(r, 0.01) == math.inf
    op = operating_point(r, math.inf, b=10)
    assert op.recall.value == 0.0 and op.fa_per_hour.value == 0.0


def test_fa_rate_unavailable_without_continuous_footage():
    r = match(GTS, PREDS, tol=0.0, unit_hours=None)
    op = operating_point(r, 0.4, b=10)
    assert op.fa_per_hour.status == "unavailable"
    assert op.recall.value == pytest.approx(0.75)
    with pytest.raises(ValueError):
        threshold_at_budget(r, 1.0)


def test_no_positives_is_unavailable_not_zero():
    r = match([G("u", 0, 1, role="ignore")], [P("u", 0, 1, 0.9)], unit_hours={"u": 1})
    assert operating_point(r, 0.5, b=10).recall.status == "unavailable"


def test_missing_unit_hours_is_an_error():
    with pytest.raises(ValueError, match="unit_hours"):
        match([G("c1", 0, 1)], [P("c9", 0, 1, 0.5)], unit_hours={"c1": 1})


def test_per_subtype_recall_and_unavailable():
    gts = [G("u1", 0, 1, subtype="pocket"), G("u1", 5, 6, subtype="pocket"), G("u2", 0, 1, subtype="bag")]
    preds = [P("u1", 0, 1, 0.9), P("u2", 0, 1, 0.2)]
    r = match(gts, preds, tol=0)
    st = per_subtype_recall(r, 0.5, b=50)
    assert st["pocket"].value == pytest.approx(0.5) and st["bag"].value == 0.0
    st_all = per_subtype_recall(r, 0.1, b=50)
    assert st_all["bag"].value == 1.0
    r2 = match([G("u", 0, 1)], preds[:1], tol=0)
    assert per_subtype_recall(r2, 0.5)["_all"].status == "unavailable"


def test_latency_from_emit_time():
    r = match([G("u", 10, 20)], [P("u", 12, 25, 0.9, emit=23.5)], tol=0)
    assert latencies(r, 0.5).tolist() == [3.5]


def test_bootstrap_ci_is_clip_level_and_deterministic():
    # 10 clips, 1 positive each, detected in 7 -> recall 0.7; CI must be reproducible
    gts = [G(f"c{i}", 0, 1) for i in range(10)]
    preds = [P(f"c{i}", 0, 1, 0.9) for i in range(7)]
    hours = {f"c{i}": 1.0 for i in range(10)}
    a = operating_point(match(gts, preds, unit_hours=hours), 0.5)
    b = operating_point(match(gts, preds, unit_hours=hours), 0.5)
    assert a.recall.value == pytest.approx(0.7) and a.recall.ci95 == b.recall.ci95
    lo, hi = a.recall.ci95
    # binomial-ish spread over 10 clips: a wide interval that contains 0.7
    assert lo < 0.7 < hi and hi - lo > 0.3
    assert np.isfinite([lo, hi]).all()

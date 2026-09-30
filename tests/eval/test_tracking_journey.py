"""IDF1 / ID switches / MOTA / detection AP / journey F1 on hand-computed toy sequences."""

import itertools

import numpy as np
import pytest

from eval.metrics.journey import JourneyEvent, journey_f1
from eval.metrics.tracking import (
    TrackFrames,
    _lsa_numpy,
    detection_ap,
    iou_matrix,
    sequence_stats,
    tracking_metrics,
)


def box(x):
    return (x, 0, x + 10, 10)


def seq(rows):
    return TrackFrames.from_rows([(f, i, *b) for f, i, b in rows])


def test_lsa_matches_brute_force_and_scipy():
    rng = np.random.default_rng(1)
    for shape in [(3, 3), (2, 4), (4, 2), (5, 5), (1, 3)]:
        c = rng.random(shape)
        r, cc = _lsa_numpy(c)
        n = min(shape)
        best = min(
            sum(c[i, j] for i, j in zip(rows, cols, strict=True))
            for rows in itertools.combinations(range(shape[0]), n)
            for cols in itertools.permutations(range(shape[1]), n)
        )
        assert c[r, cc].sum() == pytest.approx(best)
    sp = pytest.importorskip("scipy.optimize")
    for _ in range(20):
        c = rng.random((rng.integers(1, 9), rng.integers(1, 9)))
        r1, c1 = _lsa_numpy(c)
        r2, c2 = sp.linear_sum_assignment(c)
        assert c[r1, c1].sum() == pytest.approx(c[r2, c2].sum())


def test_perfect_tracking():
    gt = seq([(f, 1, box(0)) for f in range(10)])
    s = sequence_stats(gt, gt, fps=10)
    m = tracking_metrics([s], b=10)
    assert m["idf1"].value == 1.0 and m["id_switches"].value == 0 and m["mota"].value == 1.0


def test_one_id_switch_halfway():
    # GT id 1 for 10 frames; prediction id 7 for frames 0-4, id 8 for frames 5-9.
    # CLEAR: 10 TP, 1 switch (at frame 5), MOTA = (10 - 0 - 0 - 1)/10 = 0.9
    # IDF1: best one-to-one mapping 1->7 gives IDTP = 5; IDF1 = 2*5/(10+10) = 0.5
    gt = seq([(f, 1, box(0)) for f in range(10)])
    pr = seq([(f, 7 if f < 5 else 8, box(0)) for f in range(10)])
    s = sequence_stats(gt, pr, fps=10)
    assert (s.tp, s.fp, s.fn, s.idsw, s.idtp) == (10, 0, 0, 1, 5)
    m = tracking_metrics([s], b=10)
    assert m["idf1"].value == pytest.approx(0.5)
    assert m["mota"].value == pytest.approx(0.9)
    # 1 switch over 10 frames at 10 fps = 1 s = 1/60 person-minute -> 60 per person-minute
    assert m["idsw_per_person_min"].value == pytest.approx(60.0)


def test_switch_survives_a_gap_and_returning_id_is_a_switch():
    # pred 7 on frames 0-2, missing 3-4 (FN), pred 9 on 5-6, pred 7 back on 7-8: two switches
    gt = seq([(f, 1, box(0)) for f in range(9)])
    ids = {0: 7, 1: 7, 2: 7, 5: 9, 6: 9, 7: 7, 8: 7}
    pr = seq([(f, i, box(0)) for f, i in ids.items()])
    s = sequence_stats(gt, pr, fps=10)
    assert (s.tp, s.fn, s.fp, s.idsw) == (7, 2, 0, 2)
    # IDTP: 1->7 covers 5 frames. IDF1 = 2*5/(9+7)
    assert s.idtp == 5


def test_low_iou_is_fp_plus_fn():
    gt = seq([(0, 1, (0, 0, 10, 10))])
    pr = seq([(0, 1, (6, 0, 16, 10))])  # IoU = 4/16 = 0.25 < 0.5
    s = sequence_stats(gt, pr)
    assert (s.tp, s.fp, s.fn, s.idtp) == (0, 1, 1, 0)


def test_crossing_tracks_keep_previous_correspondence():
    # two GTs with identical boxes on frame 1 (overlap): motmetrics keeps last matches -> no switch
    xs = [(0, 40), (20, 20), (40, 0)]  # per frame: (x of id a, x of id b)
    gt = seq([(f, i, box(x)) for f, pair in enumerate(xs) for i, x in zip((1, 2), pair, strict=True)])
    pr = seq([(f, i, box(x)) for f, pair in enumerate(xs) for i, x in zip((5, 6), pair, strict=True)])
    s = sequence_stats(gt, pr)
    assert s.idsw == 0 and s.idtp == 6


@pytest.mark.parametrize("seed", range(3))
def test_matches_motmetrics(seed):
    mm = pytest.importorskip("motmetrics")
    rng = np.random.default_rng(seed)
    gt_rows, pr_rows = [], []
    for f in range(40):
        for g in range(4):
            x = 30 * g + f * 0.5
            gt_rows.append((f, g, x, 0, x + 20, 20))
            if rng.random() < 0.85:
                j = rng.normal(0, 3, 2)
                pid = g if rng.random() > 0.08 else g + 10 + f  # occasional identity noise
                pr_rows.append((f, pid, x + j[0], j[1], x + j[0] + 20, j[1] + 20))
    gt, pr = TrackFrames.from_rows(gt_rows), TrackFrames.from_rows(pr_rows)
    ours = sequence_stats(gt, pr, iou=0.5)
    assert sequence_stats(gt, pr, iou=0.5, use_scipy=False) == ours  # CI has no scipy
    acc = mm.MOTAccumulator(auto_id=False)
    for f in range(40):
        gi, pi = gt.frame == f, pr.frame == f
        # motmetrics 1.4's own iou_matrix uses np.asfarray (gone in numpy 2): feed it the same
        # 1 - IoU distances it would compute, NaN beyond max_iou=0.5. Matching, switches and
        # IDF1 are still motmetrics' own code.
        d = 1.0 - iou_matrix(gt.boxes[gi], pr.boxes[pi])
        d[d > 0.5] = np.nan
        acc.update(gt.ids[gi].tolist(), pr.ids[pi].tolist(), d, frameid=f)
    summ = mm.metrics.create().compute(acc, metrics=["idf1", "num_switches", "mota", "num_false_positives",
                                                     "num_misses"], name="x")
    assert ours.idsw == int(summ["num_switches"].iloc[0])
    assert ours.fp == int(summ["num_false_positives"].iloc[0])
    assert ours.fn == int(summ["num_misses"].iloc[0])
    idf1 = 2 * ours.idtp / (ours.n_gt + ours.n_pred)
    assert idf1 == pytest.approx(float(summ["idf1"].iloc[0]))


def test_detection_ap_hand():
    # 2 GT boxes in frame 0. Preds: 0.9 hits gt A, 0.8 duplicate of A (FP), 0.7 hits gt B.
    # PR points: (R.5,P1), (R.5,P.5), (R1,P2/3). envelope: P=1 up to R=.5, P=2/3 to R=1
    # AP = .5*1 + .5*2/3
    g = TrackFrames.from_rows([(0, 1, 0, 0, 10, 10), (0, 2, 50, 0, 60, 10)])
    p = TrackFrames.from_rows([(0, 1, 0, 0, 10, 10, 0.9), (0, 2, 1, 0, 11, 10, 0.8),
                               (0, 3, 50, 0, 60, 10, 0.7)])
    assert detection_ap([g], [p]).value == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_duplicate_rows_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        seq([(0, 1, box(0)), (0, 1, box(5))])


def test_journey_f1_hand():
    gt = [JourneyEvent("c", "store_exit", 10), JourneyEvent("c", "store_exit", 50),
          JourneyEvent("c", "checkout_visit", 30)]
    pred = [JourneyEvent("c", "store_exit", 11), JourneyEvent("c", "store_exit", 12),  # duplicate -> FP
            JourneyEvent("c", "store_exit", 90)]  # FP; GT@50 missed
    out = journey_f1(gt, pred, tol=3, b=20)
    # store_exit: TP 1, FP 2, FN 1 -> F1 = 2/(2+2+1) = 0.4
    assert out["store_exit"].value == pytest.approx(0.4)
    # checkout: TP 0, FN 1 -> 0
    assert out["checkout_visit"].value == 0.0
    assert journey_f1([], pred)["store_exit"].status == "unavailable"


def test_journey_prefers_more_matches_over_smaller_error():
    # GT at 0 and 4; preds at 2 and 5 (tol 3). Optimal: 0<->2, 4<->5 (2 matches)
    gt = [JourneyEvent("c", "store_exit", 0), JourneyEvent("c", "store_exit", 4)]
    pred = [JourneyEvent("c", "store_exit", 2), JourneyEvent("c", "store_exit", 5)]
    assert journey_f1(gt, pred, tol=3, b=5)["store_exit"].value == 1.0


def test_journey_person_key_blocks_match():
    gt = [JourneyEvent("c", "store_exit", 10, person="a")]
    pred = [JourneyEvent("c", "store_exit", 10, person="b")]
    assert journey_f1(gt, pred, b=5)["store_exit"].value == 0.0

"""Frame AUC-ROC/PR, clip score aggregation, and latency percentiles against hand values."""

import math

import numpy as np
import pytest

from eval.metrics.frame import (
    auc_roc,
    average_precision,
    frame_auc,
    frame_scores_from_tracks,
    gaussian_smooth,
)
from eval.metrics.latency import latency_summary


def test_auc_roc_hand_with_ties():
    # positives scores {0.8, 0.4}, negatives {0.4, 0.1}
    # pairs: (0.8>0.4)=1, (0.8>0.1)=1, (0.4 vs 0.4)=0.5, (0.4>0.1)=1 -> 3.5/4
    y = np.array([1, 1, 0, 0])
    s = np.array([0.8, 0.4, 0.4, 0.1])
    assert auc_roc(y, s) == pytest.approx(0.875)


def test_average_precision_hand():
    # sorted: 0.9(+) 0.8(-) 0.7(+) 0.6(-); P@R: R=.5 P=1 ; R=1 P=2/3 -> AP = .5*1 + .5*2/3
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert average_precision(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)
    # ties form one threshold: all 4 at 0.5 -> P=0.5 at R=1 -> AP=0.5
    assert average_precision(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_auc_degenerate_is_nan_and_unavailable():
    assert math.isnan(auc_roc(np.zeros(3), np.arange(3)))
    out = frame_auc([(np.zeros(3), np.arange(3.0))], b=10)
    assert out["auc_roc"].status == "unavailable"


def test_frame_auc_counts_and_ci():
    clips = [
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])),
        (np.array([0, 1, 0, 0]), np.array([0.3, 0.6, 0.2, 0.1])),
    ]
    out = frame_auc(clips, b=200)
    assert out["auc_roc"].value == pytest.approx(1.0)
    assert out["auc_roc"].n == {"frames": 8, "positive_frames": 3, "clips": 2, "positive_clips": 2}
    with pytest.raises(ValueError, match="NaN"):
        frame_auc([(np.array([0, 1]), np.array([0.1, np.nan]))])


@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn(seed):
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 2)  # many ties
    assert auc_roc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))


def test_frame_scores_max_over_tracks_and_fill():
    # 5 frames; track a scores frames 0,1; track b frames 1,2; frames 3,4 empty -> fill=min=0.1
    fi = np.array([0, 1, 1, 2])
    sc = np.array([0.1, 0.5, 0.7, 0.3])
    assert frame_scores_from_tracks(5, fi, sc).tolist() == pytest.approx([0.1, 0.7, 0.3, 0.1, 0.1])
    assert frame_scores_from_tracks(3, np.array([], dtype=int), np.array([])).tolist() == [0, 0, 0]


def test_gaussian_matches_scipy():
    nd = pytest.importorskip("scipy.ndimage")
    x = np.random.default_rng(0).random(50)
    for sigma in (0.7, 3, 40):
        assert gaussian_smooth(x, sigma) == pytest.approx(nd.gaussian_filter1d(x, sigma))


def test_latency_percentiles_hand():
    v = np.arange(1, 11, dtype=float)  # 1..10
    out = latency_summary(v, clock="event end -> alert emit (stream)", b=100)
    # numpy linear: p50 = 5.5, p90 = 1+0.9*9 = 9.1, p95 = 9.55, max 10
    assert out["p50"].value == pytest.approx(5.5)
    assert out["p90"].value == pytest.approx(9.1)
    assert out["p95"].value == pytest.approx(9.55)
    assert out["max"].value == 10
    assert out["p95"].n == {"samples": 10} and out["p95"].low_n
    assert latency_summary(np.array([]), clock="x")["p95"].status == "unavailable"


def test_fast_weighted_bootstrap_equals_naive_concatenation():
    from eval.metrics.frame import _WeightedBlocks, bootstrap_counts
    from eval.metrics.stats import bootstrap_indices

    rng = np.random.default_rng(3)
    clips = [(rng.integers(0, 2, n), np.round(rng.random(n), 1)) for n in rng.integers(5, 40, 12)]
    y = np.concatenate([c[0] for c in clips]).astype(bool)
    s = np.concatenate([c[1] for c in clips])
    wb = _WeightedBlocks(y, s, np.repeat(np.arange(12), [len(c[0]) for c in clips]))
    for row, cnt in zip(bootstrap_indices(12, 30, 7), bootstrap_counts(12, 30, 7), strict=True):
        yy = np.concatenate([clips[i][0] for i in row])
        sc = np.concatenate([clips[i][1] for i in row])
        assert wb.auc(cnt) == pytest.approx(auc_roc(yy, sc), nan_ok=True)
        assert wb.ap(cnt) == pytest.approx(average_precision(yy, sc), nan_ok=True)

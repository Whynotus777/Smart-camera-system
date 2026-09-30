"""Trackers on synthetic scenes: assignment, Kalman, ByteTrack behaviour, appearance re-ID, x-cam TTL."""

import itertools

import numpy as np
import pytest

from scs.contracts import Detection, FrameRef, Track
from scs.perception.base import Tracker
from scs.perception.track import KalmanFilterXYAH, KalmanFilterXYWH, _hungarian, linear_assignment
from scs.perception.track_botsort import AppearanceTracker
from scs.perception.track_bytetrack import ByteTrackConfig, ByteTracker

W, H = 640, 480


def _ref(seq, cam="cam0", fps=10.0):
    return FrameRef(camera_id=cam, epoch=0, seq=seq, frame_idx=seq, ts=1000 + seq / fps, width=W, height=H)


def _det(ref, box, score=0.9):
    return Detection(frame=ref, bbox=tuple(float(v) for v in box), score=score)


def _canvas(people):
    """people: list of (box, rgb) → HxWx3 uint8 frame with solid-colour people."""
    img = np.full((H, W, 3), 128, np.uint8)
    for (x1, y1, x2, y2), rgb in people:
        img[int(y1) : int(y2), int(x1) : int(x2)] = rgb
    return img


# ------------------------------------------------------------------ assignment


@pytest.mark.parametrize("shape", [(3, 3), (4, 6), (6, 4), (1, 5), (5, 1)])
def test_hungarian_is_optimal(shape):
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.random(shape)
        r, k = _hungarian(c)
        n = min(shape)
        if shape[0] <= shape[1]:
            best = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(shape[1]), n))
        else:
            best = min(sum(c[p[j], j] for j in range(n)) for p in itertools.permutations(range(shape[0]), n))
        assert len(r) == n and c[r, k].sum() == pytest.approx(best)


def test_linear_assignment_threshold():
    c = np.array([[0.1, 0.9], [0.95, 0.99]])
    m, ua, ub = linear_assignment(c, 0.5)
    assert m == [(0, 0)] and ua == [1] and ub == [1]
    assert linear_assignment(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])


@pytest.mark.parametrize("kf", [KalmanFilterXYAH(), KalmanFilterXYWH()])
def test_kalman_tracks_constant_velocity(kf):
    tlwh = np.array([100.0, 100, 50, 120])
    mean, cov = kf.initiate(kf.to_meas(tlwh))
    for _ in range(20):
        tlwh = tlwh + [5, 0, 0, 0]
        m, c = kf.multi_predict(mean[None], cov[None])
        mean, cov = kf.update(m[0], c[0], kf.to_meas(tlwh))
    m, _ = kf.multi_predict(mean[None], cov[None])
    np.testing.assert_allclose(kf.to_tlwh(m[0])[:2], tlwh[:2] + [5, 0], atol=1.0)


# ------------------------------------------------------------------ trackers


@pytest.mark.parametrize("make", [ByteTracker, AppearanceTracker])
def test_protocol_and_stable_ids(make):
    trk = make()
    assert isinstance(trk, Tracker)
    ids = set()
    for s in range(30):
        ref = _ref(s)
        a, b = (50 + 4 * s, 100, 110 + 4 * s, 260), (400, 120, 460, 300)
        out = trk.update([_det(ref, a), _det(ref, b)], _canvas([(a, (200, 30, 30)), (b, (30, 30, 200))]))
        assert all(isinstance(t, Track) and t.frame.identity == ref.identity for t in out)
        if s > 1:
            assert len(out) == 2
            ids |= {t.track_id for t in out}
    assert len(ids) == 2


def test_tracks_are_confirmed_after_two_frames_and_empty_frames_ok():
    trk = ByteTracker()
    box = (100, 100, 160, 260)
    assert trk.update([], None) == []
    assert trk.update([_det(_ref(1), box)], None) == []  # tentative until matched again
    out = trk.update([_det(_ref(2), box)], None)
    assert len(out) == 1 and out[0].state == "confirmed"


def test_low_score_detections_keep_tracks_alive():
    trk = ByteTracker()
    for s in range(10):
        out = trk.update([_det(_ref(s), (100 + s, 100, 160 + s, 260), 0.9)], None)
    tid = out[0].track_id
    for s in range(10, 15):  # partial occlusion: detector confidence drops to 0.3
        out = trk.update([_det(_ref(s), (100 + s, 100, 160 + s, 260), 0.3)], None)
        assert [t.track_id for t in out] == [tid]


def _shelf_occlusion(trk, gap_frames=25, fps=10.0):
    """Red walks right, hides behind a shelf for `gap_frames`, reappears further right.
    Blue stands still nearby the whole time. Returns (ids before, ids after) for red."""
    red, blue = (220, 40, 40), (40, 40, 220)
    before, after = set(), set()
    s = 0
    for k in range(15):
        ref = _ref(s, fps=fps)
        r, b = (60 + 6 * k, 100, 120 + 6 * k, 280), (420, 110, 480, 300)
        out = trk.update([_det(ref, r), _det(ref, b)], _canvas([(r, red), (b, blue)]))
        before |= {t.track_id for t in out if abs(t.bbox[0] - r[0]) < 20}
        s += 1
    for _ in range(gap_frames):
        ref = _ref(s, fps=fps)
        b = (420, 110, 480, 300)
        trk.update([_det(ref, b)], _canvas([(b, blue)]))
        s += 1
    for k in range(10):  # reappears ~1.5 body-widths right of where it vanished, not where KF expects
        ref = _ref(s, fps=fps)
        r, b = (250 + 3 * k, 105, 310 + 3 * k, 285), (420, 110, 480, 300)
        out = trk.update([_det(ref, r), _det(ref, b)], _canvas([(r, red), (b, blue)]))
        after |= {t.track_id for t in out if abs(t.bbox[0] - r[0]) < 20}
        s += 1
    return before, after


def test_bytetrack_switches_id_after_long_occlusion():
    before, after = _shelf_occlusion(ByteTracker(ByteTrackConfig(frame_rate=10)))
    assert before and after and before.isdisjoint(after)  # the failure mode the appearance tracker targets

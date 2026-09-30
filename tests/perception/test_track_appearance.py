"""Appearance-aware tracker (occlusion + re-entry re-ID), colour embedder, cross-camera associator."""

import numpy as np
import pytest
from test_track import H, W, _canvas, _det, _ref, _shelf_occlusion

from scs.contracts import FrameRef, Track
from scs.perception.track_botsort import AppearanceTrackConfig, AppearanceTracker, HsvEmbedder
from scs.perception.track_xcam import CrossCameraAssociator


def test_appearance_tracker_recovers_id_after_occlusion():
    before, after = _shelf_occlusion(AppearanceTracker())
    assert before == after and len(before) == 1


def _reentry(trk, away_frames=80):
    """Red leaves the view on the left, stays out for 8 s, comes back in. Blue stays throughout."""
    red, blue = (220, 40, 40), (40, 40, 220)
    b = (420, 110, 480, 300)
    ids_before, ids_after = set(), set()
    s = 0
    for k in range(20):
        r = (200 - 10 * k, 100, 260 - 10 * k, 280)
        out = trk.update([_det(_ref(s), r), _det(_ref(s), b)], _canvas([(r, red), (b, blue)]))
        ids_before |= {t.track_id for t in out if t.bbox[0] < 300}
        s += 1
    for _ in range(away_frames):
        trk.update([_det(_ref(s), b)], _canvas([(b, blue)]))
        s += 1
    for k in range(10):
        r = (10 + 8 * k, 100, 70 + 8 * k, 280)
        out = trk.update([_det(_ref(s), r), _det(_ref(s), b)], _canvas([(r, red), (b, blue)]))
        ids_after |= {t.track_id for t in out if t.bbox[0] < 300}
        s += 1
    return ids_before, ids_after


def test_reentry_needs_long_term_memory():
    before, after = _reentry(AppearanceTracker(AppearanceTrackConfig(frame_rate=10)))
    assert before.isdisjoint(after)  # default: expired after the lost buffer
    before, after = _reentry(AppearanceTracker(AppearanceTrackConfig(frame_rate=10, reid_memory_s=30)))
    assert before == after and len(before) == 1


def test_reid_memory_is_bounded():
    with pytest.raises(ValueError):
        AppearanceTracker(AppearanceTrackConfig(reid_memory_s=3600))


def test_appearance_tracker_does_not_steal_lookalike_far_away():
    trk = AppearanceTracker()
    col = (200, 40, 40)
    for s in range(10):
        a = (60, 100, 120, 280)
        out = trk.update([_det(_ref(s), a)], _canvas([(a, col)]))
    tid = out[0].track_id
    for s in range(10, 13):  # person A gone; same-colour person B appears across the room
        b = (560, 100, 620, 280)
        out = trk.update([_det(_ref(s), b)], _canvas([(b, col)]))
    assert out and all(t.track_id != tid for t in out)


def test_hsv_embedder_separates_colours_and_handles_chw():
    emb = HsvEmbedder()
    img = _canvas(
        [
            ((0, 0, 100, 200), (220, 40, 40)),
            ((200, 0, 300, 200), (40, 40, 220)),
            ((400, 0, 500, 200), (225, 45, 38)),
        ]
    )
    f = emb.embed(img, np.array([[0, 0, 100, 200], [200, 0, 300, 200], [400, 0, 500, 200]]), (W, H))
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    assert f[0] @ f[2] > 0.9 > f[0] @ f[1]
    chw = np.ascontiguousarray(img.transpose(2, 0, 1))
    f2 = emb.embed(chw, np.array([[0, 0, 100, 200]]), (W, H))
    np.testing.assert_allclose(f2[0] / np.linalg.norm(f2[0]), f[0], atol=1e-6)


# ------------------------------------------------------------------ cross-camera


def _trk(cam, tid, ts):
    ref = FrameRef(camera_id=cam, epoch=0, seq=0, frame_idx=0, ts=ts, width=W, height=H)
    return Track(frame=ref, track_id=tid, bbox=(0, 0, 10, 10), score=0.9)


def test_xcam_disabled_is_passthrough():
    a = CrossCameraAssociator()
    t = [_trk("c1", 1, 0.0)]
    assert a.assign(t, np.ones((1, 4))) is t and len(a) == 0


def test_xcam_links_across_cameras_and_expires():
    a = CrossCameraAssociator(enabled=True, ttl_s=60, min_track_frames=2)
    red, blue = np.array([1.0, 0, 0, 0]), np.array([0, 0, 1.0, 0])
    for k in range(3):
        out = a.assign([_trk("door", 1, k), _trk("door", 2, k)], np.stack([red, blue]))
    g_red, g_blue = out[0].global_id, out[1].global_id
    assert g_red is not None and g_blue is not None and g_red != g_blue
    for k in range(3):
        out = a.assign([_trk("shelf", 7, 10 + k)], (red + 0.05)[None])
    assert out[0].global_id == g_red
    a.assign([_trk("shelf", 7, 200)], red[None])  # 190 s later: every identity has expired
    assert len(a) == 0
    with pytest.raises(ValueError):
        CrossCameraAssociator(ttl_s=3600)

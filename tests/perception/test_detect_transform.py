"""ADR 0002 round trip: detector-input pixels ↔ main-stream pixels, for every resize mode."""

import numpy as np
import pytest

from scs.contracts import FrameRef
from scs.perception.detect import DetectorConfig, InputTransform, iou_matrix, to_detections


def _ref(w=2560, h=1440, **kw):
    return FrameRef(camera_id="cam0", epoch=0, seq=7, frame_idx=7, ts=1.0, width=w, height=h, **kw)


@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
@pytest.mark.parametrize("src", [(2560, 1440), (1920, 1080), (640, 360), (704, 480), (1080, 1920)])
def test_round_trip(mode, src):
    tf = InputTransform.fit(src[0], src[1], 640, mode)
    w, h = src
    boxes = np.array([[10.0, 20.0, 0.4 * w, 0.9 * h], [0.0, 0.0, w, h], [w - 50, 5, w, 60]])
    back = tf.to_main(tf.to_input(boxes))
    np.testing.assert_allclose(back, boxes, atol=1e-6)


def test_letterbox_geometry_2560x1440():
    tf = InputTransform.fit(2560, 1440, 640, "letterbox")
    assert tf.resized_wh == (640, 360)
    assert (tf.pad_x, tf.pad_y) == (0.0, 140.0)
    # the full padded image maps to the whole frame (padding clipped away)
    np.testing.assert_allclose(tf.to_main(np.array([[0, 0, 640, 640]])), [[0, 0, 2560, 1440]])
    assert tf.tag == "resize640x640:letterbox"


def test_substream_image_maps_to_main_stream():
    # sub-stream 640x360 decoded, main stream is 2560x1440: boxes must come out in main pixels
    tf = InputTransform.fit(640, 360, 640, "letterbox", main_wh=(2560, 1440))
    out = tf.to_main(np.array([[0, 140, 320, 320]]))  # left half, top half of the sub-stream image
    np.testing.assert_allclose(out, [[0, 0, 1280, 720]])


def test_to_detections_filters_and_tags():
    ref = _ref()
    tf = InputTransform.fit(2560, 1440, 640, "letterbox")
    raw = np.array(
        [
            [100, 200, 160, 400, 0.9],
            [100, 200, 101, 201, 0.95],
            [0, 0, 50, 50, 0.05],
            [300, 150, 360, 380, 0.4],
        ],
        dtype=np.float32,
    )
    dets = to_detections(ref, raw, tf, DetectorConfig(score_thresh=0.1, min_box_px=8))
    assert [round(d.score, 2) for d in dets] == [0.9, 0.4]  # tiny and low-score dropped, sorted
    assert all(d.frame.identity == ref.identity and d.frame.transform == tf.tag for d in dets)
    x1, y1, x2, y2 = dets[0].bbox
    assert (x1, y1, x2, y2) == pytest.approx((400, 240, 640, 1040))
    assert all(0 <= d.bbox[0] < d.bbox[2] <= 2560 and 0 <= d.bbox[1] < d.bbox[3] <= 1440 for d in dets)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    m = iou_matrix(a, a)
    np.testing.assert_allclose(np.diag(m), 1)
    assert m[0, 1] == pytest.approx(25 / 175)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)

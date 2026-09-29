from pathlib import Path

import pytest
from pydantic import ValidationError

from scs.contracts import (
    COCO17_KEYPOINTS,
    Alert,
    BehaviorScore,
    CameraProfile,
    Event,
    EventType,
    FrameRef,
    Pose,
    SiteConfig,
    Track,
)

ROOT = Path(__file__).resolve().parents[1]
FRAME = FrameRef(camera_id="cam1", frame_idx=0, ts=1.0, width=640, height=360)


def test_all_camera_profiles_load():
    paths = sorted((ROOT / "configs" / "camera_profiles").glob("*.yaml"))
    assert paths, "expected at least one camera profile"
    for p in paths:
        prof = CameraProfile.from_yaml(p)
        assert prof.id == p.stem, f"{p.name}: id must match filename"


def test_example_site_loads_and_holds_no_urls():
    text = (ROOT / "configs" / "site.example.yaml").read_text()
    assert "rtsp://" not in text
    site = SiteConfig.from_yaml(ROOT / "configs" / "site.example.yaml")
    assert site.cameras[0].zones


def test_duplicate_camera_ids_rejected():
    cam = {"camera_id": "c", "profile_id": "p", "rtsp_main_env": "X"}
    with pytest.raises(ValidationError):
        SiteConfig(site_id="s", cameras=[cam, cam])


def test_env_var_name_enforced():
    with pytest.raises(ValidationError):
        SiteConfig(site_id="s", cameras=[{"camera_id": "c", "profile_id": "p",
                                          "rtsp_main_env": "rtsp://user:pw@1.2.3.4"}])


def test_bbox_validation():
    Track(frame=FRAME, track_id=1, bbox=(0, 0, 10, 10), score=0.9)
    with pytest.raises(ValidationError):
        Track(frame=FRAME, track_id=1, bbox=(10, 0, 5, 10), score=0.9)


def test_pose_requires_coco17():
    assert len(COCO17_KEYPOINTS) == 17
    Pose(frame=FRAME, track_id=1, keypoints=[(1.0, 2.0, 0.5)] * 17, model_id="m")
    with pytest.raises(ValidationError):
        Pose(frame=FRAME, track_id=1, keypoints=[(1.0, 2.0, 0.5)] * 33, model_id="m")


def test_behavior_window_order():
    with pytest.raises(ValidationError):
        BehaviorScore(camera_id="c", track_id=1, ts_start=2, ts_end=1, model_id="m", score=0.5)


def test_json_roundtrip():
    ev = Event(type=EventType.ITEM_PICKUP, camera_id="cam1", ts=1.0, source="test@0")
    assert Event.model_validate_json(ev.model_dump_json()) == ev
    al = Alert(site_id="s", ts_open=1.0, camera_ids=["cam1"], score=0.7, reason_codes=["x"])
    assert Alert.model_validate_json(al.model_dump_json()) == al


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Event(type=EventType.STORE_EXIT, camera_id="c", ts=1, source="s", surprise=1)

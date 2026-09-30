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
FRAME = FrameRef(camera_id="cam1", epoch=0, seq=0, frame_idx=0, ts=1.0, width=640, height=360)


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


def test_frame_ref_defaults_to_main_stream_with_optional_mono():
    ref = FrameRef(camera_id="c", epoch=0, seq=0, frame_idx=0, ts=1.0, width=2560, height=1440)
    assert ref.stream == "main"
    assert ref.ts_mono is None
    ref2 = FrameRef(camera_id="c", epoch=0, seq=0, frame_idx=0, ts=1.0, width=2560, height=1440, ts_mono=3.5)
    assert ref2.ts_mono == 3.5


def test_frame_identity_fields():
    ref = FrameRef(camera_id="c", epoch=3, seq=17, frame_idx=900, ts=10.0, width=2560, height=1440,
                   source_ts=9.8, transform="resize640:letterbox")
    assert ref.identity == ("c", 3, 17)
    assert ref.source_ts == 9.8 and ref.transform == "resize640:letterbox"
    base = {"camera_id": "c", "frame_idx": 0, "ts": 1.0, "width": 2, "height": 2}
    with pytest.raises(ValidationError):
        FrameRef(**base, seq=0)  # epoch required
    with pytest.raises(ValidationError):
        FrameRef(**base, epoch=0)  # seq required
    with pytest.raises(ValidationError):
        FrameRef(**base, epoch=-1, seq=0)


def test_camera_profile_verification_levels():
    base = {"id": "p", "vendor": "v", "model": "m", "hfov_deg": 90,
            "main_stream": {"width": 1920, "height": 1080, "fps": 15}}
    assert CameraProfile(**base).verification == "approximation"
    for level in ("spec_sourced", "measured", "emulator_calibrated"):
        with pytest.raises(ValidationError):
            CameraProfile(**base, verification=level)  # a claim above approximation needs sources
        assert CameraProfile(**base, verification=level, sources=["x"]).sources == ["x"]
    with pytest.raises(ValidationError):
        CameraProfile(**base, verification="verified")
    with pytest.raises(ValidationError):
        CameraProfile(**base, verified=True)  # old field is gone (extra="forbid")


def test_shipped_profiles_are_approximations():
    for p in sorted((ROOT / "configs" / "camera_profiles").glob("*.yaml")):
        assert CameraProfile.from_yaml(p).verification == "approximation", p.name

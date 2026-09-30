"""Shared data contracts for every pipeline stage.

This module is the ONLY coupling point between workstreams. Agents may add
optional fields; renaming/removing fields or changing semantics requires an
ADR in docs/adr/ and a bump of CONTRACTS_VERSION.

Conventions
- Timestamps: float seconds since Unix epoch (UTC), taken at frame decode.
- Pixel coordinates: absolute pixels in the frame the stage received, origin top-left.
- Zone polygons: NORMALIZED [0,1] coordinates so they survive resolution changes.
- Keypoints: COCO-17 order, (x, y, confidence) in pixels.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACTS_VERSION = "0.1.0"

COCO17_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Cameras & sites
# --------------------------------------------------------------------------

Codec = Literal["h264", "h265", "mjpeg"]


class StreamSpec(_Model):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    codec: Codec = "h264"
    bitrate_kbps: int | None = Field(default=None, gt=0)


class CameraProfile(_Model):
    """Optical/encoding characteristics of a camera MODEL (not an install)."""

    id: str
    vendor: str
    model: str
    main_stream: StreamSpec
    sub_stream: StreamSpec | None = None
    hfov_deg: float = Field(gt=0, le=200)
    lens: Literal["rectilinear", "fisheye"] = "rectilinear"
    # Brown-Conrady radial terms for rectilinear, equidistant k1..k4 for fisheye.
    distortion: list[float] = Field(default_factory=list)
    ir_night_mode: bool = False
    verified: bool = False  # True only once checked against datasheet or measured
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraProfile:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class ZoneType(StrEnum):
    SHELF = "shelf"
    HIGH_VALUE = "high_value"
    CHECKOUT = "checkout"
    ENTRY_EXIT = "entry_exit"
    STAFF_ONLY = "staff_only"
    AISLE = "aisle"


NormPoint = tuple[Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)]]


class Zone(_Model):
    id: str
    type: ZoneType
    polygon: list[NormPoint] = Field(min_length=3)


class CameraInstall(_Model):
    """One physical camera at a site. Secrets are referenced by env var name only."""

    camera_id: str
    profile_id: str
    rtsp_main_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    rtsp_sub_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    mount_height_m: float | None = Field(default=None, gt=0, lt=10)
    tilt_deg: float | None = Field(default=None, ge=0, le=90)  # 90 = straight down
    zones: list[Zone] = Field(default_factory=list)


class SiteConfig(_Model):
    site_id: str
    timezone: str = "America/New_York"
    cameras: list[CameraInstall]

    @model_validator(mode="after")
    def _unique_ids(self) -> SiteConfig:
        ids = [c.camera_id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("camera_id values must be unique within a site")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SiteConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


# --------------------------------------------------------------------------
# Perception outputs
# --------------------------------------------------------------------------

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


def _check_bbox(b: BBox) -> BBox:
    if not (b[2] > b[0] and b[3] > b[1]):
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1, got {b}")
    return b


class FrameRef(_Model):
    camera_id: str
    frame_idx: int = Field(ge=0)
    ts: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    stream: Literal["main", "sub"] = "sub"


class Detection(_Model):
    frame: FrameRef
    bbox: BBox
    score: float = Field(ge=0, le=1)
    cls: str = "person"

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v: BBox) -> BBox:
        return _check_bbox(v)


class Track(_Model):
    frame: FrameRef
    track_id: int = Field(ge=0)  # local to camera_id
    global_id: int | None = None  # set by cross-camera association, may stay None
    bbox: BBox
    score: float = Field(ge=0, le=1)
    state: Literal["tentative", "confirmed", "lost"] = "confirmed"

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v: BBox) -> BBox:
        return _check_bbox(v)


Keypoint = tuple[float, float, Annotated[float, Field(ge=0, le=1)]]


class Pose(_Model):
    frame: FrameRef
    track_id: int = Field(ge=0)
    keypoints: list[Keypoint] = Field(min_length=17, max_length=17)  # COCO17 order
    model_id: str


class BehaviorScore(_Model):
    """Output of any behavior model over a window of one track."""

    camera_id: str
    track_id: int
    global_id: int | None = None
    ts_start: float
    ts_end: float
    model_id: str  # e.g. "stgnf@2026-10-01-abc123"
    score: float = Field(ge=0, le=1)  # calibrated: higher = more suspicious
    label: str | None = None  # e.g. "conceal_pocket" for supervised models

    @model_validator(mode="after")
    def _window(self) -> BehaviorScore:
        if self.ts_end < self.ts_start:
            raise ValueError("ts_end must be >= ts_start")
        return self


# --------------------------------------------------------------------------
# Events & alerts
# --------------------------------------------------------------------------


class EventType(StrEnum):
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    SHELF_INTERACTION = "shelf_interaction"  # hand enters shelf polygon
    ITEM_PICKUP = "item_pickup"  # interaction that ends with object in hand
    CONCEAL_CANDIDATE = "conceal_candidate"  # behavior model above threshold
    CHECKOUT_VISIT = "checkout_visit"
    STORE_EXIT = "store_exit"
    CAMERA_HEALTH = "camera_health"


class Event(_Model):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: EventType
    camera_id: str
    ts: float
    track_id: int | None = None
    global_id: int | None = None
    zone_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str  # component + version that emitted it
    data: dict[str, Any] = Field(default_factory=dict)


class AlertStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class Alert(_Model):
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    site_id: str
    ts_open: float
    global_id: int | None = None
    camera_ids: list[str] = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    # machine-readable, e.g. ["pickup_hv", "conceal", "exit_no_checkout"]
    reason_codes: list[str] = Field(min_length=1)
    event_ids: list[str] = Field(default_factory=list)
    clip_uris: list[str] = Field(default_factory=list)
    status: AlertStatus = AlertStatus.PENDING_REVIEW
    verifier: dict[str, Any] | None = None  # optional VLM second opinion


# --------------------------------------------------------------------------
# Bus topology (Redis Streams). Keep names here so producers/consumers agree.
# --------------------------------------------------------------------------


class Streams:
    TRACKS = "scs:tracks"  # Track (high volume; trimmed aggressively)
    POSES = "scs:poses"  # Pose
    BEHAVIOR = "scs:behavior"  # BehaviorScore
    EVENTS = "scs:events"  # Event
    ALERTS = "scs:alerts"  # Alert
    HEALTH = "scs:health"  # Event(type=CAMERA_HEALTH)

"""Run configuration for the walking skeleton: one camera, one zone, one dwell rule.

Kept as a small JSON file in the work directory so every role process (ingest, clipper,
web) reads the same settings after a restart, without re-parsing CLI flags.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from scs.contracts import NormPoint, Zone, ZoneType

# Right half of the frame, full height: where people stand in the PoC demo clips.
DEFAULT_ZONE = Zone(
    id="zone_a", type=ZoneType.SHELF, polygon=[(0.45, 0.15), (1.0, 0.15), (1.0, 1.0), (0.45, 1.0)]
)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: str = "m1-demo"
    camera_id: str = "cam01"
    camera_profile: str = "unknown"  # configs/camera_profiles id when known
    # File path, or "env:NAME" for a live camera whose rtsp:// URL is in $NAME. URLs can
    # carry credentials, so they're never written to config.json (CameraInstall convention).
    source: str
    loop: bool = True  # file sources loop forever, like a camera
    speed: float = Field(default=1.0, gt=0)  # file pacing: 1 = real time, 0 < x: x times faster
    analytics_width: int = 160
    zone: Zone = DEFAULT_ZONE
    min_dwell_s: float = 5.0
    cooldown_s: float = 10.0  # per zone; persisted, so it survives restarts
    pre_roll_s: float = 15.0  # T10 spec; M1 acceptance needs >= 10
    post_roll_s: float = 10.0  # T10 spec; M1 acceptance needs >= 5
    segment_s: float = 2.0  # live evidence segment length
    host: str = "127.0.0.1"
    port: int = 8765
    label_dataset_id: str = "scs_review"  # labels go to <workdir>/labels/<id>/converted/labels/

    @property
    def is_live(self) -> bool:
        return self.source.startswith("env:")

    def resolved_source(self) -> str:
        if not self.is_live:
            return self.source
        name = self.source[4:]
        url = os.environ.get(name)
        if not url or not url.startswith(("rtsp://", "rtsps://")):
            raise SystemExit(f"${name} must hold an rtsp:// URL")
        return url

    @property
    def zone_polygon(self) -> list[NormPoint]:
        return list(self.zone.polygon)


def config_path(workdir: Path) -> Path:
    return workdir / "config.json"


def save(cfg: AppConfig, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = config_path(workdir).with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
    tmp.replace(config_path(workdir))


def load(workdir: Path) -> AppConfig:
    return AppConfig.model_validate_json(config_path(workdir).read_text())

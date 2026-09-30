"""The Protocols are importable and structurally checkable, with no implementations."""

from pathlib import Path

from scs.behavior.base import BehaviorModel
from scs.contracts import BehaviorScore
from scs.events.base import JourneyEngine
from scs.ingest.base import FrameSource
from scs.perception.base import Detector, PoseEstimator, Tracker
from scs.verify.base import Verifier

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["ingest", "perception", "behavior", "events", "evidence", "verify", "camera_emu", "runtime"]


def test_layout_packages_exist():
    for pkg in PACKAGES:
        assert (ROOT / "src" / "scs" / pkg / "__init__.py").is_file(), pkg


def test_structural_conformance():
    class Dummy:
        model_id = "dummy@0"
        window = 24

        def score(self, poses):
            return BehaviorScore(camera_id="c", track_id=0, ts_start=0, ts_end=0, model_id=self.model_id,
                                 score=0.0)

    assert isinstance(Dummy(), BehaviorModel)
    for proto in (FrameSource, Detector, Tracker, PoseEstimator, JourneyEngine, Verifier):
        assert not isinstance(Dummy(), proto)

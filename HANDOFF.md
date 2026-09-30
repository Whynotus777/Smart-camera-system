# HANDOFF from T03 (detection + tracking)

Changes T03 needs outside its owned paths. T03 did **not** make them (AGENTS.md rule 2).

## pyproject.toml (owner: T00 / T13)
T03's GPU path needs these on top of `.[dev,perception,torch]`. The tested set is pinned
in `docs/reports/T03-env.lock.txt` (sm_120, RTX 5090). Proposed extras:

```toml
detect = ["tensorrt-cu12>=10.8,<11", "onnx>=1.17", "onnxslim>=0.1", "transformers>=4.56",
          "motmetrics>=1.4", "scipy>=1.13"]
# AGPL-3.0: R&D only (docs/DATA.md). Keep out of any customer build.
detect-rnd = ["ultralytics>=8.3"]
```
The CPU CI path needs none of them: trackers fall back to a numpy Hungarian when scipy
is missing, and detector modules import torch/TensorRT lazily.

## T02 (ingest): image convention for `FrameSource` → `Detector`
`Detector.detect` assumes frames are uint8 **HxWx3 RGB** (numpy or torch, any device) and
`FrameRef.width/height` are main-stream size even when a smaller image was decoded.
`DetectorConfig.channel_order="bgr"` exists for OpenCV-decoded input. Please confirm or
state the convention in `ingest/base.py` (see PR "Blockers").

## T00 (contracts/interfaces): `Detector` Protocol frame type
`FrameSource.frames()` yields `np.ndarray | torch.Tensor`, but `Detector.detect` in
`perception/base.py` types frames as `Tensor`. T03's detectors accept both (`Any`), which is
structurally compatible. Aligning the Protocol text needs an ADR-lite change by the owner.

## T09 (eval): readers and the `smartspaces_track` suite
`src/scs/perception/track_data.py` reads SmartSpaces `ground_truth.txt` and MEVA KPF
`*.geom.yml` directly, mirroring T09's splits (PR #9: by scene / by site). It exists only because
the canonical converters weren't there yet. When T09's converters land, point
`track_eval.py` at them and delete `track_data.py`; the splits are listed at the top of it.
MEVA only labels activity participants: precision/mAP/IDF1 there are lower bounds.

## T12 (runtime): wiring
`MicroBatcher(detector, on_result, max_batch=16, deadline_s=0.020)` is the only object the
runtime needs. Run one per GPU process, `submit()` from each camera's decode loop, and hand
`on_result` output to a per-camera `Tracker` (one instance per camera).
`CrossCameraAssociator` is off by default.

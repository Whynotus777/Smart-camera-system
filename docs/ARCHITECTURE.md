# Architecture

Status: target design for the rebuild. Owner of changes: whoever holds the ADR.
Data types referenced below live in `src/scs/contracts.py`.

## 1. Product framing drives the design

- 7-Eleven policy is that clerks don't confront shoplifters. The value is **evidence
  clips, shrink analytics, and deterrence**, not real-time apprehension. That means
  **precision matters far more than latency**; a 5–20 s alert delay is fine.
- A false accusation is a legal and brand event. So every theft alert goes to a
  **human review queue** first (T10). The model's job is to make that queue short
  and rich, not to be perfect.
- Operational analytics (dwell, queue length, empty-shelf, spills) reuse the same
  tracks/zones and are lower-risk upsells. Keep the event engine generic enough.
- The system reports **observable interactions worth reviewing** (item to bag, item
  concealed under clothing, obscured interaction, exit without checkout). It never
  asserts "theft"; that's the reviewer's and the store's call. UI copy, alert reason
  codes, and labels follow this.

## 1b. Two tracks, one product

| Track | Purpose | Success = |
|---|---|---|
| **Release** (T02, T10, T12, T13) | Cameras → persisted events → playable clips → review → recorded outcome, surviving restarts and failures, installable on the store box. | Milestone M1/M2 in ROADMAP pass on target hardware. |
| **Model & data** (T03–T09, T11) | Better detection/tracking/pose/behavior; sim and emulation. | Improvement on the held-out **real** test set at the same false-alert budget. |

The model track plugs into the release track through the interfaces below. It never
blocks it: M1 runs with a trivial rule or an injected test event.

## 2. Pipeline

```
 RTSP main stream (per camera)
   │  T02 ingest: GStreamer/PyAV + NVDEC, reconnect, timestamps, health events
   ├──► ENCODED packet tap ──► evidence ring buffer (T10, last 60 s). Never subject to
   │                           analytics frame-dropping.
   ▼
 Frame (GPU, analytics path; may drop oldest under load)
   │  GPU resize → detector input (≈640 px)
   ▼
 T03 Detector (person; later: hand/item)  →  Detection
   ▼
 T03 Tracker (per camera)                 →  Track
   │  optional cross-camera association (appearance, in-memory, TTL)
   ▼
 Zone gate: only tracks in/near SHELF, HIGH_VALUE, CHECKOUT, ENTRY_EXIT get pose
   ▼
 T04 Pose on HIGH-RES crops from main stream  →  Pose (COCO17)
   ▼
 T06 Behavior models over pose windows (24–48 frames) → BehaviorScore
   ▼
 T05 Journey state machine per global/local track  →  Event, Alert(pending_review)
   ▼
 T10 Evidence: clip export from ring buffer, review queue API/UI
   ▼
 T11 VLM verifier (optional second opinion on the clip)  → Alert.verifier
   ▼
 Reviewer confirms → store notification / daily report
```

Transport between stages: in-process for the hot path (frames never leave the GPU
process), **Redis Streams** (`contracts.Streams`) for Tracks/Poses/Events/Alerts so
that consumers can be restarted and recordings can be replayed.

## 3. Key decisions (and why)

| # | Decision | Why | Revisit when |
|---|---|---|---|
| D1 | Decode **main stream only**, resize on GPU for detection; all boxes/keypoints are main-stream pixels (ADR 0002) | Sub-stream (640×360) puts a hand at ~6 px at 5 m; main (2560×1440) ≈ 25 px. One decode avoids syncing two streams with different timestamps. | NVDEC budget exceeded on edge target (T12 measures). |
| D2 | **Journey logic, not gesture alarms** | Single-frame "concealing/looking around" rules fire on phones and chairs (see legacy demos). Theft = sequence across zones; exit-without-checkout is the strongest cheap signal. | — |
| D3 | **Pose is a baseline signal, not the only one.** T06 compares pose-only, visual (person/hand-region crops over time), and fused (visual + pose + track + zone). | Pose alone can't tell "phone out of own pocket" from "product into jacket"; the object and hand region carry that. Pose is still cheap, privacy-friendly, and has public data (PoseLift). | Decided by T06's comparison on the real held-out set. |
| D4 | Zone-gated pose | Pose is the most expensive per-person stage; most people in a c-store at a given moment aren't at a shelf. | — |
| D5 | Human review before store | Precision can't be guaranteed pre-deployment; reviewer labels become training data. | Measured precision after review sustained > target for 60 days. |
| D6 | Models behind interfaces (`Detector`, `Tracker`, `PoseEstimator`, `BehaviorModel`, `Verifier`) | Several best-in-class options are AGPL / NC. We must be able to swap to permissive ones before customer deployment. | — |
| D7 | Redis Streams over pub/sub | Pub/sub drops messages when a consumer is down; Streams give persistence, consumer groups, replay. | Multi-site fleet → consider MQTT to cloud. |
| D8 | Python orchestration + TensorRT engines; DeepStream is optional | Faster for agents to build and test; batched TRT handles 10 cams on a 5090. DeepStream port is a CTO-phase optimization if the edge box needs it. | T12 shows edge target can't hit budget. |
| D9 | **Micro-batching with a deadline**, not batch = number of cameras | A slow or dead camera must never delay healthy ones. Batch whatever frames are ready within ~20 ms, up to a max size. | T12 perf data. |
| D10 | **Traceable frame identity**: `camera_id + epoch (connection id) + seq`, plus source (RTP) timestamp and the preprocessing transform | Proves a detector box and a high-res crop came from the same image; makes latency numbers honest (capture vs decode time). | — |
| D11 | Crop retention for model development | So T06 can train visual/fused models later without re-ingesting: on gated tracks, optionally persist hand/person crops (lab and sim only by default; store only with consent). | Privacy review before store. |

## 4. Interfaces (each implemented by one task)

```python
class FrameSource(Protocol):          # T02  (rtsp | file | sim | emulated)
    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray | torch.Tensor]]: ...

class Detector(Protocol):             # T03
    def detect(self, batch: list[tuple[FrameRef, Tensor]]) -> list[list[Detection]]: ...

class Tracker(Protocol):              # T03 (one instance per camera)
    def update(self, dets: list[Detection], frame: Tensor | None) -> list[Track]: ...

class PoseEstimator(Protocol):        # T04
    def estimate(self, frame: FrameRef, image: Tensor, tracks: list[Track]) -> list[Pose]: ...

class BehaviorModel(Protocol):        # T06
    model_id: str
    window: int                       # frames
    def score(self, poses: Sequence[Pose]) -> BehaviorScore: ...

class JourneyEngine(Protocol):        # T05
    def on_track(self, t: Track) -> list[Event]: ...
    def on_pose(self, p: Pose) -> list[Event]: ...
    def on_behavior(self, b: BehaviorScore) -> list[Event]: ...
    def poll_alerts(self, now: float) -> list[Alert]: ...

class Verifier(Protocol):             # T11
    def verify(self, alert: Alert, clip_paths: list[Path]) -> dict: ...
```

Task owners put the Protocol in `src/scs/<area>/base.py` in their first PR, exactly
as above unless an ADR changes it.

## 5. Compute budget (targets to validate in T12)

Assumptions: 10 cameras, 15 fps decode, detection at 10 fps, ≤ 3 gated people per
camera at a time, pose at 10 fps on gated tracks.

| Stage | Load | Target on RTX 5090 | Notes |
|---|---|---|---|
| Decode | 10 × 1440p15 H.264/H.265 | NVDEC, < 15% | Measure H.265 separately; Reolink may default to it. |
| Detection | 100 img/s @ 640 | < 10 ms per batch of 10 | TRT FP16. |
| Pose | ≤ 300 crops/s | batched, < 15 ms per batch | Top-down on 256×192 crops. |
| Behavior | ≤ 30 windows/s | negligible | Small GCN / flow model. |
| End-to-end alert latency | — | p95 < 5 s from event to `Alert` | Clip export adds ~20 s (post-roll). |

The edge target (Jetson Orin NX/AGX vs. small x86 + RTX 4000-class) is chosen in T12
from these measurements. The legacy Jetson **Nano** is out: end-of-life, Python 3.6,
unsupported by current toolchains.

## 6. Legacy PoC disposition

| File | Keep? | Notes |
|---|---|---|
| `legacy/camera_system_with_lstm.py` | Reference only | Imports 4 modules not in repo; LSTM weights never existed. Zone JSON format and reconnect loop are worth porting. |
| `legacy/deepsort_poc.py` | Reference only | Source of the demo videos (now `tests/fixtures/video/`); item-overlap rule generates false positives. |
| `legacy/dispatcher_agent.py`, `legacy/simulation_trigger.py` | Park | Robot dispatch is out of scope for the pilot. The sim stub's intent moves to T08 (synthetic training data). |
| `legacy/alerts/notifier.py`, `legacy/stream/multi_cam_stream.py` | Replace | T10 / T02. |

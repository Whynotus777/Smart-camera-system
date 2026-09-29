# Legacy PoC (Sept 2025): reference only

This directory is the original lab proof of concept, quarantined here so agents don't
mistake it for the product. **Don't extend it, and don't import from it.** Port ideas
into `src/scs/` through the task briefs instead. `docs/ARCHITECTURE.md` §6 says what
is worth keeping from each file.

| File | Was | Disposition |
|---|---|---|
| `camera_system_with_lstm.py` | Multi-camera YOLO + ByteTrack + pose/LSTM gesture alerts | Reference. Port the zone JSON format and the reconnect loop. **Does not run**: see below. |
| `deepsort_poc.py` | YOLO + DeepSORT single camera; produced the demo videos | Reference. The item-overlap rule causes false positives. |
| `testcamera.py` | RTSP smoke test | Superseded by T02 `scripts/rtsp_probe.py`. |
| `dispatcher_agent.py`, `simulation_trigger.py` | Redis robot-dispatch demo | Parked: out of scope for the pilot. |
| `alerts/notifier.py` | SMTP alert stub | Replaced by T10 notification adapters. |
| `stream/multi_cam_stream.py` | Threaded OpenCV capture | Replaced by T02. |
| `requirements.txt` | PoC dependencies | Includes AGPL packages (ultralytics, boxmot). Don't copy into `pyproject.toml`. |

## Missing modules

`camera_system_with_lstm.py` imports four modules that were never committed:
`pose_action_detector`, `lstm_action_classifier`, `video_recorder` and
`byte_tracker_fixed`. The LSTM weights it expects never existed either. Until the
originals are added here (ROADMAP H0b), the file compiles but can't run, and its
zone and recorder logic can only be ported from what's visible in this file. Don't
recreate these modules from guesses.

## Running it

Camera URLs come from environment variables (`SCS_CAM1_RTSP_URL`; see `.env.example`
at the repo root). The code expects to run from this directory. Nothing in
`src/scs/` depends on it.

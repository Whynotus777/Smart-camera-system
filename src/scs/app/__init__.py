"""Walking skeleton (milestone M1): video → Event → evidence clip → review → label.

T13 owns this package. It wires the thinnest possible stand-ins together behind the
ARCHITECTURE §4 interfaces so the whole product path runs and survives crashes before
the real components exist. Each stand-in is meant to be deleted when its owner lands:

    FfmpegSource            → T02 FileSource / RtspSource        (scs.ingest.base.FrameSource)
    BackgroundDiffDetector  → T03 detector                       (scs.perception.base.Detector)
    IouTracker              → T03 tracker                        (scs.perception.base.Tracker)
    DwellEngine             → T05 journey engine                 (scs.events.base.JourneyEngine)
    FileLoopEvidence /
    SegmentEvidence         → T10 packet ring buffer + clip export (scs.app.evidence.EvidenceStore)
    web.py (stdlib HTTP)    → T10 review queue service

SQLite (`store.py`) is the system of record. Durability rules every role follows:
- IDs are deterministic (derived from camera + media position), so re-processing after a
  crash re-derives the same rows and `INSERT OR IGNORE` drops them: no duplicates.
- Everything a frame produces (events, alerts, clip jobs, checkpoint, engine state) is
  committed in one transaction, so a kill -9 leaves either all of it or none of it.
- Files are written to a temp name, fsynced, then renamed; the DB row that points at a
  file is updated only after the rename.
"""

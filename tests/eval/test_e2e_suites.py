"""End-to-end: synthetic video -> streaming pipeline -> suite -> report, with hand-known answers.

A tiny MEVA-shaped dataset is built in a temp data root: 3 clips (val / test / test) of
20 s at 10 fps, with labeled interactions. Reference pipelines give exact expectations:
GT replay -> recall 1.0 and 0 false detections; periodic alerts every 5 s -> 3 alerts
per 20-s clip (t=5, 10, 15) -> FA/h = 3 / (20/3600) = 540 per camera-hour.
"""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pyarrow")

from eval.canonical import ClipLabels, LabelEvent, LabelSupport, converted_dir, write_labels  # noqa: E402
from eval.e2e import ProtocolPipeline, VideoFileSource, alert_preds, interaction_preds, run_clip  # noqa: E402
from eval.metrics.stats import MetricValue  # noqa: E402
from eval.report import compare  # noqa: E402
from eval.run import main as run_main  # noqa: E402
from eval.splits import ClipMeta, Splits, SplitSpec, generate  # noqa: E402
from eval.suites.base import RunContext, SuiteResult, load_model  # noqa: E402
from eval.suites.meva_fa import MevaFA  # noqa: E402
from eval.suites.meva_interaction import MevaInteraction  # noqa: E402

FPS, SECONDS = 10, 20


def _video(path, n=FPS * SECONDS):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(n):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, i % 64] = 255
        w.write(img)
    w.release()


@pytest.fixture
def meva_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SCS_DATA_ROOT", str(tmp_path))
    (tmp_path / "meva").mkdir()
    (tmp_path / "meva/MANIFEST.json").write_text(json.dumps({"use": "prod", "files": {}}))
    clips = {"v1": ("school.G423", "val"), "t1": ("bus.G331", "test"), "t2": ("school.G421", "test")}
    for cid, (cam, _) in clips.items():
        vid = tmp_path / "meva/raw" / f"{cid}.avi"
        vid.parent.mkdir(parents=True, exist_ok=True)
        _video(vid)
        events = [
            LabelEvent(type="item_pickup", t_start=2.0, t_end=3.0, track_id=1),
            LabelEvent(type="item_put_down", t_start=12.0, t_end=12.5, track_id=1),
            LabelEvent(type="item_pickup", t_start=6.0, t_end=7.0, extra={"src_status": "not_good"}),
        ]
        write_labels(
            converted_dir("meva") / "labels" / f"{cid}.json",
            ClipLabels(
                clip_id=cid,
                fps=FPS,
                label_source="human",
                events=events,
                dataset="meva",
                camera_id=cam,
                site_id=cam.split(".")[0],
                duration_s=SECONDS,
                continuous=True,
                video=f"meva/raw/{cid}.avi",
                supports=LabelSupport(events=True, exhaustive=True),
            ),
        )
    sd = tmp_path / "splits"
    sd.mkdir()
    metas = [ClipMeta(c, camera_id=cam, site_id=cam.split(".")[0]) for c, (cam, _) in clips.items()]
    Splits(
        generate(
            "meva",
            metas,
            SplitSpec(
                holdout={
                    "test": {"sites": ["bus"], "cameras": ["school.G421"]},
                    "val": {"cameras": ["school.G423"]},
                }
            ),
        )
    ).save(sd)
    return tmp_path, sd


def _ctx(root, sd, pipeline, **kw):
    return RunContext(
        pipeline=pipeline,
        data_root=root,
        split_dir=sd,
        bootstrap=100,
        out_dir=None,
        log=lambda *_: None,
        **kw,
    )


def test_video_source_media_time_and_causality(tmp_path):
    p = tmp_path / "x.avi"
    _video(p, 25)
    src = VideoFileSource(p, "cam")
    refs = [r for r, _ in src.frames()]
    assert [r.frame_idx for r in refs] == list(range(25)) and src.fps == pytest.approx(FPS)
    assert all(b.ts >= a.ts for a, b in zip(refs, refs[1:], strict=False))


def test_gt_replay_gives_perfect_recall(meva_root):
    root, sd = meva_root
    res = MevaInteraction().run(_ctx(root, sd, "eval.models.reference_pipelines:GTReplayPipeline"))
    assert res.status == "ok"
    # 2 positives per test clip (not_good is ignore) -> 4; all detected at 0 false/h
    assert res.headline["recall_at_budget"].value == 1.0
    assert res.headline["recall_at_budget"].n == {"events": 4, "units": 2}
    assert res.headline["fa_per_hour"].value == 0.0
    assert res.metrics["threshold_source"].startswith("val")
    lat = res.metrics["latency_event_end_to_emit"]["p50"].value
    assert 0 <= lat <= 0.1 + 1e-9  # emitted on the first frame at/after t_end


def test_null_pipeline_zero_recall(meva_root):
    root, sd = meva_root
    res = MevaInteraction().run(_ctx(root, sd, "eval.models.reference_pipelines:NullPipeline"))
    assert res.headline["recall_at_budget"].value == 0.0


def test_periodic_alerts_exact_fa_rate(meva_root):
    root, sd = meva_root
    ctx = _ctx(
        root,
        sd,
        "eval.models.reference_pipelines:PeriodicAlertPipeline",
        policy={"period_s": 5.0},
        options={"min_hours": "0"},
    )
    res = MevaFA().run(ctx)
    # lineage unknown -> test clips only: 2 clips x 3 alerts over 2 x 20 s
    fa = res.headline["false_alerts_per_camera_hour"]
    assert fa.value == pytest.approx(6 / (40 / 3600))
    assert res.metrics["alerts"]["false"] + res.metrics["alerts"]["duplicates"] == 6
    assert res.params["pool_splits"] == ["test"]


def test_crash_is_reported_not_hidden(tmp_path):
    p = tmp_path / "x.avi"
    _video(p, 10)

    class Boom:
        def __init__(self, **_):
            pass

        def process(self, ref, image):
            if ref.frame_idx == 5:
                raise RuntimeError("kaboom")
            return []

        def flush(self, now):
            return []

    o = run_clip("c", "cam", p, FPS, Boom, {})
    assert o.error and "kaboom" in o.error and o.n_frames == 5


def test_protocol_pipeline_composes_interfaces(tmp_path):
    from scs.contracts import Alert, Detection, Event, EventType, Pose, Track

    p = tmp_path / "x.avi"
    _video(p, 30)

    class Det:
        def detect(self, batch):
            return [[Detection(frame=r, bbox=(1, 1, 20, 40), score=0.9)] for r, _ in batch]

    class Trk:
        def update(self, dets, frame):
            return [Track(frame=d.frame, track_id=7, bbox=d.bbox, score=d.score) for d in dets]

    class PoseEst:
        def estimate(self, frame, image, tracks):
            return [
                Pose(frame=frame, track_id=t.track_id, keypoints=[(1.0, 1.0, 0.5)] * 17, model_id="p")
                for t in tracks
            ]

    beh = load_model("behavior", "eval.models.dummy:ConstantBehavior").obj

    class Journey:
        def __init__(self):
            self.n_beh = 0

        def on_track(self, t):
            return []

        def on_pose(self, p):
            return []

        def on_behavior(self, b):
            self.n_beh += 1
            return [Event(type=EventType.CONCEAL_CANDIDATE, camera_id=b.camera_id, ts=b.ts_end, source="t")]

        def poll_alerts(self, now):
            return (
                [Alert(site_id="s", ts_open=now, camera_ids=["cam"], score=0.7, reason_codes=["x"])]
                if self.n_beh == 1
                else []
            )

    j = Journey()

    def factory(**_):
        return ProtocolPipeline(Det(), Trk(), j, PoseEst(), beh, detect_every=1)

    o = run_clip("c", "cam", p, FPS, factory, {})
    assert o.error is None
    # window 24 -> first behavior score on frame index 23 -> 7 behavior events (frames 23..29)
    assert sum(1 for it in o.items if it.kind == "event") == 7
    alerts = alert_preds(o)
    assert len(alerts) == 1 and alerts[0].t_end == pytest.approx(2.3)
    assert interaction_preds(o) == []  # conceal_candidate isn't an interaction detection


def test_runner_writes_report_and_compare(meva_root, tmp_path, capsys):
    root, sd = meva_root
    out = tmp_path / "runs"
    rc = run_main(
        [
            "--suite",
            "meva_interaction",
            "--pipeline",
            "eval.models.reference_pipelines:GTReplayPipeline",
            "--bootstrap",
            "50",
            "--out",
            str(out),
            "--split-dir",
            str(sd),
            "--compare",
            "main",
        ]
    )
    assert rc == 0
    [d] = [p for p in out.iterdir() if p.is_dir()]
    doc = json.loads((d / "meva_interaction.json").read_text())
    assert doc["result"]["headline"]["recall_at_budget"]["value"] == 1.0
    assert doc["env"]["git_sha"] and "gpu" in doc["env"] and doc["result"]["datasets"][0]["split_digest"]
    md = (d / "summary.md").read_text()
    assert "proxy, not retail" in md and "recall_at_budget (gated)" in md
    assert doc["compare"]["found"] is False  # no baseline report for main in this temp dir


def test_runner_rejects_predictions_for_e2e_suites(capsys):
    with pytest.raises(SystemExit):
        run_main(["--suite", "meva_fa", "--predictions", "unused-dir"])


def test_fa_rate_forbidden_on_non_continuous_suite():
    with pytest.raises(ValueError, match="continuous"):
        SuiteResult("x", "ok", {"fa_per_hour": MetricValue(1.0)}, continuous=False)


def test_compare_two_point_rule():
    def doc(v, ci=None):
        return {
            "env": {"git_sha": "b"},
            "result": {
                "gated": ["recall_at_budget"],
                "headline": {
                    "recall_at_budget": {"value": v, "status": "ok", "ci95": ci},
                    "false_alerts_per_camera_hour": {"value": v, "status": "ok", "ci95": ci},
                },
            },
        }

    c = compare(doc(0.70, [0.6, 0.8]), doc(0.73), "main")
    rows = {r["metric"]: r for r in c["rows"]}
    assert rows["recall_at_budget"]["verdict"] == "REGRESSION" and c["blocking"] == ["recall_at_budget"]
    assert compare(doc(0.715), doc(0.73), "main")["blocking"] == []
    # rate: worse only if the whole CI is above the baseline
    assert rows["false_alerts_per_camera_hour"]["verdict"] == "ok"
    c2 = compare(doc(0.9, [0.85, 0.95]), doc(0.7), "main")
    assert {r["metric"]: r["verdict"] for r in c2["rows"]}["false_alerts_per_camera_hour"] == "REGRESSION"

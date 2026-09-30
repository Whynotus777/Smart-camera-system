import inspect
import os
import uuid

import pytest
from pydantic import BaseModel

import scs.contracts as C
from scs.bus import DEFAULT_MAXLEN, InMemoryBus, RedisBus, decode, encode

FRAME = C.FrameRef(camera_id="cam1", frame_idx=3, ts=1.5, width=2560, height=1440, stream="sub",
                   ts_mono=812.25)
PROFILE = C.CameraProfile(
    id="p", vendor="v", model="m",
    main_stream=C.StreamSpec(width=2560, height=1440, fps=15, codec="h265", bitrate_kbps=6144),
    sub_stream=C.StreamSpec(width=640, height=360, fps=20), hfov_deg=87, distortion=[-0.3, 0.1],
)
ZONE = C.Zone(id="z", type=C.ZoneType.SHELF, polygon=[(0, 0), (1, 0), (0.5, 1)])
INSTALL = C.CameraInstall(camera_id="cam1", profile_id="p", rtsp_main_env="SCS_CAM1_RTSP_MAIN",
                          mount_height_m=2.7, tilt_deg=35, zones=[ZONE])

# One non-trivial sample per contract model. test_every_contract_model_has_a_sample
# fails if a model is added to contracts.py without a sample here.
SAMPLES: dict[type[BaseModel], BaseModel] = {
    C.StreamSpec: PROFILE.main_stream,
    C.CameraProfile: PROFILE,
    C.Zone: ZONE,
    C.CameraInstall: INSTALL,
    C.SiteConfig: C.SiteConfig(site_id="s", cameras=[INSTALL]),
    C.FrameRef: FRAME,
    C.Detection: C.Detection(frame=FRAME, bbox=(1, 2, 30, 40), score=0.8),
    C.Track: C.Track(frame=FRAME, track_id=7, global_id=3, bbox=(1, 2, 30, 40), score=0.8, state="lost"),
    C.Pose: C.Pose(frame=FRAME, track_id=7, keypoints=[(1.5, 2.5, 0.9)] * 17, model_id="m@1"),
    C.BehaviorScore: C.BehaviorScore(camera_id="cam1", track_id=7, ts_start=1, ts_end=2.4,
                                     model_id="m@1", score=0.3, label="conceal_pocket"),
    C.Event: C.Event(type=C.EventType.SHELF_INTERACTION, camera_id="cam1", ts=1.5, track_id=7,
                     zone_id="z", confidence=0.6, source="t@0", data={"wrist": "right", "n": 4}),
    C.Alert: C.Alert(site_id="s", ts_open=2.0, camera_ids=["cam1", "cam2"], score=0.7,
                     reason_codes=["pickup_hv", "exit_no_checkout"], event_ids=["e1"],
                     clip_uris=["file:///tmp/c.mp4"], status=C.AlertStatus.CONFIRMED,
                     verifier={"concealment_observed": True, "confidence": 0.8}),
}


def _contract_models() -> set[type[BaseModel]]:
    return {obj for _, obj in inspect.getmembers(C, inspect.isclass)
            if issubclass(obj, C._Model) and obj is not C._Model}


def test_every_contract_model_has_a_sample():
    assert _contract_models() == set(SAMPLES)


@pytest.mark.parametrize("model_cls", list(SAMPLES), ids=lambda c: c.__name__)
def test_roundtrip_in_memory_bus(model_cls):
    bus = InMemoryBus()
    sent = SAMPLES[model_cls]
    entry_id = bus.publish("t:roundtrip", sent)
    [(got_id, got)] = bus.consume("t:roundtrip", "g", "c1", model_cls)
    assert got_id == entry_id
    assert type(got) is model_cls
    assert got == sent


def test_encode_decode_accepts_bytes():
    ev = SAMPLES[C.Event]
    fields = {k.encode(): v.encode() for k, v in encode(ev).items()}
    assert decode(fields, C.Event) == ev
    assert encode(ev)["v"] == C.CONTRACTS_VERSION


def test_group_semantics_and_ack():
    bus = InMemoryBus()
    ev = SAMPLES[C.Event]
    ids = [bus.publish(C.Streams.EVENTS, ev) for _ in range(5)]
    first = bus.consume(C.Streams.EVENTS, "g", "c1", C.Event, count=3)
    rest = bus.consume(C.Streams.EVENTS, "g", "c2", C.Event, count=10)
    assert [i for i, _ in first] + [i for i, _ in rest] == ids  # each entry once per group
    assert bus.consume(C.Streams.EVENTS, "g", "c1", C.Event) == []
    assert len(bus.consume(C.Streams.EVENTS, "other", "c1", C.Event)) == 5  # groups independent
    assert bus.pending(C.Streams.EVENTS, "g") == 5
    assert bus.ack(C.Streams.EVENTS, "g", *ids[:2]) == 2
    assert bus.ack(C.Streams.EVENTS, "g", ids[0]) == 0
    assert bus.pending(C.Streams.EVENTS, "g") == 3


def test_maxlen_trimming():
    bus = InMemoryBus(maxlen={C.Streams.TRACKS: 3})
    tr = SAMPLES[C.Track]
    ids = [bus.publish(C.Streams.TRACKS, tr) for _ in range(5)]
    assert bus.xlen(C.Streams.TRACKS) == 3
    assert [i for i, _ in bus.consume(C.Streams.TRACKS, "g", "c", C.Track)] == ids[2:]


def test_every_stream_has_a_maxlen():
    names = {v for k, v in vars(C.Streams).items() if not k.startswith("_")}
    assert names == set(DEFAULT_MAXLEN)


class _RecordingRedis:
    """Records calls RedisBus makes, to check the wire contract without a server."""

    def __init__(self):
        self.calls = []

    def xadd(self, stream, fields, maxlen=None, approximate=False):
        self.calls.append(("xadd", stream, fields, maxlen, approximate))
        return b"1-0"

    def xgroup_create(self, stream, group, id="$", mkstream=False):
        self.calls.append(("xgroup_create", stream, group, id, mkstream))

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        self.calls.append(("xreadgroup", group, consumer, streams, count, block))
        return [[b"s", [(b"1-0", {b"v": b"0.1.0", b"json": SAMPLES[C.Event].model_dump_json().encode()})]]]

    def xack(self, stream, group, *ids):
        return len(ids)


def test_redis_bus_uses_approximate_maxlen_per_stream():
    r = _RecordingRedis()
    bus = RedisBus(r, maxlen={C.Streams.TRACKS: 42})
    assert bus.publish(C.Streams.TRACKS, SAMPLES[C.Track]) == "1-0"
    assert bus.publish(C.Streams.ALERTS, SAMPLES[C.Alert]) == "1-0"
    assert r.calls[0][3:] == (42, True)
    assert r.calls[1][3:] == (DEFAULT_MAXLEN[C.Streams.ALERTS], True)


def test_redis_bus_creates_group_once_and_decodes():
    r = _RecordingRedis()
    bus = RedisBus(r)
    for _ in range(2):
        [(entry_id, ev)] = bus.consume("s", "g", "c", C.Event, count=5, block_ms=10)
        assert entry_id == "1-0" and ev == SAMPLES[C.Event]
    assert [c[0] for c in r.calls] == ["xgroup_create", "xreadgroup", "xreadgroup"]
    assert r.calls[0][3:] == ("0", True)


def _live_redis():
    url = os.environ.get("SCS_REDIS_URL", "redis://localhost:6379/15")
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
        client.ping()
    except Exception:
        return None
    return client


@pytest.mark.parametrize("model_cls", list(SAMPLES), ids=lambda c: c.__name__)
def test_roundtrip_live_redis(model_cls):
    client = _live_redis()
    if client is None:
        pytest.skip("no Redis reachable at SCS_REDIS_URL / localhost:6379")
    stream = f"scs:test:{uuid.uuid4().hex}"
    try:
        bus = RedisBus(client)
        bus.publish(stream, SAMPLES[model_cls])
        [(entry_id, got)] = bus.consume(stream, "g", "c", model_cls, block_ms=100)
        assert got == SAMPLES[model_cls]
        assert bus.ack(stream, "g", entry_id) == 1
    finally:
        client.delete(stream)

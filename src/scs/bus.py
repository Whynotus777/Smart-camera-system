"""Thin Redis Streams helper for passing contract models between processes.

Tracks, poses, behavior scores, events and alerts move between stages over Redis
Streams (ARCHITECTURE D7) so consumers can restart without losing messages and
recordings can be replayed. This module is the only place that knows the wire
format, so producers and consumers can't drift apart:

- each entry has two fields: `v` (CONTRACTS_VERSION of the producer) and `json`
  (`model.model_dump_json()`);
- every `publish` trims its stream with `MAXLEN ~ n` (per-stream limits below);
- `consume` reads through a consumer group, creating it on first use, and
  returns `(entry_id, model)` pairs; callers `ack` once they've handled them
  (at-least-once delivery).

`InMemoryBus` has the same interface and semantics without a server, for tests
and single-process replay.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from scs.contracts import CONTRACTS_VERSION, Streams

M = TypeVar("M", bound=BaseModel)

# Approximate per-stream retention. Placeholders sized for ~10 cameras; T12 tunes them.
DEFAULT_MAXLEN: dict[str, int] = {
    Streams.TRACKS: 100_000,  # ~5 min at 10 cams x 3 people x 10 fps
    Streams.POSES: 100_000,
    Streams.BEHAVIOR: 100_000,
    Streams.EVENTS: 100_000,
    Streams.ALERTS: 10_000,
    Streams.HEALTH: 10_000,
}
FALLBACK_MAXLEN = 10_000


def encode(model: BaseModel) -> dict[str, str]:
    """Contract model → stream entry fields."""
    return {"v": CONTRACTS_VERSION, "json": model.model_dump_json()}


def decode(fields: Mapping[Any, Any], model_cls: type[M]) -> M:
    """Stream entry fields → contract model. Accepts str or bytes keys/values."""
    norm = {_s(k): _s(v) for k, v in fields.items()}
    return model_cls.model_validate_json(norm["json"])


def _s(x: str | bytes) -> str:
    return x.decode() if isinstance(x, bytes) else x


class Bus(Protocol):
    def publish(self, stream: str, model: BaseModel) -> str: ...

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        model_cls: type[M],
        count: int = 100,
        block_ms: int | None = None,
    ) -> list[tuple[str, M]]: ...

    def ack(self, stream: str, group: str, *entry_ids: str) -> int: ...


class RedisBus:
    """`Bus` backed by a real Redis (>= 7) client."""

    def __init__(self, client: Any, maxlen: Mapping[str, int] | None = None) -> None:
        self._r = client
        self._maxlen = {**DEFAULT_MAXLEN, **(maxlen or {})}
        self._groups: set[tuple[str, str]] = set()

    @classmethod
    def from_url(cls, url: str, maxlen: Mapping[str, int] | None = None) -> RedisBus:
        import redis

        return cls(redis.Redis.from_url(url), maxlen)

    def publish(self, stream: str, model: BaseModel) -> str:
        n = self._maxlen.get(stream, FALLBACK_MAXLEN)
        return _s(self._r.xadd(stream, encode(model), maxlen=n, approximate=True))

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        model_cls: type[M],
        count: int = 100,
        block_ms: int | None = None,
    ) -> list[tuple[str, M]]:
        self._ensure_group(stream, group)
        resp = self._r.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
        out: list[tuple[str, M]] = []
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                out.append((_s(entry_id), decode(fields, model_cls)))
        return out

    def ack(self, stream: str, group: str, *entry_ids: str) -> int:
        return int(self._r.xack(stream, group, *entry_ids)) if entry_ids else 0

    def _ensure_group(self, stream: str, group: str) -> None:
        if (stream, group) in self._groups:
            return
        import redis

        try:
            # id="0": a new group sees whatever history is still retained (replay).
            self._r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._groups.add((stream, group))


@dataclass
class _Group:
    next_idx: int = 0  # index into the stream's global sequence
    pending: set[str] = field(default_factory=set)


class InMemoryBus:
    """In-process fake of `RedisBus`: same wire format, groups, acks and trimming.

    Trimming is exact rather than approximate. Blocking is not emulated;
    `block_ms` is accepted and ignored.
    """

    def __init__(self, maxlen: Mapping[str, int] | None = None) -> None:
        self._maxlen = {**DEFAULT_MAXLEN, **(maxlen or {})}
        self._streams: dict[str, list[tuple[int, str, dict[str, str]]]] = {}
        self._groups: dict[tuple[str, str], _Group] = {}
        self._seq = itertools.count(1)

    def publish(self, stream: str, model: BaseModel) -> str:
        seq = next(self._seq)
        entry_id = f"{seq}-0"
        entries = self._streams.setdefault(stream, [])
        entries.append((seq, entry_id, encode(model)))
        n = self._maxlen.get(stream, FALLBACK_MAXLEN)
        if len(entries) > n:
            del entries[: len(entries) - n]
        return entry_id

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        model_cls: type[M],
        count: int = 100,
        block_ms: int | None = None,
    ) -> list[tuple[str, M]]:
        g = self._groups.setdefault((stream, group), _Group())
        out: list[tuple[str, M]] = []
        for seq, entry_id, fields in self._streams.get(stream, []):
            if len(out) >= count:
                break
            if seq <= g.next_idx:
                continue
            out.append((entry_id, decode(fields, model_cls)))
            g.pending.add(entry_id)
            g.next_idx = seq
        return out

    def ack(self, stream: str, group: str, *entry_ids: str) -> int:
        g = self._groups.get((stream, group))
        if g is None:
            return 0
        acked = g.pending.intersection(entry_ids)
        g.pending -= acked
        return len(acked)

    def xlen(self, stream: str) -> int:
        return len(self._streams.get(stream, []))

    def pending(self, stream: str, group: str) -> int:
        g = self._groups.get((stream, group))
        return len(g.pending) if g else 0

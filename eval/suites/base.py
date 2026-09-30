"""Suite base class, registry, run context and result type.

A suite = datasets + split + metric recipe. Suites are discovered by importing every
module in `eval/suites/`, so T06 (`public_pose.py`), T07 (`emu_matrix.py`) and T12
(`perf*.py`, `soak*.py`) add a file and nothing else. Framework defaults (e.g. the
`public_pose` reference implementation in `eval.pose_bench`) are registered with
`builtin=True` and are replaced by a suite file of the same name.

Result invariants (checked in `SuiteResult.__post_init__`): every headline metric is a
`MetricValue` (value + CI + counts, or explicit "unavailable"), and a suite that
declares `continuous=False` can't report false alerts per hour.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from eval.metrics.stats import DEFAULT_B, MetricValue

FA_KEYS = ("fa_per_hour", "false_alerts_per_camera_hour")


@dataclass
class ModelSpec:
    role: str  # e.g. "behavior", "pipeline", or a sim_transfer variant name
    spec: str  # "module:attr" (+ optional JSON kwargs after '?')
    obj: Any
    model_id: str
    lineage: dict[str, Any] | None  # {"train": [{"dataset": ..., "split": ...}], "license": ...}

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "spec": self.spec, "model_id": self.model_id, "lineage": self.lineage}


def resolve_attr(spec: str) -> Any:
    """'pkg.mod:attr' -> the attribute itself, never called (pipeline/source factories)."""
    mod, _, attr = spec.partition(":")
    if not attr or "?" in spec:
        raise ValueError(f"factory spec must be module:attr, got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


def load_object(spec: str) -> Any:
    """'pkg.mod:Name' or 'pkg.mod:Name?{"seed": 1}' -> object. Classes are instantiated (with the
    JSON kwargs); other callables are called only when kwargs are given; anything else is returned."""
    target, _, kw = spec.partition("?")
    mod, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"model spec must be module:attr, got {spec!r}")
    obj = getattr(importlib.import_module(mod), attr)
    kwargs = json.loads(kw) if kw else {}
    if isinstance(obj, type) or kwargs:
        return obj(**kwargs)
    return obj


def load_model(role: str, spec: str) -> ModelSpec:
    obj = load_object(spec)
    mid = getattr(obj, "model_id", None) or spec
    return ModelSpec(role, spec, obj, str(mid), getattr(obj, "lineage", None))


@dataclass
class RunContext:
    models: dict[str, ModelSpec] = field(default_factory=dict)
    policy: dict[str, Any] | None = None
    policy_path: str | None = None
    pipeline: str | None = None  # e2e: "module:factory" building a streaming pipeline per camera
    source: str | None = (
        None  # e2e: FrameSource factory "module:factory" (default: T02 FileSource if present)
    )
    predictions: Path | None = None  # non-e2e suites only: saved canonical predictions
    split: str = "test"
    bootstrap: int = DEFAULT_B
    limit: int | None = None  # max clips (smoke runs; the report says so)
    options: dict[str, str] = field(default_factory=dict)  # --opt key=value, suite-specific
    data_root: Path | None = None
    split_dir: Path | None = None
    out_dir: Path | None = None
    workers: int = 1
    log: Callable[[str], None] = print

    def opt(self, key: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
        return cast(self.options[key]) if key in self.options else default


@dataclass
class SuiteResult:
    suite: str
    status: str  # ok | partial | unavailable
    headline: dict[str, MetricValue]
    metrics: dict[str, Any] = field(default_factory=dict)  # nested dicts of MetricValue / plain values
    curves: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # "proxy, not retail", "synthetic", ...
    notes: list[str] = field(default_factory=list)
    gated: list[str] = field(default_factory=list)  # headline keys the 2-pt regression rule applies to
    params: dict[str, Any] = field(default_factory=dict)
    continuous: bool = False

    def __post_init__(self) -> None:
        for k, v in self.headline.items():
            if not isinstance(v, MetricValue):
                raise TypeError(f"headline {k!r} must be a MetricValue, got {type(v).__name__}")
            if not self.continuous and k in FA_KEYS and v.status == "ok":
                raise ValueError(f"{self.suite}: false alerts/hour only from continuous-footage suites")

    @classmethod
    def unavailable(cls, suite: str, reason: str, **kw: Any) -> SuiteResult:
        return cls(suite, "unavailable", {}, notes=[reason], **kw)

    def to_json(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, MetricValue):
                return x.to_json()
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            if isinstance(x, list | tuple):
                return [conv(v) for v in x]
            return x

        return {
            "suite": self.suite,
            "status": self.status,
            "labels": self.labels,
            "headline": conv(self.headline),
            "gated": self.gated,
            "metrics": conv(self.metrics),
            "curves": self.curves,
            "datasets": self.datasets,
            "params": self.params,
            "notes": self.notes,
            "continuous": self.continuous,
        }


class Suite:
    name: ClassVar[str]
    description: ClassVar[str] = ""
    datasets: ClassVar[tuple[str, ...]] = ()
    labels: ClassVar[tuple[str, ...]] = ()  # always printed in the report header
    continuous: ClassVar[bool] = False  # may report false alerts per hour
    end_to_end: ClassVar[bool] = False  # must drive the streaming pipeline from video
    gated: ClassVar[tuple[str, ...]] = ()

    def run(self, ctx: RunContext) -> SuiteResult:  # pragma: no cover - interface
        raise NotImplementedError

    def result(self, status: str, headline: dict[str, MetricValue], **kw: Any) -> SuiteResult:
        return SuiteResult(
            self.name,
            status,
            headline,
            labels=list(self.labels),
            gated=list(self.gated),
            continuous=self.continuous,
            **kw,
        )

    def unavailable(self, reason: str, **kw: Any) -> SuiteResult:
        return SuiteResult.unavailable(
            self.name, reason, labels=list(self.labels), continuous=self.continuous, **kw
        )


SUITES: dict[str, type[Suite]] = {}
_BUILTIN: set[str] = set()


def register_suite(cls: type[Suite] | None = None, *, builtin: bool = False):
    def deco(c: type[Suite]) -> type[Suite]:
        name = c.name
        if name in SUITES and name not in _BUILTIN and not builtin:
            if SUITES[name] is not c and SUITES[name].__qualname__ != c.__qualname__:
                raise ValueError(
                    f"suite {name!r} registered twice ({SUITES[name].__module__}, {c.__module__})"
                )
        if builtin and name in SUITES and name not in _BUILTIN:
            return c  # a suite file already overrides this framework default
        SUITES[name] = c
        if builtin:
            _BUILTIN.add(name)
        else:
            _BUILTIN.discard(name)
        return c

    return deco(cls) if cls is not None else deco


def discover() -> dict[str, type[Suite]]:
    import pkgutil

    import eval.pose_bench  # noqa: F401  framework default public_pose
    import eval.suites as pkg

    for m in pkgutil.iter_modules([str(Path(pkg.__file__).parent)]):
        if not m.name.startswith("_") and m.name != "base":
            importlib.import_module(f"eval.suites.{m.name}")
    return SUITES

"""CLI: python -m eval.run --suite <name> [--models role=module:attr ...] [--policy p.yaml] ...

Writes runs/eval/<git-sha>/<suite>.json + summary.md; `--compare main` adds a delta table
against runs/eval/<sha of main>/<suite>.json. GPU pipelines must be launched through the
GPU lock: `scripts/gpu shared -- python -m eval.run ...` (benchmarks: `exclusive`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from eval.report import compare, environment, policy_digest, write_result
from eval.suites.base import RunContext, discover, load_model

REPO = Path(__file__).resolve().parents[1]


def _kv(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"--{what} expects key=value, got {it!r}")
        out[k] = v
    return out


def _resolve(ref: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, cwd=REPO, check=False)  # noqa: S603,S607
    return r.stdout.strip() if r.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suite", action="append", help="suite name (repeatable)")
    ap.add_argument("--list", action="store_true", help="list registered suites")
    ap.add_argument("--models", nargs="*", default=[], help="role=module:attr[?{json kwargs}]")
    ap.add_argument("--policy", help="YAML/JSON policy file passed to the pipeline (recorded by hash)")
    ap.add_argument("--pipeline", help="e2e: streaming pipeline factory module:attr")
    ap.add_argument("--source", help="e2e: FrameSource factory module:attr (default: fallback decoder)")
    ap.add_argument("--predictions", type=Path, help="non-e2e suites: dir of canonical predictions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="max clips per split (smoke run; marked partial)")
    ap.add_argument("--opt", nargs="*", default=[], help="suite options key=value")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "eval")
    ap.add_argument("--split-dir", type=Path, help="frozen split files (default eval/splits/)")
    ap.add_argument("--compare", help="git ref (e.g. main) or sha to compare against")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 3 on a blocking regression")
    a = ap.parse_args(argv)

    suites = discover()
    if a.list:
        for name, cls in sorted(suites.items()):
            tags = [t for t, on in (("e2e", cls.end_to_end), ("continuous", cls.continuous)) if on]
            print(f"{name:20s} {cls.description} {tags or ''} {list(cls.labels) or ''}")
        return 0
    if not a.suite:
        ap.error("--suite is required (or --list)")
    unknown = [s for s in a.suite if s not in suites]
    if unknown:
        ap.error(f"unknown suite(s) {unknown}; registered: {sorted(suites)}")
    policy = None
    if a.policy:
        policy = yaml.safe_load(Path(a.policy).read_text())
    models = {r: load_model(r, s) for r, s in _kv(a.models, "models").items()}
    env = environment()
    rc = 0
    for name in a.suite:
        cls = suites[name]
        if cls.end_to_end and a.predictions:
            ap.error(
                f"{name} is end-to-end: it must drive the streaming pipeline (--pipeline), "
                "not saved predictions"
            )
        ctx = RunContext(
            models=models,
            policy=policy,
            policy_path=a.policy,
            pipeline=a.pipeline,
            source=a.source,
            predictions=a.predictions,
            split=a.split,
            bootstrap=a.bootstrap,
            limit=a.limit,
            options=_kv(a.opt, "opt"),
            out_dir=a.out / env["git_sha"],
            workers=a.workers,
            split_dir=a.split_dir,
            log=lambda m: print(m, file=sys.stderr),
        )
        t0 = time.perf_counter()
        res = cls().run(ctx)
        doc = {
            "env": env,
            "result": res.to_json(),
            "models": [m.to_json() for m in models.values()],
            "policy_path": a.policy,
            "policy_digest": policy_digest(policy),
            "runtime_s": round(time.perf_counter() - t0, 3),
            "argv": sys.argv if argv is None else ["eval.run", *argv],
        }
        if a.compare:
            base_sha = _resolve(a.compare) or a.compare
            p = a.out / base_sha / f"{name}.json"
            base = json.loads(p.read_text()) if p.exists() else None
            doc["compare"] = compare(doc, base, a.compare)
            if a.fail_on_regression and doc["compare"].get("blocking"):
                rc = 3
        path = write_result(a.out, doc)
        h = ", ".join(f"{k}={v.get('value')}" for k, v in doc["result"]["headline"].items())
        print(f"{name}: {res.status} {h} -> {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

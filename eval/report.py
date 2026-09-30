"""Report writing: `runs/eval/<sha>/<suite>.json`, `summary.md`, and `--compare` delta tables.

Every report records what docs/EVAL.md requires: git sha (+dirty), model ids and
lineage, dataset versions (converter + source manifest digest + split digest), policy
hash, camera profiles, GPU + driver (queried read-only from nvidia-smi), and runtime.

Regression rule (AGENTS.md rule 7): a gated metric in [0, 1] that drops by more than
2 points (0.02) blocks merge. Rates (false alerts/hour, switches/minute) have no
"points"; they're flagged REGRESSION when the new value's CI lies entirely above the
baseline value (a significant increase). Both rules are printed in the table.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scs.contracts import CONTRACTS_VERSION

LOWER_IS_BETTER = ("fa_per_hour", "false_alerts", "idsw", "latency", "p95")
POINTS = 0.02


def environment() -> dict[str, Any]:
    from eval.suites._common import git_sha

    env: dict[str, Any] = {
        "git_sha": git_sha(),
        "contracts_version": CONTRACTS_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run(  # noqa: S603 - fixed argv, resolved binary
            [
                smi,
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        env["gpu"] = (
            [line.strip() for line in r.stdout.splitlines()] if r.returncode == 0 else "nvidia-smi failed"
        )
    else:
        env["gpu"] = None
    return env


def policy_digest(policy: dict | None) -> str | None:
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16] if policy else None


def write_result(out_root: Path, doc: dict[str, Any]) -> Path:
    d = out_root / doc["env"]["git_sha"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{doc['result']['suite']}.json"
    p.write_text(json.dumps(doc, indent=1, default=str))
    (d / "summary.md").write_text(summary_md(d))
    return p


def _fmt(m: dict[str, Any] | None) -> str:
    if not m:
        return "—"
    if m.get("status") != "ok" or m.get("value") is None:
        return f"unavailable ({m.get('reason', '')})".replace(" ()", "")
    s = f"{m['value']:.3f}"
    if m.get("ci95") and None not in m["ci95"]:
        s += f" [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
    if m.get("n"):
        s += " (" + ", ".join(f"{k}={v}" for k, v in m["n"].items()) + ")"
    if m.get("low_n"):
        s += " **LOW-N**"
    return s


def summary_md(d: Path) -> str:
    docs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    if not docs:
        return ""
    env = docs[0]["env"]
    lines = [
        f"# Eval summary — `{env['git_sha']}`",
        "",
        f"contracts {env['contracts_version']} · python {env['python']} · GPU {env.get('gpu')} · "
        f"{env['time_utc']}",
        "",
    ]
    for doc in docs:
        r = doc["result"]
        lines += [f"## `{r['suite']}` — {r['status']}", ""]
        for lab in r.get("labels", []):
            lines.append(f"> **{lab}**")
        if r.get("labels"):
            lines.append("")
        lines += ["| metric | value [95% CI] (n) |", "|---|---|"]
        for k, v in r["headline"].items():
            lines.append(f"| {k}{' (gated)' if k in r.get('gated', []) else ''} | {_fmt(v)} |")
        lines.append("")
        models = ", ".join(f"{m['role']}=`{m['model_id']}`" for m in doc.get("models", [])) or "none"
        lines.append(
            f"models: {models} · policy: {doc.get('policy_digest') or 'none'} · "
            f"runtime {doc.get('runtime_s', 0):.1f} s"
        )
        for ds in r.get("datasets", []):
            lines.append(
                f"data: `{ds.get('dataset')}` converter {ds.get('converter')} split "
                f"v{ds.get('split_version')} ({ds.get('split_digest')}) source {ds.get('source')}"
            )
        for n in r.get("notes", []):
            lines.append(f"- {n}")
        if doc.get("compare"):
            lines += ["", compare_md(doc["compare"])]
        lines.append("")
    return "\n".join(lines)


def _higher_better(k: str) -> bool:
    return not any(t in k for t in LOWER_IS_BETTER)


def compare(new: dict[str, Any], base: dict[str, Any] | None, base_ref: str) -> dict[str, Any]:
    if base is None:
        return {
            "base": base_ref,
            "found": False,
            "hint": f"no baseline report for {base_ref}: run the same command at that commit "
            f"(git worktree add ../scs-main {base_ref})",
        }
    rows = []
    for k, v in new["result"]["headline"].items():
        b = base["result"]["headline"].get(k)
        row: dict[str, Any] = {"metric": k, "gated": k in new["result"].get("gated", []), "new": v, "base": b}
        if v.get("value") is None or not b or b.get("value") is None:
            row["delta"], row["verdict"] = None, "n/a"
        else:
            delta = v["value"] - b["value"]
            row["delta"] = delta
            bounded = 0 <= v["value"] <= 1 and 0 <= b["value"] <= 1 and "per_" not in k and "fa" not in k
            if bounded:
                worse = -delta if _higher_better(k) else delta
                row["verdict"] = "REGRESSION" if worse > POINTS else ("better" if worse < 0 else "same")
            else:
                ci = v.get("ci95") or [None, None]
                if _higher_better(k):
                    sig = ci[1] is not None and ci[1] < b["value"]
                else:
                    sig = ci[0] is not None and ci[0] > b["value"]
                row["verdict"] = (
                    "REGRESSION"
                    if sig
                    else ("worse (n.s.)" if (delta < 0) == _higher_better(k) and delta != 0 else "ok")
                )
        rows.append(row)
    blocking = [r["metric"] for r in rows if r["gated"] and r["verdict"] == "REGRESSION"]
    return {
        "base": base_ref,
        "base_sha": base["env"]["git_sha"],
        "found": True,
        "rows": rows,
        "blocking": blocking,
    }


def compare_md(c: dict[str, Any]) -> str:
    if not c.get("found"):
        return f"**compare vs {c['base']}:** {c['hint']}"
    out = [
        f"**compare vs {c['base']} (`{c['base_sha']}`)** — rule: gated [0,1] metric drop > 2 pts, or rate "
        f"CI entirely worse than baseline = REGRESSION",
        "",
        "| metric | base | new | delta | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in c["rows"]:
        d = "—" if r["delta"] is None else f"{r['delta']:+.3f}"
        out.append(
            f"| {r['metric']}{' (gated)' if r['gated'] else ''} | {_fmt(r['base'])} | {_fmt(r['new'])} | "
            f"{d} | {r['verdict']} |"
        )
    if c["blocking"]:
        out.append(f"\n**BLOCKING regressions: {', '.join(c['blocking'])}**")
    return "\n".join(out)

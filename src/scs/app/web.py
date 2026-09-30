"""Minimal review page + API (stand-in for T10's review queue service).

Standard library only (`http.server`), so the M1 flow and its CI test need no new
dependency. T10 replaces it with its FastAPI service; the endpoints below are the
contract `tests/integration/` drives, so T10 should keep them (or update the tests
in the same PR):

    GET  /                               HTML review queue (clips play inline)
    GET  /api/alerts                     [{alert, clip, review}]
    POST /api/alerts/<id>/review         {"decision": "confirmed"|"dismissed", "reason", "reviewer",
                                          "request_id"} → 200 {review, created} | 409 conflict
    GET  /clips/<alert_id>.mp4           clip, with HTTP Range support (browsers seek with it)
    GET  /healthz

A 200 on POST means the outcome is committed to SQLite (synchronous=FULL) and exported
as a label. Retrying the same decision is safe (idempotent); a different decision for
an already-reviewed alert is a 409.

Binds 127.0.0.1 by default: clips show people, and access control is M3 work.
"""

from __future__ import annotations

import html
import json
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scs.app import config as cfgmod
from scs.app.labels import export_labels
from scs.app.roles import clip_dir, db_path, file_info, log, role_lock
from scs.app.store import DECISIONS, ReviewConflict, Store

REVIEW_RE = re.compile(r"^/api/alerts/([0-9a-f]{32})/review$")
CLIP_RE = re.compile(r"^/clips/([0-9a-f]{32})\.mp4$")


class App:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.cfg = cfgmod.load(workdir)
        self.store = Store(db_path(workdir))
        self.lock = threading.Lock()  # one sqlite connection, serialized across request threads
        self.fps = None if self.cfg.is_live else file_info(workdir, self.cfg.source).fps

    def export(self, only: str | None = None) -> list[Path]:
        return export_labels(
            self.store,
            self.workdir / "labels",
            self.cfg.label_dataset_id,
            self.cfg.camera_profile,
            self.fps,
            only,
        )

    def alerts_json(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.store.alerts()
        return [
            {
                "alert": json.loads(a.model_dump_json()),
                "clip": None
                if j is None
                else {
                    "status": j.status,
                    "clip_start": j.clip_start,
                    "clip_end": j.clip_end,
                    "event_ts": a.ts_open,
                },
                "review": None if r is None else r.__dict__,
            }
            for a, j, r in rows
        ]


def _page(items: list[dict[str, Any]]) -> str:
    cards = []
    for it in items:
        a, clip, rev = it["alert"], it["clip"], it["review"]
        aid = a["alert_id"]
        pre = post = ""
        if clip and clip["status"] == "done":
            pre = f"{clip['event_ts'] - clip['clip_start']:.1f}"
            post = f"{clip['clip_end'] - clip['event_ts']:.1f}"
            media = f'<video controls preload="metadata" width="480" src="/clips/{aid}.mp4"></video>'
        else:
            media = f"<p>clip {html.escape(clip['status'] if clip else 'missing')}…</p>"
        if rev:
            action = f"<p><b>{html.escape(rev['decision'])}</b> {html.escape(rev.get('reason') or '')}</p>"
        else:
            action = (
                f'<p><input id="r-{aid}" placeholder="reason code (optional)"> '
                f"<button onclick=\"review('{aid}','confirmed')\">Confirm</button> "
                f"<button onclick=\"review('{aid}','dismissed')\">Dismiss</button></p>"
            )
        title = f"{html.escape(', '.join(a['reason_codes']))} · {html.escape(a['camera_ids'][0])}"
        cards.append(
            f"<section><h3>{title}</h3><p><code>{aid}</code> · pre-roll {pre}s · post-roll {post}s"
            f" · status {html.escape(a['status'])}</p>"
            f"{media}{action}</section>"
        )
    body = "\n".join(cards) or "<p>No alerts yet. This page refreshes itself.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SCS review queue</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:15px system-ui,sans-serif;margin:16px;max-width:760px}}section{{border:1px solid #ccc;
border-radius:8px;padding:8px 12px;margin:12px 0}}video{{max-width:100%}}</style></head><body>
<h1>Review queue</h1><p>Observable interactions worth reviewing. Nothing here asserts theft.</p>{body}
<script>
async function review(id, decision) {{
  const body = JSON.stringify({{decision, reason: document.getElementById('r-'+id).value || null,
                               reviewer: 'web', request_id: crypto.randomUUID()}});
  for (let i = 0; i < 20; i++) {{  // retry until the server confirms the outcome is stored
    try {{ const r = await fetch('/api/alerts/'+id+'/review', {{method:'POST', body,
             headers:{{'content-type':'application/json'}}}});
           if (r.ok || r.status === 409) return location.reload(); }} catch (e) {{}}
    await new Promise(res => setTimeout(res, 1000));
  }}
  alert('Could not record the review; try again.');
}}
if (!document.querySelector('input')) setTimeout(() => location.reload(), 5000);
</script></body></html>"""


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "scs-review/m1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            if self.path == "/api/alerts":
                return self._json(200, app.alerts_json())
            if self.path in ("/", "/index.html"):
                return self._send(200, _page(app.alerts_json()).encode(), "text/html; charset=utf-8")
            if m := CLIP_RE.match(self.path):
                return self._clip(m.group(1))
            self._json(404, {"error": "not found"})

        def _clip(self, alert_id: str) -> None:
            p = clip_dir(app.workdir) / f"{alert_id}.mp4"
            if not p.exists():
                return self._json(404, {"error": "clip not ready"})
            data = p.read_bytes()
            rng = self.headers.get("Range")
            if rng and (m := re.match(r"bytes=(\d*)-(\d*)$", rng)):
                start = int(m.group(1)) if m.group(1) else len(data) - int(m.group(2))
                end = int(m.group(2)) if m.group(1) and m.group(2) else len(data) - 1
                if start >= len(data) or start > end:
                    return self._send(416, b"", "video/mp4", {"Content-Range": f"bytes */{len(data)}"})
                end = min(end, len(data) - 1)
                return self._send(
                    206,
                    data[start : end + 1],
                    "video/mp4",
                    {"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"},
                )
            self._send(200, data, "video/mp4", {"Accept-Ranges": "bytes"})

        def do_POST(self) -> None:  # noqa: N802
            m = REVIEW_RE.match(self.path)
            if not m:
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                decision = body["decision"]
                if decision not in DECISIONS:
                    raise ValueError(decision)
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._json(400, {"error": f"body must be JSON with decision in {DECISIONS}"})
            try:
                with app.lock:
                    rev, created = app.store.record_review(
                        m.group(1),
                        decision,
                        str(body.get("request_id") or uuid.uuid4()),
                        reason=body.get("reason"),
                        reviewer=body.get("reviewer"),
                    )
                    app.export(only=m.group(1))
            except KeyError:
                return self._json(404, {"error": "unknown alert"})
            except ReviewConflict as e:
                return self._json(HTTPStatus.CONFLICT, {"error": str(e)})
            self._json(200, {"review": rev.__dict__, "created": created})

    return Handler


def run_web(workdir: Path) -> None:
    with role_lock(workdir, "web"):
        app = App(workdir)
        with app.lock:
            app.export()  # repair: a crash between a committed review and its export
        srv = ThreadingHTTPServer((app.cfg.host, app.cfg.port), make_handler(app))
        srv.daemon_threads = True
        log("web", f"review queue on http://{app.cfg.host}:{app.cfg.port}/")
        srv.serve_forever()

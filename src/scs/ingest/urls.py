"""Camera URLs come from environment variables and are never logged in full (docs/SECURITY.md §1).

Anything that formats a URL for a log line, exception, health event or report goes
through `redact_url`. GStreamer and FFmpeg error strings can echo the URL back, so
`redact_text` scrubs free text too.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit, urlunsplit

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
# Reolink and others also accept credentials as query parameters.
_QUERY_SECRET = re.compile(r"(?i)\b(user(?:name)?|pass(?:word)?|pwd|token|auth)=[^&\s]*")


class UrlEnvError(RuntimeError):
    pass


def url_from_env(name: str) -> str:
    """Read a camera URL from env var `name` (UPPER_SNAKE only, like CameraInstall)."""
    if not _ENV_NAME.match(name):
        raise UrlEnvError(f"not an env var name: {name!r}")
    val = os.environ.get(name, "").strip()
    if not val:
        raise UrlEnvError(f"env var {name} is not set")
    return val


def redact_url(url: str) -> str:
    """`rtsp://<user>:<pw>@host:554/path?password=x` → `rtsp://***@host:554/path?password=***`."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return redact_text(url)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]
    query = _QUERY_SECRET.sub(lambda m: f"{m.group(1)}=***", parts.query)
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def redact_text(text: str) -> str:
    """Scrub userinfo and credential query params from arbitrary text (error messages)."""
    text = _USERINFO.sub(lambda m: m.group("scheme") + "***@", text)
    return _QUERY_SECRET.sub(lambda m: f"{m.group(1)}=***", text)

"""A private RTSP test farm: mediamtx in Docker plus supervised ffmpeg publishers.

Stand-in for T14's MEVA replay farm (same topology and faults) on a free localhost port,
so tests never touch the shared farm on :8554.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

IMAGE = "bluenviron/mediamtx:1.21.1"
# TCP-only RTSP; every other listener off (MoQ would otherwise bind :8892 on all interfaces).
MTX_CONFIG = "logLevel: warn\nreadTimeout: 30s\napi: no\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\n" \
             "moq: no\nrtspTransports: [tcp]\nrtspAddress: {addr}\npaths:\n  all_others:\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_cmd(*cmd: str, timeout: float = 60) -> subprocess.CompletedProcess:
    """Run a command; a wedged Docker daemon fails the test instead of hanging it."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603


class Publisher:
    """ffmpeg looping a clip into mediamtx; restarted whenever it exits (unless paused)."""

    def __init__(self, clip: Path, url: str) -> None:
        self.cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i",
                    str(clip), "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", url]
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._t = threading.Thread(target=self._supervise, daemon=True)

    def start(self) -> Publisher:
        self._t.start()
        return self

    def _supervise(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set() and (self.proc is None or self.proc.poll() is not None):
                self.proc = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL,  # noqa: S603
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._stop.wait(0.5)

    def drop(self, seconds: float) -> None:
        """Kill the publisher and keep it down for `seconds` (the path goes offline)."""
        self._paused.set()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(5)
        time.sleep(seconds)
        self._paused.clear()

    def signal(self, sig: int) -> None:
        assert self.proc is not None
        os.kill(self.proc.pid, sig)

    def stop(self) -> None:
        self._stop.set()
        self._t.join(2)
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)
            self.proc.terminate()
            self.proc.wait(5)

class Mediamtx:
    """A private mediamtx on a free localhost port (never T14's farm port 8554).

    Uses the release binary when `SCS_MEDIAMTX_BIN` (or `mediamtx` on PATH) exists, else the
    Docker image. Same kill/restart semantics either way: `kill` = server gone, readers and
    publishers disconnected; `restart` = same port back.
    """

    def __init__(self, port: int, cfg: Path, docker: str | None = None, name: str = "",
                 binary: str | None = None) -> None:
        self.port, self.cfg, self.docker, self.name, self.binary = port, cfg, docker, name, binary
        self.proc: subprocess.Popen | None = None

    @classmethod
    def start_or_skip(cls, tmp: Path) -> Mediamtx:
        port = _free_port()
        cfg = tmp / "mediamtx.yml"
        binary = os.environ.get("SCS_MEDIAMTX_BIN") or shutil.which("mediamtx")
        if binary:
            cfg.write_text(MTX_CONFIG.format(addr=f"127.0.0.1:{port}"))
            m = cls(port, cfg, binary=binary)
            m.restart()
            return m
        docker = shutil.which("docker")
        if not docker or run_cmd(docker, "image", "inspect", IMAGE).returncode != 0:
            pytest.skip(f"no mediamtx binary (SCS_MEDIAMTX_BIN) and no docker image {IMAGE}")
        cfg.write_text(MTX_CONFIG.format(addr=":8554"))
        name = f"scs-t02-test-{os.getpid()}-{port}"
        r = run_cmd(docker, "run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:8554", "-v",
                    f"{cfg}:/mediamtx.yml:ro", IMAGE)
        assert r.returncode == 0, r.stderr
        time.sleep(1.0)
        return cls(port, cfg, docker=docker, name=name)

    def url(self, path: str) -> str:
        return f"rtsp://127.0.0.1:{self.port}/{path}"

    def kill(self) -> None:
        if self.binary:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(10)
        else:
            run_cmd(self.docker or "docker", "kill", self.name)

    def restart(self) -> None:
        if self.binary:
            self.proc = subprocess.Popen([self.binary, str(self.cfg)], stdin=subprocess.DEVNULL,  # noqa: S603
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:  # wait until the RTSP port accepts
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", self.port)) == 0:
                        return
                time.sleep(0.05)
            raise RuntimeError("mediamtx did not start")
        run_cmd(self.docker or "docker", "start", self.name)

    def remove(self) -> None:
        if self.binary:
            self.kill()
        else:
            run_cmd(self.docker or "docker", "rm", "-f", self.name)

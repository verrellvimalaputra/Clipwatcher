"""Records the stream with streamlink and keeps a wall-clock -> byte-offset index.

Why the index: chat timestamps are wall-clock time, but a video file has its own
timeline, and Twitch ad breaks get filtered out by streamlink, which leaves gaps.
So "spike at 21:43:10, go back 30 s" can't be a fixed offset into the file.
Instead we pipe streamlink to stdout, write the bytes ourselves, and note when
each chunk arrived. To cut a clip we copy the bytes between two wall-clock
times. Ad gaps simply contain no bytes, so the alignment never drifts.
"""
from __future__ import annotations

import bisect
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger("recorder")
TS_PACKET = 188


@dataclass
class Session:
    path: Path
    times: list[float] = field(default_factory=list)    # arrival time of each chunk
    offsets: list[int] = field(default_factory=list)    # file offset where that chunk starts
    size: int = 0
    ended: bool = False

    def add(self, t: float, nbytes: int):
        # one index entry per ~0.25 s is plenty
        if not self.times or t - self.times[-1] >= 0.25:
            self.times.append(t)
            self.offsets.append(self.size)
        self.size += nbytes

    def covers(self, t: float) -> bool:
        return bool(self.times) and self.times[0] - 5 <= t and (not self.ended or t <= self.times[-1] + 5)

    def offset_at(self, t: float) -> int:
        i = bisect.bisect_right(self.times, t) - 1
        off = self.offsets[max(i, 0)] if t >= self.times[0] else 0
        if t > self.times[-1]:
            off = self.size
        return off - off % TS_PACKET  # stay on a TS packet boundary

    def extract(self, t0: float, t1: float, dest: Path) -> int:
        a, b = self.offset_at(t0), self.offset_at(t1)
        with open(self.path, "rb") as src, open(dest, "wb") as out:
            src.seek(a)
            remaining = b - a
            while remaining > 0:
                chunk = src.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        return b - a


class Recorder(threading.Thread):
    def __init__(self, channel: str, outdir: Path, quality: str, stop: threading.Event, url: str | None = None,
                 extra_args: list[str] | None = None):
        super().__init__(daemon=True, name="recorder")
        self.channel, self.outdir, self.quality, self.stop = channel, outdir, quality, stop
        self.url = url or f"https://twitch.tv/{channel}"  # streamlink handles Twitch and YouTube Live
        self.extra_args = extra_args or []
        self.sessions: list[Session] = []
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None

    def shutdown(self):
        self.stop.set()
        if self.proc:
            self.proc.kill()

    @property
    def live_session(self) -> Session | None:
        with self.lock:
            s = self.sessions[-1] if self.sessions else None
        return s if s and not s.ended else None

    def session_for(self, t: float) -> Session | None:
        with self.lock:
            for s in reversed(self.sessions):
                if s.covers(t):
                    return s
        return None

    def run(self):
        while not self.stop.is_set():
            cmd = ["streamlink", "--retry-streams", "30", "--retry-open", "3",
                   *self.extra_args, self.url, self.quality, "--stdout"]
            log.info("waiting for %s to go live...", self.channel)
            # keep streamlink's own messages: they explain disconnects ("why does it reconnect every 7 min?")
            errlog = open(self.outdir / f"{self.channel}.streamlink.log", "ab")
            errlog.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} start\n".encode())
            errlog.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errlog, bufsize=0)
            self.proc = proc
            try:
                self._record(proc)
            finally:
                proc.kill()
                proc.wait()
                errlog.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} exit code {proc.returncode}\n".encode())
                errlog.close()
                self._trim_log(self.outdir / f"{self.channel}.streamlink.log")
            self.stop.wait(10)

    @staticmethod
    def _trim_log(path: Path, keep: int = 200_000):
        try:
            if path.stat().st_size > keep * 2:
                data = path.read_bytes()[-keep:]
                path.write_bytes(data)
        except OSError:
            pass

    def _record(self, proc: subprocess.Popen):
        first = proc.stdout.read(TS_PACKET * 350)  # blocks until the stream is live
        if not first:
            return
        path = self.outdir / f"{self.channel}_{datetime.now():%Y%m%d_%H%M%S}.ts"
        sess = Session(path)
        with self.lock:
            self.sessions.append(sess)
        log.info("LIVE, recording to %s", path)
        with open(path, "wb") as f:
            chunk = first
            while chunk and not self.stop.is_set():
                f.write(chunk)
                f.flush()
                sess.add(time.time(), len(chunk))
                chunk = proc.stdout.read(65536)
        sess.ended = True
        log.info("stream ended (%.1f MB recorded)", sess.size / 1e6)

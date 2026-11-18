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


class BoxScanner:
    """Finds top-level MP4 box boundaries in a byte stream that arrives in chunks.

    Twitch serves some channels as fragmented MP4 (CMAF) instead of MPEG-TS. There a
    slice can only start at a fragment (`moof`), and it needs the init segment
    (`ftyp` + `moov`) in front, so the recorder has to know where those boxes are.
    """

    def __init__(self):
        self.next = 0        # stream offset of the next box header
        self.buf = b""       # stream bytes from buf_start on, kept only while a header is incomplete
        self.buf_start = 0
        self.fed = 0
        self.broken = False  # not a box stream after all: stop scanning

    def feed(self, chunk: bytes) -> list[tuple[int, bytes, int]]:
        """Returns (offset, type, size) for every box header that is complete now."""
        found = []
        if self.broken:
            return found
        if self.next >= self.fed + len(chunk):  # inside a big box (mdat): nothing to look at
            self.fed += len(chunk)
            return found
        if not self.buf:
            self.buf_start = self.fed
        self.buf += chunk
        self.fed += len(chunk)
        while True:
            rel = self.next - self.buf_start
            if rel >= len(self.buf):
                self.buf, self.buf_start = b"", self.fed
                break
            avail = len(self.buf) - rel
            size = int.from_bytes(self.buf[rel:rel + 4], "big") if avail >= 8 else None
            if size == 1 and avail < 16:
                size = None
            if size is None:  # header split across chunks: keep just the start of it
                self.buf, self.buf_start = self.buf[rel:], self.next
                break
            typ = self.buf[rel + 4:rel + 8]
            if size == 1:
                size = int.from_bytes(self.buf[rel + 8:rel + 16], "big")
            if size < 8 or not typ.isascii() or not typ.isalnum():  # size 0 (to end of file) included
                self.broken = True
                break
            found.append((self.next, typ, size))
            self.next += size
        return found


@dataclass
class Session:
    path: Path
    times: list[float] = field(default_factory=list)    # arrival time of each cut point
    offsets: list[int] = field(default_factory=list)    # file offset of that cut point
    size: int = 0
    ended: bool = False
    fmp4: bool = False                                   # fragmented MP4 instead of MPEG-TS
    inits: list[tuple[int, int]] = field(default_factory=list)  # fMP4 init segments: (start, end) in the file

    def add(self, t: float, nbytes: int, boxes: list[tuple[int, bytes, int]] | None = None):
        if self.fmp4:
            # fMP4: only a fragment start is a valid cut point
            for off, typ, size in boxes or []:
                if typ == b"ftyp":
                    self.inits.append((off, off + size))
                elif typ == b"moov" and self.inits:
                    self.inits[-1] = (self.inits[-1][0], off + size)
                elif typ == b"moof" and (not self.offsets or off > self.offsets[-1]):
                    self.times.append(t)
                    self.offsets.append(off)
        # TS: one index entry per ~0.25 s is plenty
        elif not self.times or t - self.times[-1] >= 0.25:
            self.times.append(t)
            self.offsets.append(self.size)
        self.size += nbytes

    def covers(self, t: float) -> bool:
        return bool(self.times) and self.times[0] - 5 <= t and (not self.ended or t <= self.times[-1] + 5)

    def offset_at(self, t: float, end: bool = False) -> int:
        if self.fmp4:
            if not self.times or t < self.times[0]:
                return 0
            i = bisect.bisect_left(self.times, t) if end else bisect.bisect_right(self.times, t) - 1
            return self.offsets[i] if i < len(self.offsets) else self.size
        i = bisect.bisect_right(self.times, t) - 1
        off = self.offsets[max(i, 0)] if t >= self.times[0] else 0
        if t > self.times[-1]:
            off = self.size
        return off - off % TS_PACKET  # stay on a TS packet boundary

    def extract(self, t0: float, t1: float, dest: Path) -> int:
        if not self.fmp4:
            return self._copy(t0, t1, dest)
        # fMP4 slice -> MPEG-TS (stream copy, ~1 s): the mp4 demuxer counts the duration from
        # timestamp 0, so a slice 5 h into the stream would look 5 h long to everything downstream
        frag = dest.with_name(dest.name + ".frag.mp4")
        try:
            n = self._copy(t0, t1, frag)
            if n:
                r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(frag),
                                    "-map", "0", "-c", "copy", "-f", "mpegts", str(dest)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    log.warning("remuxing fMP4 slice failed, keeping it as is: %s", r.stderr.strip()[-300:])
                    frag.replace(dest)
            return n
        finally:
            frag.unlink(missing_ok=True)

    def _copy(self, t0: float, t1: float, dest: Path) -> int:
        a, b = self.offset_at(t0), self.offset_at(t1, end=True)
        init = None
        if self.fmp4:
            init = next((r for r in reversed(self.inits) if r[0] <= a), self.inits[0] if self.inits else None)
            if init and a <= init[1]:
                a, init = init[0], None  # the slice starts with the init segment anyway
        with open(self.path, "rb") as src, open(dest, "wb") as out:
            if init:
                src.seek(init[0])
                out.write(src.read(init[1] - init[0]))
            src.seek(a)
            remaining = b - a
            while remaining > 0:
                chunk = src.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        return max(b - a, 0)


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
            except Exception:
                # a dead recorder thread means silent "0 bytes" for every later spike: log and start over
                log.exception("recording %s failed, restarting", self.channel)
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
        sess = Session(path, fmp4=first[4:8] == b"ftyp")
        scanner = BoxScanner() if sess.fmp4 else None
        with self.lock:
            self.sessions.append(sess)
        log.info("LIVE, recording to %s%s", path, " (fragmented MP4)" if sess.fmp4 else "")
        f = open(path, "wb")
        try:
            chunk = first
            while chunk and not self.stop.is_set():
                f = self._write(f, sess, chunk)
                sess.add(time.time(), len(chunk), scanner.feed(chunk) if scanner else None)
                chunk = proc.stdout.read(65536)
        finally:
            sess.ended = True
            try:
                f.close()
            except OSError:
                pass
        log.info("stream ended (%.1f MB recorded)", sess.size / 1e6)

    @staticmethod
    def _write(f, sess: Session, chunk: bytes, tries: int = 6):
        """Write one chunk; on an I/O error reopen the file at the last good offset and retry.

        Docker Desktop's Windows bind mount (9p) sometimes fails a write with EIO for a
        moment. The index only counts bytes that were fully written, so rewriting
        from sess.size keeps it exact.
        """
        for attempt in range(tries):
            try:
                f.write(chunk)
                f.flush()
                return f
            except (OSError, ValueError) as e:  # ValueError: file still closed after a failed reopen
                if attempt == tries - 1:
                    raise
                log.warning("write to %s failed (%s), retrying", sess.path.name, e)
                time.sleep(min(2 ** attempt, 10))
                try:
                    f.close()
                except OSError:
                    pass
                try:
                    f = open(sess.path, "r+b")
                    f.seek(sess.size)
                except OSError:
                    pass  # still failing; the next write raises and we retry again
        return f

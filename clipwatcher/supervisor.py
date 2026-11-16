"""Runs one watcher (recorder + chat + detector + clipper) per enabled channel,
starts/stops them as channels are added, removed or toggled in the studio,
and exposes live status for the UI."""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from pathlib import Path

from .chat import ChatReader
from .clipper import ClipWorker
from .db import DB
from .detector import DetectorConfig, SpikeDetector
from .recorder import Recorder

log = logging.getLogger("supervisor")


class ChannelWatcher:
    def __init__(self, name: str, data: Path, db: DB, on_clip):
        self.name, self.db = name, db
        self.stop = threading.Event()
        self.started = time.time()
        rec_dir, clip_dir, chat_dir = data / "recordings", data / "clips", data / "chat"
        for d in (rec_dir, clip_dir, chat_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.det = SpikeDetector(self._det_cfg())
        self.det_lock = threading.Lock()
        self.msg_times: deque[float] = deque()
        self.history: deque[tuple[float, float, float]] = deque(maxlen=300)  # (t, score, baseline) every 2 s
        self.last_spike: float | None = None

        s = db.settings()
        self.rec = Recorder(name, rec_dir, s["quality"], self.stop)
        self.clipper = ClipWorker(self.rec, name, clip_dir,
                                  lambda: db.clip_config(name), self.stop, on_clip)
        self.chat = ChatReader(name, chat_dir / f"{name}.jsonl", self._on_msg, self.stop)
        self._on_clip = on_clip
        self.clipper.on_clip = self._with_heat
        self.loop = threading.Thread(target=self._run, daemon=True, name=f"detect-{name}")
        for t in (self.rec, self.clipper, self.chat, self.loop):
            t.start()

    def _with_heat(self, meta: dict):
        """Attach the chat-energy curve around the clip, for the studio's heat strip."""
        start = meta.get("src_start", meta["created"] - 32)
        end = meta["created"] + 30
        pts = [[round(t - start, 1), sc] for (t, sc, _) in self.history if start <= t <= end]
        meta["heat"] = {"points": pts, "baseline": meta["baseline"], "spike": round(meta["created"] - start, 1)}
        self._on_clip(meta)

    def _det_cfg(self) -> DetectorConfig:
        s = self.db.settings()
        row = next((c for c in self.db.channels() if c["name"] == self.name), {})
        return DetectorConfig(ratio=row.get("ratio") or s["ratio"], min_score=row.get("min_score") or s["min_score"],
                              cooldown=s["cooldown"], clip_calls=int(s["clip_calls"]) or 10**9)

    def reload(self):
        with self.det_lock:
            self.det.cfg = self._det_cfg()

    def _on_msg(self, t, user, text):
        with self.det_lock:
            self.det.feed(t, user, text)
        self.msg_times.append(t)

    def _run(self):
        n = 0
        while not self.stop.is_set():
            self.stop.wait(1)
            now = time.time()
            while self.msg_times and self.msg_times[0] < now - 60:
                self.msg_times.popleft()
            with self.det_lock:
                spike = self.det.tick(now)
                sc = self.det.scores
                score = sc[-1][1] if sc else 0.0
                past = [v for (t, v) in sc if now - 310 <= t < now - 10]
                base = max(sum(past) / len(past) if past else 0.0, self.det.cfg.baseline_floor)
            n += 1
            if n % 2 == 0:
                self.history.append((now, round(score, 1), round(base, 1)))
            if spike and self.rec.session_for(spike.t):
                self.last_spike = spike.t
                log.info("[%s] SPIKE %s score=%.1f baseline=%.1f", self.name, spike.reason, spike.score, spike.baseline)
                self.clipper.submit(spike)

    def status(self) -> dict:
        live = self.rec.live_session
        return {
            "live": live is not None,
            "live_since": live.times[0] if live and live.times else None,
            "recorded_mb": round(live.size / 1e6, 1) if live else 0,
            "chat_connected": self.chat.connected,
            "msgs_per_min": len(self.msg_times),
            "last_spike": self.last_spike,
            "pending_jobs": self.clipper.jobs.qsize(),
            "history": list(self.history)[-150:],
        }

    def shutdown(self):
        self.stop.set()
        self.rec.shutdown()
        self.chat.shutdown()


class Supervisor:
    def __init__(self, data: Path, db: DB):
        self.data, self.db = data, db
        self.watchers: dict[str, ChannelWatcher] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def _on_clip(self, meta: dict):
        s = self.db.settings()
        when = time.strftime("%d %b", time.localtime(meta["created"]))
        title = f"{meta['channel']} moment ({when})"
        link = f"https://twitch.tv/{meta['channel']}"
        meta.update(title=title, caption=render_caption(s["caption_template"], title, meta["channel"], link),
                    source_url=link)
        self.db.add_clip(meta)

    def sync(self):
        """Make running watchers match the enabled channels in the DB."""
        want = {c["name"] for c in self.db.channels() if c["enabled"]}
        with self.lock:
            for name in list(self.watchers):
                if name not in want:
                    log.info("stopping %s", name)
                    self.watchers.pop(name).shutdown()
            for name in want - self.watchers.keys():
                log.info("starting %s", name)
                self.watchers[name] = ChannelWatcher(name, self.data, self.db, self._on_clip)
            for w in self.watchers.values():
                w.reload()

    def status(self) -> dict[str, dict]:
        with self.lock:
            return {n: w.status() for n, w in self.watchers.items()}

    def start_housekeeping(self):
        def loop():
            while not self.stop.wait(600):
                try:
                    self.cleanup()
                except Exception:
                    log.exception("cleanup failed")
        threading.Thread(target=loop, daemon=True, name="housekeeping").start()

    def cleanup(self):
        """Delete full recordings older than keep_recordings_hours (clips keep
        their own raw slice, so recordings are only needed for a short while)."""
        keep = float(self.db.settings()["keep_recordings_hours"]) * 3600
        active = {w.rec.live_session.path for w in self.watchers.values() if w.rec.live_session}
        for f in (self.data / "recordings").glob("*.ts"):
            if f not in active and time.time() - f.stat().st_mtime > keep:
                log.info("deleting old recording %s", f.name)
                f.unlink(missing_ok=True)

    def shutdown(self):
        self.stop.set()
        with self.lock:
            for w in self.watchers.values():
                w.shutdown()


def render_caption(template: str, title: str, channel: str, link: str | None = None) -> str:
    """Fill the caption template. {link} = source (Twitch channel or video URL),
    {tag} = channel name usable as a hashtag."""
    tag = re.sub(r"[^0-9A-Za-z_]", "", channel) or "clips"
    if link == "":  # no public source (an uploaded file): drop the lines about it
        template = "\n".join(ln for ln in template.splitlines() if "{link}" not in ln and "{channel}" not in ln)
        template = template.replace(" #{tag}", "")
    try:
        return template.format(title=title, channel=channel, link=link or f"https://twitch.tv/{channel}", tag=tag)
    except (KeyError, IndexError, ValueError):
        return template

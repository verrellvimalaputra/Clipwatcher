"""Turns a Twitch chat spike into a finished clip: cut -> transcribe -> render."""
from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .detector import Spike
from .editing import ClipConfig, duration_of, ffmpeg, render, thumbnail, transcribe  # noqa: F401 (re-exported)
from .recorder import Recorder

log = logging.getLogger("clipper")

class ClipWorker(threading.Thread):
    """One per channel. Waits for the post-roll, cuts bytes via the recorder's
    wall-clock index, keeps the raw slice (so the studio can re-trim or re-layout
    later without the full recording), renders, then calls on_clip(meta)."""

    def __init__(self, recorder: Recorder, channel: str, outdir: Path,
                 get_cfg: Callable[[], ClipConfig], stop: threading.Event,
                 on_clip: Callable[[dict], None] | None = None):
        super().__init__(daemon=True, name=f"clipper-{channel}")
        self.recorder, self.channel, self.outdir, self.get_cfg, self.stop = recorder, channel, outdir, get_cfg, stop
        self.on_clip = on_clip
        self.jobs: queue.Queue[Spike] = queue.Queue()

    def submit(self, spike: Spike):
        self.jobs.put(spike)

    def run(self):
        while not self.stop.is_set():
            try:
                spike = self.jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._make(spike)
            except Exception:
                log.exception("clip failed")

    def _make(self, spike: Spike):
        cfg = self.get_cfg()
        t0, t1 = spike.t - cfg.pre_roll, spike.t + cfg.post_roll
        while time.time() < t1 + 4 and not self.stop.is_set():
            time.sleep(0.5)
        sess = self.recorder.session_for(spike.t)
        if not sess:
            log.warning("spike at %s but nothing recorded then (stream offline?)", spike.t)
            return

        stem = f"{self.channel}_{datetime.fromtimestamp(spike.t):%Y%m%d_%H%M%S}"
        raw = self.outdir / f"{stem}.src.ts"
        n = sess.extract(t0 - 2, t1, raw)  # 2 s extra so decoding can find a keyframe
        if n < 100_000:
            log.warning("only %d bytes for %s, skipping", n, stem)
            raw.unlink(missing_ok=True)
            return

        captions = []
        if cfg.captions:
            try:
                with tempfile.TemporaryDirectory() as td:
                    wav = Path(td) / "a.wav"
                    ffmpeg("-i", str(raw), "-vn", "-ac", "1", "-ar", "16000", str(wav))
                    captions = transcribe(wav, cfg)
            except Exception as e:  # never lose a clip over captions
                log.warning("captions skipped: %s", e)

        out = self.outdir / f"{stem}.mp4"
        used = render(raw, out, cfg, captions)
        thumb = self.outdir / f"{stem}.jpg"
        thumbnail(out, thumb)
        meta = {
            "layout": used, "facecam": cfg.facecam,
            "channel": self.channel, "created": spike.t, "file": out.name, "thumb": thumb.name,
            "src": raw.name, "src_start": t0 - 2, "captions": captions, "duration": duration_of(out),
            "score": spike.score, "baseline": spike.baseline, "reason": spike.reason,
            "chat_sample": spike.sample, "transcript": " ".join(c[2] for c in captions),
        }
        log.info("CLIP %s (score %.1f vs baseline %.1f, %s)", out.name, spike.score, spike.baseline, spike.reason)
        if self.on_clip:
            self.on_clip(meta)

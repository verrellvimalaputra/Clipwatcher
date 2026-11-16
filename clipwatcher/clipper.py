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
from .editing import (FAST_MODEL, ClipConfig, duration_of, ffmpeg, lines_from_words, render,  # noqa: F401
                      thumbnail, transcribe_words)
from .moments import twitch_cut
from .recorder import Recorder

log = logging.getLogger("clipper")
RAW_PRE, RAW_POST = 90.0, 20.0  # seconds of footage kept around each chat spike

class ClipWorker(threading.Thread):
    """One per channel. Waits for the post-roll, cuts bytes via the recorder's
    wall-clock index, keeps the raw slice (so the studio can re-trim or re-layout
    later without the full recording), renders, then calls on_clip(meta)."""

    def __init__(self, recorder: Recorder, channel: str, outdir: Path,
                 get_cfg: Callable[[], ClipConfig], stop: threading.Event,
                 on_clip: Callable[[dict], None] | None = None,
                 get_chat: Callable[[float, float], list] | None = None,
                 context_on: Callable[[], bool] = lambda: True):
        super().__init__(daemon=True, name=f"clipper-{channel}")
        self.recorder, self.channel, self.outdir, self.get_cfg, self.stop = recorder, channel, outdir, get_cfg, stop
        self.on_clip = on_clip
        self.get_chat, self.context_on = get_chat, context_on
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
        # keep a long raw slice: the story often starts well before the 30 s pre-roll,
        # and the studio's trim handles can reach back into it
        t0, t1 = spike.t - max(cfg.pre_roll, RAW_PRE), spike.t + max(cfg.post_roll, RAW_POST)
        while time.time() < t1 + 4 and not self.stop.is_set():
            time.sleep(0.5)
        sess = self.recorder.session_for(spike.t)
        if not sess:
            log.warning("spike at %s but nothing recorded then (stream offline?)", spike.t)
            return

        stem = f"{self.channel}_{datetime.fromtimestamp(spike.t):%Y%m%d_%H%M%S}"
        raw = self.outdir / f"{stem}.src.ts"
        raw_start = max(t0 - 2, sess.times[0] if sess.times else t0 - 2)  # 2 s extra to find a keyframe
        n = sess.extract(raw_start, t1, raw)
        if n < 100_000:
            log.warning("only %d bytes for %s, skipping", n, stem)
            raw.unlink(missing_ok=True)
            return
        dur = duration_of(raw)
        spike_rel = min(spike.t - raw_start, dur)

        # 1. rough transcript of the whole slice (fast model) -> where does the story start/end?
        words, segs = [], []
        try:
            words, segs = transcribe_words(raw, ClipConfig(**{**cfg.__dict__, "whisper_model": FAST_MODEL}))
        except Exception as e:
            log.warning("transcription failed, cutting on time only: %s", e)
        chat = [(t - raw_start, u, x) for t, u, x in (self.get_chat(raw_start, t1) if self.get_chat else [])]
        label = self.label() if hasattr(self, "label") else self.channel
        cut = twitch_cut(words, segs, chat, spike_rel, dur, label, cfg.pre_roll, cfg.post_roll)

        # 2. accurate captions for just the chosen part; rough words cover the rest of the slice
        captions = []
        if cfg.captions:
            captions = self._captions(raw, words, cut["start"], cut["end"], cfg)

        out = self.outdir / f"{stem}.mp4"
        cfg.context = cut["context"] or None if self.context_on() else None
        used = render(raw, out, cfg, captions, cut["start"], cut["end"])
        thumb = self.outdir / f"{stem}.jpg"
        thumbnail(out, thumb)
        meta = {
            "layout": used, "facecam": cfg.facecam, "context": cut["context"] or None,
            "title": cut["title"] or None, "trim_start": cut["start"], "trim_end": cut["end"],
            "channel": self.channel, "created": spike.t, "file": out.name, "thumb": thumb.name,
            "src": raw.name, "src_start": raw_start, "captions": captions, "duration": duration_of(out),
            "score": spike.score, "baseline": spike.baseline, "reason": spike.reason,
            "chat_sample": spike.sample,
            "transcript": " ".join(c[2] for c in captions if cut["start"] <= c[0] < cut["end"]),
            "why": "Claude picked where this story starts and ends" if cut["how"] == "claude" else None,
        }
        log.info("CLIP %s (score %.1f vs baseline %.1f, %s, cut by %s)", out.name, spike.score, spike.baseline,
                 spike.reason, cut["how"])
        if self.on_clip:
            self.on_clip(meta)

    def _captions(self, raw: Path, rough: list, start: float, end: float, cfg: ClipConfig) -> list:
        """Configured (accurate) model on the chosen part only; the fast transcript
        fills the rest of the slice so a widened trim still has captions."""
        if cfg.whisper_model == FAST_MODEL:
            return lines_from_words(rough)
        a, b = max(start - 1.0, 0), end + 1.0
        try:
            with tempfile.TemporaryDirectory() as td:
                part = Path(td) / "part.wav"
                ffmpeg("-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", str(raw), "-vn", "-ac", "1", "-ar", "16000",
                       str(part))
                fine, _ = transcribe_words(part, cfg)
            fine = [[w[0] + a, w[1] + a, w[2]] for w in fine]
        except Exception as e:  # never lose a clip over captions
            log.warning("accurate captions failed, using the rough transcript: %s", e)
            return lines_from_words(rough)
        outside = [w for w in rough if w[1] <= a or w[0] >= b]
        return lines_from_words(sorted(outside + fine, key=lambda w: w[0]))

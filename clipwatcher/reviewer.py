"""AI reviewer: a second, independent Claude pass that judges each finished clip
the way a picky short-form editor would, and scores it 0-10.

It runs in its own thread so it never slows recording or editing. It only sees
what's in the final cut (the trimmed transcript, title, context line, chat),
not the reasoning that picked the moment, so it can't just agree with itself.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import moments
from .db import DB

log = logging.getLogger("reviewer")

MAX_TRIES = 2
FRAMES = 6  # stills in the contact sheet: enough to see what happens, ~1k image tokens


def review_model() -> str:
    """Scoring a short clip is an easy job: Haiku uses far less of the plan's limits than Sonnet."""
    return os.environ.get("CLAUDE_REVIEW_MODEL", "haiku")


def contact_sheet(video: Path, out: Path, duration: float) -> bool:
    """6 evenly spaced frames of the final clip (captions included) in one 3x2 image,
    so the reviewer reads one picture instead of six."""
    d = max(float(duration or 0), 1.0)
    skip = d / FRAMES / 2  # start half a step in, so frames sit in the middle of each sixth
    vf = (f"fps={FRAMES}/{d - skip:.3f},scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)',"
          f"tile=3x{(FRAMES + 2) // 3}")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{skip:.2f}",
                        "-i", str(video), "-vf", vf, "-frames:v", "1", "-q:v", "4", str(out)],
                       capture_output=True, timeout=120)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0

PROMPT = """You review short vertical clips (TikTok / YouTube Shorts / Reels) cut from livestreams, before they get posted.
You get frames.jpg: a contact sheet of 6 stills taken evenly through the clip (left to right, top to bottom), showing what the viewer sees, captions included. Read it first. Use it for what happens visually that the words miss: crashes, fails, chaos, big reactions on the facecam.
You also get the clip's title, the context line shown on screen for the first seconds, the timed transcript of exactly what is in the final cut (times are seconds from the start of the clip), and some live chat from that moment. The transcript may be in German or another language; judge it in its own language and culture, including slang, memes and streamer in-jokes.

Judge it as a strict editor would, for a viewer scrolling past who has never watched this streamer:
- interesting: does the first 3 seconds make you stay, and does something actually happen?
- funny: is there a real laugh (a joke, a fail, a reaction, absurd situation)? 0 if it isn't trying to be funny.
- clear: can a stranger follow it without having watched the stream? Missing setup, unexplained references, or ending before the payoff lower this.
- score: overall chance this does well as a short. Calibrate: most clips are 3-6. 7 = solid, post it. 8-9 = likely to do well. 10 = rare. A clip that isn't funny can still score high if it's interesting and clear.

Reply with only JSON, no other text:
{"interesting": 0-10, "funny": 0-10, "clear": 0-10, "score": 0-10, "verdict": "one short sentence on why", "fix": "one concrete edit that would make it better, or empty", "saw_frames": true if you could look at frames.jpg, else false}
Write verdict and fix in English. Numbers may have one decimal."""


def review_input(c: dict) -> str:
    """What the reviewer sees: only the final cut."""
    t0 = float(c.get("trim_start") or 0)
    t1 = c.get("trim_end")
    t1 = float(t1) if t1 is not None else t0 + float(c.get("duration") or 60)
    lines = []
    for a, b, txt, *_ in c.get("captions") or []:  # live clips add per-word timings as a 4th item
        if b > t0 and a < t1 and str(txt).strip():
            lines.append(f"[{max(a - t0, 0):5.1f}] {str(txt).strip()}")
    chat = [m if isinstance(m, str) else (m.get("text") or m.get("msg") or str(m)) for m in (c.get("chat_sample") or [])][:40]
    parts = [
        f"Streamer: {c.get('channel')}",
        f"Title: {c.get('title') or '(none)'}",
        f"Context line on screen: {c.get('context') or '(none)'}",
        f"Length: {t1 - t0:.0f} s",
        "Transcript of the final cut:",
        "\n".join(lines) or "(no speech found)",
    ]
    if chat:
        parts += ["Live chat at that moment:", "\n".join(f"- {m}" for m in chat)]
    return "\n".join(parts)


def parse_review(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("reviewer returned no JSON")
    d = json.loads(m.group(0))

    def num(k):
        try:
            return round(min(max(float(d.get(k)), 0.0), 10.0), 1)
        except (TypeError, ValueError):
            return None

    out = {k: num(k) for k in ("interesting", "funny", "clear", "score")}
    if out["score"] is None:
        raise ValueError("reviewer returned no score")
    out["verdict"] = str(d.get("verdict") or "")[:300]
    out["fix"] = str(d.get("fix") or "")[:300]
    out["saw_frames"] = d.get("saw_frames") is True
    return out


def review_clip(c: dict, clips_dir: Path | None = None) -> dict:
    files = {}
    with tempfile.TemporaryDirectory() as td:
        video = clips_dir / c["file"] if clips_dir and c.get("file") else None
        sheet = Path(td) / "frames.jpg"
        if video and video.exists():
            try:
                if contact_sheet(video, sheet, c.get("duration") or 0):
                    files["frames.jpg"] = sheet
            except Exception as e:  # no frames is fine: it reviews the text alone
                log.warning("no frames for clip %s: %s", c["id"], e)
        prompt = PROMPT if files else PROMPT.replace("Read it first.", "(Not available for this clip: judge from the text.)")
        text = moments.run_claude(prompt, review_input(c), timeout=300, task="review", model=review_model(),
                                  files=files or None)
    r = parse_review(text)
    r["saw_frames"] = r["saw_frames"] and bool(files)
    r["model"] = review_model()
    return r


class ReviewWorker(threading.Thread):
    """Scores pending clips that don't have a score yet, one at a time."""

    def __init__(self, db: DB, clips_dir: Path | None = None):
        super().__init__(daemon=True, name="reviewer")
        self.db, self.clips_dir = db, clips_dir
        self.wake = threading.Event()

    def poke(self):
        self.wake.set()

    def next_clip(self) -> dict | None:
        rows = self.db.q("SELECT id FROM clips WHERE status='pending' AND ai_score IS NULL AND ai_tries < ? "
                         "ORDER BY created DESC LIMIT 1", (MAX_TRIES,))
        return self.db.clip(rows[0]["id"]) if rows else None

    def run(self):
        while True:
            self.wake.wait(30)
            self.wake.clear()
            while (time.time() >= self.paused_until and self.db.settings().get("ai_review", True)
                   and moments.claude_available()):
                c = self.next_clip()
                if not c:
                    break
                self.review_one(c)

    paused_until = 0.0

    def review_one(self, c: dict) -> dict | None:
        """Score one clip. A bad answer counts against the clip (MAX_TRIES); Claude Code
        being down doesn't: the worker backs off and tries again later."""
        try:
            r = review_clip(c, self.clips_dir)
        except ValueError as e:  # Claude answered, but not with a usable score
            log.warning("review of clip %s unusable: %s", c["id"], e)
            self.db.x("UPDATE clips SET ai_tries=ai_tries+1 WHERE id=?", (c["id"],))
            return None
        except Exception as e:  # login, rate limit, timeout: wait for the next round
            log.warning("AI review paused, Claude Code failed: %s", e)
            self.paused_until = time.time() + 300
            return None
        # the clip may have been re-trimmed while Claude was thinking: then it gets reviewed again
        now = self.db.q("SELECT trim_start, trim_end FROM clips WHERE id=?", (c["id"],))
        if now and (now[0]["trim_start"], now[0]["trim_end"]) == (c.get("trim_start"), c.get("trim_end")):
            self.db.x("UPDATE clips SET ai_score=?, ai_review=? WHERE id=?",
                      (r["score"], json.dumps(r, ensure_ascii=False), c["id"]))
        return r

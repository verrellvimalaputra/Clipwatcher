"""Find the clip-worthy moments in a long video.

Primary: Claude Code (on the user's subscription, never the paid API) reads the
timestamped transcript plus hard signals and picks self-contained moments.
Fallback: a heuristic over the same signals. Both return Moment objects whose
boundaries get snapped to sentence/word edges so clips never start or end mid-word.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("moments")

LENGTHS = {"short": (15, 35), "medium": (30, 60), "long": (55, 90)}


@dataclass
class Moment:
    start: float
    end: float
    title: str
    why: str = ""
    score: float = 50.0
    source: str = "ai"   # ai | heuristic
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def fmt(t: float) -> str:
    t = max(t, 0)
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


# ---------------------------------------------------------------- signals
def heatmap_peaks(heatmap: list[dict] | None, top: int = 12) -> list[tuple[float, float, float]]:
    """YouTube "Most replayed" -> [(start, end, value 0..1)] strongest first."""
    if not heatmap:
        return []
    hm = sorted(heatmap, key=lambda h: -h.get("value", 0))
    return [(h["start_time"], h["end_time"], round(h.get("value", 0), 2)) for h in hm[:top]
            if h.get("value", 0) >= 0.35]


def heat_at(heatmap: list[dict] | None, t: float) -> float:
    for h in heatmap or []:
        if h["start_time"] <= t < h["end_time"]:
            return h.get("value", 0.0)
    return 0.0


def loudness_curve(media: Path, step: float = 0.5) -> list[float]:
    """RMS loudness per `step` seconds (decoded at 8 kHz mono)."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    n = int(8000 * step)
    if len(a) < n:
        return []
    a = a[: len(a) // n * n].reshape(-1, n)
    return [round(float(v), 1) for v in np.sqrt((a ** 2).mean(axis=1))]


def loud_peaks(curve: list[float], step: float = 0.5, top: int = 10) -> list[tuple[float, float]]:
    """Moments noticeably louder than their surroundings (laughs, yelling, music hits)."""
    import numpy as np
    if len(curve) < 120:
        return []
    c = np.array(curve)
    k = int(60 / step)  # compare against the surrounding minute
    local = np.convolve(c, np.ones(k) / k, mode="same") + 1e-6
    ratio = c / local
    win = int(4 / step)
    sm = np.convolve(ratio, np.ones(win) / win, mode="same")
    peaks = []
    for i in np.argsort(-sm):
        t = i * step
        if sm[i] < 1.6 or len(peaks) >= top:
            break
        if all(abs(t - p[0]) > 30 for p in peaks):
            peaks.append((round(t, 1), round(float(sm[i]), 2)))
    return peaks


def youtube_chat(path: Path) -> list[tuple[float, str, str]]:
    """Parse a yt-dlp live_chat.json(l) replay into [(video_offset_s, user, text)]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                act = d["replayChatItemAction"]
                t = int(act["videoOffsetTimeMsec"]) / 1000
                for a in act.get("actions", []):
                    r = a.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer")
                    if not r:
                        continue
                    text = "".join(run.get("text") or (run.get("emoji", {}).get("shortcuts") or [""])[0]
                                   for run in r.get("message", {}).get("runs", []))
                    user = r.get("authorName", {}).get("simpleText", "?")
                    out.append((t, user, text))
            except (ValueError, KeyError, TypeError):
                continue
    return sorted(out)


def chat_peaks(msgs: list[tuple[float, str, str]]) -> list[tuple[float, float, list[str]]]:
    """Run the live spike detector over a chat replay -> [(t, ratio, sample)]."""
    from .detector import DetectorConfig, SpikeDetector
    if not msgs:
        return []
    det = SpikeDetector(DetectorConfig(warmup=60, cooldown=30, min_score=6))
    peaks, i, t, end = [], 0, msgs[0][0], msgs[-1][0]
    while t <= end:
        while i < len(msgs) and msgs[i][0] <= t:
            det.feed(*msgs[i])
            i += 1
        s = det.tick(t)
        if s:
            peaks.append((round(s.t, 1), round(s.score / max(s.baseline, 1), 1), s.sample[-4:]))
        t += 1
    return peaks


# ---------------------------------------------------------------- snapping
def snap(m: Moment, words: list[list], segments: list[list], lo: float, hi: float) -> Moment | None:
    """Move boundaries to sentence edges (±2.5 s), else word edges; enforce length."""
    starts = [s[0] for s in segments]
    ends = [s[1] for s in segments]

    def nearest(vals, t, tol):
        best = min(vals, key=lambda v: abs(v - t), default=None)
        return best if best is not None and abs(best - t) <= tol else None

    s = nearest(starts, m.start, 2.5)
    if s is None:
        s = nearest([w[0] for w in words], m.start, 1.5) or m.start
    e = nearest(ends, m.end, 2.5)
    if e is None:
        e = nearest([w[1] for w in words], m.end, 1.5) or m.end
    if e - s > hi + 5:  # too long: cut at the last sentence end that fits
        fits = [x for x in ends if s + lo <= x <= s + hi]
        e = max(fits) if fits else s + hi
    if e - s < lo - 5:
        return None
    m.start, m.end = round(max(s - 0.15, 0), 2), round(e + 0.35, 2)
    return m


def dedupe(ms: list[Moment]) -> list[Moment]:
    out = []
    for m in sorted(ms, key=lambda m: -m.score):
        if all(m.end <= o.start + 2 or m.start >= o.end - 2 for o in out):
            out.append(m)
    return sorted(out, key=lambda m: m.start)


def target_count(duration: float, count) -> int:
    if str(count).isdigit() and int(count) > 0:
        return min(int(count), 30)
    return max(2, min(15, round(duration / 240)))  # ~1 clip per 4 minutes


# ---------------------------------------------------------------- Claude Code
def claude_available() -> bool:
    if not shutil.which("claude"):
        return False
    return bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")) or (Path.home() / ".claude" / ".credentials.json").exists()


def run_claude(instructions: str, stdin_text: str, timeout: int = 900) -> str:
    """claude -p on the subscription: ANTHROPIC_API_KEY is stripped so it can never
    bill the API, tools are off, and it runs in an empty dir (no CLAUDE.md)."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["claude", "-p", instructions, "--output-format", "json", "--tools", "",
                            "--model", os.environ.get("CLAUDE_MODEL", "sonnet")],
                           input=stdin_text, capture_output=True, text=True, timeout=timeout, cwd=td, env=env)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError((r.stderr or r.stdout).strip()[:300] or "no output from Claude Code")
    if r.returncode != 0 or out.get("is_error"):
        raise RuntimeError(str(out.get("result") or r.stderr)[:300])
    return out.get("result", "")


def _transcript_block(segments, t0, t1) -> str:
    return "\n".join(f"[{fmt(a)}] {txt}" for a, b, txt in segments if b > t0 and a < t1)


def claude_moments(info: dict, segments: list[list], signals: dict, length: str, n: int,
                   t0: float = 0, t1: float | None = None) -> list[Moment]:
    lo, hi = LENGTHS[length]
    t1 = t1 or info.get("duration") or (segments[-1][1] if segments else 0)
    hints = []
    if signals.get("heatmap"):
        hints.append("YouTube 'Most replayed' peaks (viewers rewatched these): " + ", ".join(
            f"{fmt(a)}–{fmt(b)} ({v})" for a, b, v in signals["heatmap"] if t0 <= a < t1))
    if signals.get("chat"):
        hints.append("Live chat spikes (chat reacted right AFTER these): " + ", ".join(
            f"{fmt(t)} ({r}x)" for t, r, _ in signals["chat"] if t0 <= t < t1))
    if signals.get("loud"):
        hints.append("Audio much louder than surroundings (laughs, shouting): " + ", ".join(
            f"{fmt(t)}" for t, _ in signals["loud"] if t0 <= t < t1))
    body = (f"VIDEO: {info.get('title', '')}\nCHANNEL: {info.get('channel', '')}\n"
            f"DESCRIPTION: {(info.get('description') or '')[:600]}\n\n"
            + ("SIGNALS:\n" + "\n".join(hints) + "\n\n" if hints else "")
            + "TRANSCRIPT ([mm:ss] or [h:mm:ss] = start of line):\n" + _transcript_block(segments, t0, t1))
    instructions = (
        f"You are a short-form video editor (like Opus Clip). From the video on stdin, pick the {n} best "
        f"moments to cut into standalone vertical clips for TikTok, YouTube Shorts and Reels.\n"
        f"Rules: each clip {lo}-{hi} seconds; it must make sense with zero context; it must START on a strong "
        f"hook (a bold claim, question, conflict, funny line, surprising fact). Never start on filler like "
        f"'so', 'anyway', 'yeah'. It must END on a payoff or punchline, not mid-thought. No overlapping clips. "
        f"Prefer moments the signals point to, but only if the transcript there is actually good. Skip intros, "
        f"sponsor reads and outros.\n"
        f"Titles: max 70 characters, same language as the video, curiosity-driven but true, no hashtags.\n"
        f"Reply with ONLY JSON: {{\"clips\": [{{\"start\": <seconds>, \"end\": <seconds>, \"title\": \"...\", "
        f"\"why\": \"one sentence on why this will hold attention\", \"score\": <0-100 virality>}}]}}")
    text = run_claude(instructions, body)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError("Claude Code didn't return JSON")
    out = []
    for c in json.loads(m.group(0)).get("clips", []):
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append(Moment(s, e, str(c.get("title", ""))[:100], str(c.get("why", "")),
                              float(c.get("score", 50)), "ai"))
    return out


# ---------------------------------------------------------------- heuristic
def heuristic_moments(info: dict, segments: list[list], words: list[list], signals: dict,
                      loud: list[float], length: str, n: int) -> list[Moment]:
    """No-AI fallback: score windows that start at sentence starts by replay
    heat, chat spikes, loudness and speech density."""
    lo, hi = LENGTHS[length]
    target = (lo + hi) / 2
    hm = info.get("heatmap")
    chat = signals.get("chat", [])
    peaks = signals.get("loud") or []
    cands = []
    for i, (a, _, text) in enumerate(segments):
        if len(text.split()) < 3:
            continue
        e = a + target
        fits = [s[1] for s in segments[i:] if a + lo <= s[1] <= a + hi]
        if fits:
            e = min(fits, key=lambda x: abs(x - (a + target)))
        e = min(e, segments[-1][1])
        L = e - a
        if L < lo:
            continue
        # a good clip builds up to the moment: reward signals in the middle-to-late part,
        # not ones sitting right at the edges
        inside = lambda t: a + 0.25 * L <= t <= e - 3  # noqa: E731
        density = len([w for w in words if a <= w[0] < e]) / max(L, 1)
        heat = max((heat_at(hm, t) for t in range(int(a + 0.2 * L), int(e), 2)), default=0)
        chat_hit = max((r for t, r, _ in chat if a + 0.3 * L <= t <= e + 5), default=0)
        loud_hit = max((r for t, r in peaks if inside(t)), default=0)
        score = 40 * heat + 6 * min(chat_hit, 5) + 10 * min(loud_hit, 3) + 5 * min(density, 3.5)
        title = re.split(r"(?<=[.!?])\s", text.strip())[0][:70]
        cands.append(Moment(a, e, title, "Picked from replay/chat/loudness signals (Claude Code not set up)",
                            round(min(score, 100), 1), "heuristic"))
    return sorted(dedupe(cands), key=lambda m: -m.score)[:n]


def find_moments(info: dict, words: list[list], segments: list[list], signals: dict, loud: list[float],
                 length: str = "medium", count="auto", use_claude: bool = True) -> tuple[list[Moment], str]:
    """Returns (moments sorted by time, finder used: 'claude' | 'heuristic')."""
    lo, hi = LENGTHS.get(length, LENGTHS["medium"])
    dur = info.get("duration") or (segments[-1][1] if segments else 0)
    n = target_count(dur, count)
    finder, ms = "heuristic", []
    if use_claude and claude_available() and segments:
        try:
            chunk = 3600  # transcripts over an hour go in 1 h windows
            for t0 in range(0, int(dur) + 1, chunk):
                share = max(1, round(n * min(chunk, dur - t0) / dur))
                ms += claude_moments(info, segments, signals, length, share, t0, min(t0 + chunk, dur))
            finder = "claude"
        except Exception as e:
            log.warning("Claude Code moment finding failed, using heuristic: %s", e)
            ms = []
    if not ms:
        ms = heuristic_moments(info, segments, words, signals, loud, length, n)
    snapped = [x for x in (snap(m, words, segments, lo, hi) for m in ms) if x]
    best = sorted(dedupe(snapped), key=lambda m: -m.score)[:n]
    return sorted(best, key=lambda m: m.start), finder

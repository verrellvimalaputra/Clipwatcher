"""Chat spike detection.

Pure logic, no I/O: feed it messages with feed(), call tick(now) about once a
second, and it returns a Spike when chat reacts hard enough. Because it has no
I/O, the same class runs live and in `replay` mode over a saved chat log, which
is how you tune the thresholds for a channel.

Score of a window = unique chatters in the last `window` seconds, plus a bonus
for chatters using hype words (KEKW, Pog, "clip it", ...). Counting unique
chatters instead of raw messages stops one spammer or a bot wave from
triggering a clip.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

DEFAULT_BOTS = {
    "nightbot", "streamelements", "streamlabs", "moobot", "fossabot",
    "wizebot", "soundalerts", "sery_bot", "kofistreambot", "botrixoficial",
    "commanderroot", "pokemoncommunitygame",
}

HYPE_WORDS = [
    r"kekw", r"lul", r"lmao+", r"omegalul", r"pog\w*", r"pogchamp", r"poggers",
    r"w+", r"hah(a|ah)+", r"xd+", r"wtf", r"no+ way", r"holy", r"insane",
    r"clip", r"monkas", r"pepelaugh", r"icant", r"aware", r"kappa",
]
HYPE_RE = re.compile(r"(?<!\w)(" + "|".join(HYPE_WORDS) + r")(?!\w)", re.IGNORECASE)
CLIP_RE = re.compile(r"(?<!\w)clip(\s*(it|that|this))?(?!\w)", re.IGNORECASE)


@dataclass
class Spike:
    t: float                 # wall-clock time the spike was detected
    score: float
    baseline: float
    reason: str              # "ratio" or "clip-calls"
    sample: list[str] = field(default_factory=list)  # a few messages from the window


@dataclass
class DetectorConfig:
    window: float = 10.0       # seconds per scoring window
    baseline_span: float = 300.0  # compare against the previous 5 minutes
    ratio: float = 3.0         # spike if score >= ratio * baseline
    min_score: float = 8.0     # ...and at least this many (weighted) chatters
    baseline_floor: float = 2.0  # dead chats: treat baseline as at least this
    warmup: float = 120.0      # need this much history before triggering
    cooldown: float = 60.0     # min seconds between two spikes
    hype_bonus: float = 0.5    # extra weight per chatter using a hype word
    clip_calls: int = 3        # this many different people saying "clip" triggers too
    bots: set[str] = field(default_factory=lambda: set(DEFAULT_BOTS))


class SpikeDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.msgs: deque[tuple[float, str, str]] = deque()
        self.scores: deque[tuple[float, float]] = deque()
        self.first_t: float | None = None
        self.last_spike = float("-inf")

    def feed(self, t: float, user: str, text: str) -> None:
        user = user.lower()
        if user in self.cfg.bots or text.startswith("!"):
            return  # bots and chat commands are noise
        if self.first_t is None:
            self.first_t = t
        self.msgs.append((t, user, text))

    def _window(self, now: float):
        lo = now - self.cfg.window
        return [m for m in self.msgs if lo < m[0] <= now]

    def _score(self, win) -> tuple[float, int]:
        users = {}
        for _, u, text in win:
            users[u] = users.get(u, False) or bool(HYPE_RE.search(text))
        hype = sum(users.values())
        clippers = len({u for _, u, text in win if CLIP_RE.search(text)})
        return len(users) + self.cfg.hype_bonus * hype, clippers

    def tick(self, now: float) -> Spike | None:
        c = self.cfg
        keep = now - c.baseline_span - c.window - 5
        while self.msgs and self.msgs[0][0] < keep:
            self.msgs.popleft()
        while self.scores and self.scores[0][0] < keep:
            self.scores.popleft()

        win = self._window(now)
        score, clippers = self._score(win)

        # baseline: mean score over the span *before* the current window
        past = [s for (t, s) in self.scores if now - c.window - c.baseline_span <= t < now - c.window]
        baseline = max(sum(past) / len(past) if past else 0.0, c.baseline_floor)
        self.scores.append((now, score))

        if self.first_t is None or now - self.first_t < c.warmup:
            return None
        if now - self.last_spike < c.cooldown:
            return None

        reason = None
        if score >= c.min_score and score >= c.ratio * baseline:
            reason = "ratio"
        elif clippers >= c.clip_calls:
            reason = "clip-calls"
        if not reason:
            return None

        self.last_spike = now
        sample = [f"{u}: {txt}" for _, u, txt in win[-8:]]
        return Spike(t=now, score=round(score, 1), baseline=round(baseline, 1), reason=reason, sample=sample)

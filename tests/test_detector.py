"""Detector tests on synthetic chat: real moments must trigger, bot floods and
single spammers must not."""
import random

from clipwatcher.detector import SpikeDetector


def synth_chat(seed=1):
    random.seed(seed)
    t0, out = 1_700_000_000, []
    users = [f"viewer{i}" for i in range(300)]
    for s in range(3600):
        t = t0 + s
        for _ in range(random.choice([0, 1, 1, 2])):
            out.append((t + random.random(), random.choice(users), random.choice(["hi", "lol", "nice", "gg"])))
        if 900 <= s < 915:    # real moment
            for _ in range(12):
                out.append((t + random.random(), random.choice(users), random.choice(["KEKW", "LMAOOO", "no way", "CLIP IT"])))
        if 1800 <= s < 1810:  # bot flood: must be ignored
            for _ in range(30):
                out.append((t + random.random(), "StreamElements", "thanks for the sub!"))
        if 2400 <= s < 2410:  # one spammer: must be ignored
            for _ in range(20):
                out.append((t + random.random(), "spammer42", "POG POG POG"))
        if 3000 <= s < 3020:  # second real moment
            for _ in range(8):
                out.append((t + random.random(), random.choice(users), random.choice(["Pog", "WWWW", "holy"])))
    out.sort()
    return t0, out


def run(msgs, det):
    spikes, i = [], 0
    t, end = msgs[0][0], msgs[-1][0]
    while t <= end:
        while i < len(msgs) and msgs[i][0] <= t:
            det.feed(*msgs[i])
            i += 1
        s = det.tick(t)
        if s:
            spikes.append(s)
        t += 1
    return spikes


def test_real_moments_trigger_and_noise_does_not():
    t0, msgs = synth_chat()
    spikes = run(msgs, SpikeDetector())
    offsets = sorted(round(s.t - t0) for s in spikes)
    assert len(offsets) == 2, offsets
    assert 898 <= offsets[0] <= 920
    assert 2998 <= offsets[1] <= 3025


def test_cooldown_blocks_double_clip():
    t0, msgs = synth_chat()
    det = SpikeDetector()
    det.cfg.cooldown = 10_000
    assert len(run(msgs, det)) == 1

"""clipwatcher: clip live Twitch streams wherever chat spikes.

  python -m clipwatcher studio                 # web studio: many channels, review, auto-post
  python -m clipwatcher run CHANNEL            # headless, one channel, no review
  python -m clipwatcher replay data/chat/x.jsonl  # test thresholds on a saved chat log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

from .chat import ChatReader
from .clipper import ClipConfig, ClipWorker
from .detector import DetectorConfig, SpikeDetector
from .recorder import Recorder

log = logging.getLogger("clipwatcher")


def env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def add_detector_args(p):
    d = DetectorConfig()
    p.add_argument("--ratio", type=float, default=env("RATIO", d.ratio, float),
                   help=f"spike = score >= ratio x 5-min baseline (default {d.ratio})")
    p.add_argument("--min-score", type=float, default=env("MIN_SCORE", d.min_score, float),
                   help=f"minimum unique-chatter score for a spike (default {d.min_score})")
    p.add_argument("--cooldown", type=float, default=env("COOLDOWN", d.cooldown, float))
    p.add_argument("--window", type=float, default=env("WINDOW", d.window, float))
    p.add_argument("--clip-calls", type=int, default=env("CLIP_CALLS", d.clip_calls, int),
                   help="N different people typing 'clip' also triggers (0 = off)")


def detector_from(a) -> SpikeDetector:
    return SpikeDetector(DetectorConfig(ratio=a.ratio, min_score=a.min_score, cooldown=a.cooldown,
                                        window=a.window, clip_calls=a.clip_calls or 10**9))


def cmd_run(a):
    data = Path(a.data)
    (data / "recordings").mkdir(parents=True, exist_ok=True)
    (data / "clips").mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    det = detector_from(a)
    lock = threading.Lock()

    def on_msg(t, user, text):
        with lock:
            det.feed(t, user, text)

    rec = Recorder(a.channel, data / "recordings", a.quality, stop)
    cfg = ClipConfig(pre_roll=a.pre, post_roll=a.post, captions=not a.no_captions,
                     whisper_model=a.model, language=a.lang, layout="fit" if a.vertical else "original")

    def save_meta(meta):
        keep = {k: v for k, v in meta.items() if k not in ("captions",)}
        keep["credit"] = f"Clip from twitch.tv/{a.channel}, go follow them!"
        (data / "clips" / (Path(meta["file"]).stem + ".json")).write_text(json.dumps(keep, indent=2, ensure_ascii=False))

    clipper = ClipWorker(rec, a.channel, data / "clips", lambda: cfg, stop, save_meta)
    chat = ChatReader(a.channel, data / "chat.jsonl", on_msg, stop)
    for th in (rec, clipper, chat):
        th.start()

    try:
        while not stop.is_set():
            time.sleep(1)
            with lock:
                spike = det.tick(time.time())
            if spike:
                live = rec.session_for(spike.t) is not None
                log.info("SPIKE %s score=%.1f baseline=%.1f (%s)%s", spike.reason, spike.score,
                         spike.baseline, datetime.fromtimestamp(spike.t).strftime("%H:%M:%S"),
                         "" if live else " [not recording, ignored]")
                with open(data / "spikes.jsonl", "a") as f:
                    f.write(json.dumps(spike.__dict__) + "\n")
                if live:
                    clipper.submit(spike)
    except KeyboardInterrupt:
        stop.set()
    log.info("shutting down")


def cmd_studio(a):
    import uvicorn
    from .web import create_app
    if a.host not in ("127.0.0.1", "localhost") and not os.environ.get("STUDIO_PASSWORD"):
        log.warning("studio is reachable from the network without a password: set STUDIO_PASSWORD")
    uvicorn.run(create_app(Path(a.data)), host=a.host, port=a.port, log_level="warning")


def cmd_replay(a):
    msgs = [json.loads(l) for l in open(a.chatlog, encoding="utf-8") if l.strip()]
    if not msgs:
        print("empty log")
        return
    det = detector_from(a)
    i, t, end, n = 0, msgs[0]["t"], msgs[-1]["t"], 0
    while t <= end:
        while i < len(msgs) and msgs[i]["t"] <= t:
            det.feed(msgs[i]["t"], msgs[i]["user"], msgs[i]["text"])
            i += 1
        s = det.tick(t)
        if s:
            n += 1
            print(f"{datetime.fromtimestamp(s.t):%Y-%m-%d %H:%M:%S}  {s.reason:10} "
                  f"score={s.score:<6} baseline={s.baseline:<5}  e.g. {s.sample[-1][:60]!r}")
        t += 1
    hours = (end - msgs[0]["t"]) / 3600
    print(f"\n{n} spikes in {hours:.1f} h of chat ({len(msgs)} msgs). "
          f"Too many? raise --ratio/--min-score. Too few? lower them.")


def main():
    p = argparse.ArgumentParser(prog="clipwatcher", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="watch a channel and clip spikes")
    r.add_argument("channel", nargs="?", default=env("CHANNEL", None))
    r.add_argument("--data", default=env("DATA_DIR", "data"))
    r.add_argument("--quality", default=env("QUALITY", "best"),
                   help="streamlink quality, e.g. best, 1080p60, 720p60")
    r.add_argument("--pre", type=float, default=env("PRE_ROLL", 30.0, float))
    r.add_argument("--post", type=float, default=env("POST_ROLL", 15.0, float))
    r.add_argument("--no-captions", action="store_true", default=env("NO_CAPTIONS", "") == "1")
    r.add_argument("--model", default=env("WHISPER_MODEL", "small"))
    r.add_argument("--lang", default=env("LANGUAGE", None), help="caption language, default auto")
    r.add_argument("--vertical", action="store_true", default=env("VERTICAL", "") == "1",
                   help="9:16 for Shorts/Reels/TikTok (whole frame on a blurred background)")
    add_detector_args(r)
    r.set_defaults(fn=cmd_run)

    st = sub.add_parser("studio", help="web studio: channels, clip review, auto-posting")
    st.add_argument("--data", default=env("DATA_DIR", "data"))
    st.add_argument("--host", default=env("HOST", "127.0.0.1"))
    st.add_argument("--port", type=int, default=env("PORT", 8080, int))
    st.set_defaults(fn=cmd_studio)

    rp = sub.add_parser("replay", help="run the detector over a saved chat.jsonl")
    rp.add_argument("chatlog")
    add_detector_args(rp)
    rp.set_defaults(fn=cmd_replay)

    a = p.parse_args()
    if a.cmd == "run" and not a.channel:
        p.error("give a channel (or set CHANNEL)")
    if a.cmd == "run":
        a.channel = a.channel.lower().removeprefix("https://twitch.tv/").strip("/")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-10s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    a.fn(a)


if __name__ == "__main__":
    main()

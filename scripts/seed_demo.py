"""Fill a studio data dir with a few fake channels and rendered clips, so you can
work on the UI without a live Twitch stream.

    python scripts/seed_demo.py data-demo
    CLIPWATCHER_DEMO=1 python -m clipwatcher studio --data data-demo
"""
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clipwatcher.clipper import ClipConfig, duration_of, render, thumbnail  # noqa: E402
from clipwatcher.db import DB  # noqa: E402
from clipwatcher.supervisor import render_caption  # noqa: E402

data = Path(sys.argv[1] if len(sys.argv) > 1 else "data-demo")
clips = data / "clips"
clips.mkdir(parents=True, exist_ok=True)
db = DB(data / "studio.db")
for ch in ("papaplatte", "trymacs", "windah_basudara"):
    db.add_channel(ch)
db.x("UPDATE channels SET enabled=0")  # demo channels shouldn't try to record

said = ["no way he actually hit that shot from across the map",
        "chat I swear the controller disconnected",
        "okay that was the worst parachute landing ever"]
chat = [["mika: KEKW", "zoe_: NO WAY", "kev: CLIP IT"], ["a1: Pog", "b2: WWWW", "c3: holy"],
        ["yy: KEKW", "zz: no way", "vv: HAHAHA"]]
random.seed(3)
now = time.time()
for i, ch in enumerate(["papaplatte", "windah_basudara", "papaplatte"]):
    stem = f"{ch}_demo{i}"
    raw = clips / f"{stem}.src.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
                    "-f", "lavfi", "-i", f"sine=frequency={300 + i * 120}:duration=20",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-c:a", "aac", "-f", "mpegts", str(raw)],
                   check=True)
    words = said[i].split()
    caps = [(j * 1.2 + 1, j * 1.2 + 2.1, " ".join(words[j * 3:j * 3 + 3])) for j in range((len(words) + 2) // 3)]
    out = clips / f"{stem}.mp4"
    render(raw, out, ClipConfig(layout="fit"), caps)
    thumbnail(out, clips / f"{stem}.jpg")
    base = random.uniform(5, 9)
    pts = [[float(t), round(base + random.uniform(-1.5, 1.5) + (max(0, 9 - abs(t - 17)) * 3 if t > 13 else 0), 1)]
           for t in range(0, 21)]
    created = now - (i * 1900 + 400)
    title = f"{ch} moment ({time.strftime('%d %b', time.localtime(created))})"
    db.add_clip({"channel": ch, "created": created, "file": out.name, "src": raw.name, "captions": caps,
                 "thumb": f"{stem}.jpg", "duration": duration_of(out), "score": max(p[1] for p in pts),
                 "baseline": round(base, 1), "reason": "ratio", "chat_sample": chat[i],
                 "heat": {"points": pts, "baseline": round(base, 1), "spike": 14.0}, "transcript": said[i],
                 "title": title, "caption": render_caption(db.settings()["caption_template"], title, ch)})
print(f"seeded {data}: {len(db.q('SELECT id FROM clips'))} clips")

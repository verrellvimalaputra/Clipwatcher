"""SQLite storage: channels, clips, posts, settings, platform tokens."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    name TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    ratio REAL, min_score REAL,          -- NULL = use global default
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    created REAL NOT NULL,               -- spike wall-clock time
    file TEXT NOT NULL,                  -- rendered clip (relative to clips dir)
    src TEXT,                            -- raw .ts slice; re-renders (trim/layout) start from this
    captions TEXT,                       -- JSON [[start, end, text], ...] relative to src
    trim_start REAL NOT NULL DEFAULT 0, trim_end REAL,
    thumb TEXT,
    duration REAL,
    score REAL, baseline REAL, reason TEXT,
    chat_sample TEXT,                    -- JSON list
    heat TEXT,                           -- JSON {"points": [[t_rel_src, score]], "baseline": x, "spike": t_rel_src}
    transcript TEXT,
    title TEXT, caption TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    reviewed REAL
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id),
    platform TEXT NOT NULL,              -- youtube | instagram | tiktok
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | uploading | posted | failed
    remote_id TEXT, url TEXT, error TEXT, note TEXT,
    updated REAL NOT NULL,
    UNIQUE(clip_id, platform)
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS brands (             -- one set of YouTube/Instagram/TikTok accounts to post to
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',           -- default speech language for its channels ('' = detect)
    caption_template TEXT,                       -- NULL = the global caption template
    handle TEXT NOT NULL DEFAULT ''              -- our page's handle shown small on the video, '' = none
);
CREATE TABLE IF NOT EXISTS ai_usage (           -- Claude Code calls per local day and task
    day TEXT NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
    calls INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
    input INTEGER NOT NULL DEFAULT 0, cached INTEGER NOT NULL DEFAULT 0, output INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0,               -- what it would cost at API prices (you're not billed this)
    PRIMARY KEY (day, task, model)
);
CREATE TABLE IF NOT EXISTS jobs (                -- video imports (YouTube link or upload)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,                  -- url | upload
    url TEXT, title TEXT, channel TEXT, duration REAL,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | downloading | transcribing | finding | rendering | done | failed | cancelled
    step TEXT, progress REAL NOT NULL DEFAULT 0,
    options TEXT,                        -- JSON {length, count, layout, language, chat}
    error TEXT, clips INTEGER NOT NULL DEFAULT 0, finder TEXT,
    created REAL NOT NULL, updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS clips_status ON clips(status, created);
"""

DEFAULT_SETTINGS = {
    "ratio": 3.0, "min_score": 8.0, "cooldown": 60.0, "clip_calls": 3,
    "pre_roll": 30.0, "post_roll": 15.0, "quality": "best",
    "captions": True, "whisper_model": "small", "language": "",
    "layout": "split",          # split (facecam top, gameplay below) | auto (follows faces) | fit | crop | original
    "caption_template": "{title}\n\n🎥 {channel} → {link}\n#shorts #{tag}",
    "youtube_privacy": "public",
    "tiktok_mode": "inbox",     # inbox (draft, you tap Post in the app) | direct
    "tiktok_privacy": "SELF_ONLY",
    "keep_recordings_hours": 48,
    "context_overlay": True,
    "ai_review": True,
    "credit_overlay": True,     # streamer credit (+ the brand's handle, smaller) on the video the whole clip          # a second Claude pass scores each clip 0-10 (interesting, funny, clear)    # show each clip's context line on the video for the first seconds
    "import_layout": "auto",    # default layout for imported videos (talking heads, podcasts)
    "ytdlp_cookies": "",        # path to a cookies.txt if YouTube asks to "confirm you're not a bot"
    "public_url": "",           # https://studio.example.com, needed for Instagram + OAuth redirects
}


class DB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()
        self._migrate()

    def _migrate(self):
        """Add columns introduced after a DB was first created."""
        have_jobs = {r["name"] for r in self.q("PRAGMA table_info(jobs)")}
        if "mode" not in have_jobs:
            self.x("ALTER TABLE jobs ADD COLUMN mode TEXT")
        if "brand" not in have_jobs:
            self.x("ALTER TABLE jobs ADD COLUMN brand INTEGER NOT NULL DEFAULT 1")
        if "handle" not in {r["name"] for r in self.q("PRAGMA table_info(brands)")}:
            self.x("ALTER TABLE brands ADD COLUMN handle TEXT NOT NULL DEFAULT ''")
        if not self.q("SELECT id FROM brands WHERE id=1"):  # existing accounts become brand 1
            self.x("INSERT INTO brands(id, name) VALUES(1, 'Main')")
        have_ch = {r["name"] for r in self.q("PRAGMA table_info(channels)")}
        for col, ddl in [("platform", "TEXT NOT NULL DEFAULT 'twitch'"), ("url", "TEXT"), ("label", "TEXT"),
                         ("brand", "INTEGER NOT NULL DEFAULT 1")]:
            if col not in have_ch:
                self.x(f"ALTER TABLE channels ADD COLUMN {col} {ddl}")
        have = {r["name"] for r in self.q("PRAGMA table_info(clips)")}
        for col, ddl in [("source_kind", "TEXT NOT NULL DEFAULT 'twitch'"), ("source_url", "TEXT"),
                         ("job_id", "INTEGER"), ("why", "TEXT"), ("layout", "TEXT"), ("facecam", "TEXT"), ("context", "TEXT"),
                         ("ai_score", "REAL"), ("ai_review", "TEXT"), ("ai_tries", "INTEGER NOT NULL DEFAULT 0"),
                         ("brand", "INTEGER NOT NULL DEFAULT 1")]:
            if col not in have:
                self.x(f"ALTER TABLE clips ADD COLUMN {col} {ddl}")

    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def x(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            return cur.lastrowid

    # ---- settings / secrets (kv) ----
    def get(self, key: str, default=None):
        r = self.q("SELECT v FROM kv WHERE k=?", (key,))
        return json.loads(r[0]["v"]) if r else default

    def put(self, key: str, value) -> None:
        self.x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
               (key, json.dumps(value)))

    def delete(self, key: str) -> None:
        self.x("DELETE FROM kv WHERE k=?", (key,))

    def clip_config(self, channel: str | None = None):
        """ClipConfig from the settings, with the channel's saved facecam box (if any)."""
        from .editing import ClipConfig
        cfg = ClipConfig.from_settings(self.settings())
        if channel:
            cfg.facecam = self.get(f"facecam:{channel}")
            ch = self.channel(channel)
            if not cfg.language and ch:  # the brand's language beats auto-detect
                cfg.language = self.brand(ch.get("brand") or 1).get("language") or None
            cfg.credit, cfg.handle = self.overlay(channel, (ch or {}).get("brand") or 1)
        return cfg

    def overlay(self, channel: str, brand: int, video_channel: str | None = None) -> tuple:
        """(credit, handle) to burn into a clip: the streamer first, our page smaller.
        video_channel: the uploader's name, for clips cut from imported videos."""
        if not self.settings().get("credit_overlay", True):
            return None, None
        ch = self.channel(channel) if channel else None
        if ch and (ch.get("platform") or "twitch") == "twitch":
            credit = f"twitch.tv/{ch['name']}"
        elif ch:
            credit = f"YouTube: {ch.get('label') or ch['name'].removeprefix('yt-')}"
        else:
            credit = f"YouTube: {video_channel or channel}" if (video_channel or channel) else None
        if credit and credit.endswith(("Upload", ": Video")):  # uploads with no real channel name
            credit = None
        handle = (self.brand(brand).get("handle") or "").strip() or None
        if handle and not handle.startswith("@"):
            handle = "@" + handle
        return credit, handle

    # brands: one set of posting accounts each (e.g. German and Indonesian clip pages)
    def brands(self) -> list[dict]:
        return self.q("SELECT * FROM brands ORDER BY id")

    def brand(self, bid: int) -> dict:
        r = self.q("SELECT * FROM brands WHERE id=?", (bid,))
        return r[0] if r else self.q("SELECT * FROM brands WHERE id=1")[0]

    def caption_template(self, bid: int) -> str:
        return self.brand(bid).get("caption_template") or self.settings()["caption_template"]

    def settings(self) -> dict:
        return {k: self.get(f"setting:{k}", v) for k, v in DEFAULT_SETTINGS.items()}

    def update_settings(self, patch: dict) -> dict:
        for k, v in patch.items():
            if k in DEFAULT_SETTINGS:
                default = DEFAULT_SETTINGS[k]
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, (int, float)):
                    v = type(default)(v)
                self.put(f"setting:{k}", v)
        return self.settings()

    # ---- channels ----
    def channels(self) -> list[dict]:
        return self.q("SELECT * FROM channels ORDER BY added")

    def add_channel(self, name: str, platform: str = "twitch", url: str | None = None,
                    label: str | None = None, brand: int = 1) -> None:
        self.x("INSERT OR IGNORE INTO channels(name, added, platform, url, label, brand) VALUES(?,?,?,?,?,?)",
               (name, time.time(), platform, url, label, brand))

    def channel(self, name: str) -> dict | None:
        r = self.q("SELECT * FROM channels WHERE name=?", (name,))
        return r[0] if r else None

    # ---- clips ----
    def add_clip(self, c: dict) -> int:
        return self.x(
            """INSERT INTO clips(channel, created, file, src, captions, thumb, duration, score, baseline,
               reason, chat_sample, heat, transcript, title, caption, source_kind, source_url, job_id, why,
               layout, trim_start, trim_end, facecam, context, brand)
               VALUES(:channel,:created,:file,:src,:captions,:thumb,:duration,:score,:baseline,:reason,
               :chat_sample,:heat,:transcript,:title,:caption,:source_kind,:source_url,:job_id,:why,
               :layout,:trim_start,:trim_end,:facecam,:context,:brand)""",
            {"source_kind": "twitch", "source_url": None, "job_id": None, "why": None, "layout": None, "context": None,
             "brand": 1,
             "trim_start": 0.0, "trim_end": None, **c, "chat_sample": json.dumps(c.get("chat_sample", []), ensure_ascii=False),
             "captions": json.dumps(c.get("captions", []), ensure_ascii=False),
             "heat": json.dumps(c.get("heat")),
             "facecam": json.dumps(c["facecam"]) if c.get("facecam") else None})

    def add_usage(self, task: str, out: dict):
        """Record one `claude -p --output-format json` result."""
        import time as _t
        u = out.get("usage") or {}
        models = list((out.get("modelUsage") or {}).keys())
        fresh = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
        self.x("""INSERT INTO ai_usage(day, task, model, calls, failed, input, cached, output, cost)
                  VALUES(?,?,?,1,?,?,?,?,?)
                  ON CONFLICT(day, task, model) DO UPDATE SET calls=calls+1, failed=failed+excluded.failed,
                  input=input+excluded.input, cached=cached+excluded.cached, output=output+excluded.output,
                  cost=cost+excluded.cost""",
               (_t.strftime("%Y-%m-%d"), task, ",".join(sorted(models)), 1 if out.get("is_error") else 0, fresh,
                int(u.get("cache_read_input_tokens") or 0), int(u.get("output_tokens") or 0),
                float(out.get("total_cost_usd") or 0)))

    def usage(self, days: int = 7) -> list[dict]:
        import time as _t
        since = _t.strftime("%Y-%m-%d", _t.localtime(_t.time() - (days - 1) * 86400))
        return self.q("SELECT * FROM ai_usage WHERE day>=? ORDER BY day DESC, task", (since,))

    def clip(self, cid: int) -> dict | None:
        r = self.q("SELECT * FROM clips WHERE id=?", (cid,))
        if not r:
            return None
        c = r[0]
        c["chat_sample"] = json.loads(c["chat_sample"] or "[]")
        c["captions"] = json.loads(c["captions"] or "[]")
        c["heat"] = json.loads(c["heat"] or "null")
        c["facecam"] = json.loads(c["facecam"] or "null")
        c["ai_review"] = json.loads(c["ai_review"] or "null")
        c["channel_facecam"] = self.get(f"facecam:{c['channel']}")
        c["posts"] = self.q("SELECT * FROM posts WHERE clip_id=? ORDER BY platform", (cid,))
        return c

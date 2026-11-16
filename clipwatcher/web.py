"""clipwatcher studio: web UI + JSON API."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .clipper import ClipConfig, duration_of, render, thumbnail
from .db import DB
from .importer import ImportWorker
from .moments import claude_available
from .publishers import PostWorker, new_state, platforms, public_token
from .supervisor import Supervisor, render_caption

log = logging.getLogger("studio")
STATIC = Path(__file__).parent / "static"
CHANNEL_RE = re.compile(r"^[a-z0-9_]{3,25}$")


def create_app(data: Path) -> FastAPI:
    data.mkdir(parents=True, exist_ok=True)
    clips_dir = data / "clips"
    clips_dir.mkdir(exist_ok=True)
    db = DB(data / "studio.db")
    secret = db.get("secret") or secrets.token_hex(32)
    db.put("secret", secret)
    password = os.environ.get("STUDIO_PASSWORD", "")

    def public_base() -> str:
        return (os.environ.get("PUBLIC_URL") or db.settings()["public_url"] or "").rstrip("/")

    sup = Supervisor(data, db)
    poster = PostWorker(db, clips_dir, public_base, secret)
    importer = ImportWorker(db, data)
    app = FastAPI(title="clipwatcher studio", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start():
        sup.sync()
        sup.start_housekeeping()
        poster.start()
        importer.start()

    @app.on_event("shutdown")
    def _stop():
        sup.shutdown()

    # ---------------------------------------------------------------- auth
    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        if password and not request.url.path.startswith("/public/"):
            ok = False
            h = request.headers.get("authorization", "")
            if h.startswith("Basic "):
                try:
                    _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                    ok = secrets.compare_digest(pw, password)
                except Exception:
                    ok = False
            if not ok:
                return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="clipwatcher studio"'})
        return await call_next(request)

    def base_url(request: Request) -> str:
        return public_base() or str(request.base_url).rstrip("/")

    def get_clip(cid: int) -> dict:
        c = db.clip(cid)
        if not c:
            raise HTTPException(404, "clip not found")
        return c

    # ---------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/media/{name}")
    def media(name: str):
        f = clips_dir / Path(name).name
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f)

    @app.get("/public/{token}/{name}")
    def public_media(token: str, name: str):
        name = Path(name).name
        if not secrets.compare_digest(token, public_token(secret, name)) or not (clips_dir / name).exists():
            raise HTTPException(404)
        return FileResponse(clips_dir / name, media_type="video/mp4")

    # ---------------------------------------------------------------- state
    @app.get("/api/state")
    def state():
        st = sup.status()
        counts = {r["channel"] + ":" + r["status"]: r["n"] for r in
                  db.q("SELECT channel, status, COUNT(*) n FROM clips GROUP BY channel, status")}
        chans = []
        for c in db.channels():
            chans.append({**c, "status": st.get(c["name"]),
                          "pending": counts.get(c["name"] + ":pending", 0),
                          "total": sum(v for k, v in counts.items() if k.startswith(c["name"] + ":"))})
        totals = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) n FROM clips GROUP BY status")}
        posting = db.q("SELECT COUNT(*) n FROM posts WHERE status IN ('queued','uploading')")[0]["n"]
        failed = db.q("SELECT COUNT(*) n FROM posts WHERE status='failed'")[0]["n"]
        importing = db.q("SELECT COUNT(*) n FROM jobs WHERE status NOT IN ('done','failed','cancelled')")[0]["n"]
        with bulk_lock:
            rer = {"left": len(bulk["queue"]), "done": bulk["done"], "total": bulk["total"]}
        return {"channels": chans, "totals": totals, "posting": posting, "failed_posts": failed,
                "importing": importing, "rerendering": rer,
                "platforms": [p.info() for p in platforms(db).values()],
                "claude": claude_available(),
                "demo": os.environ.get("CLIPWATCHER_DEMO") == "1"}

    # ---------------------------------------------------------------- channels
    @app.post("/api/channels")
    def add_channel(body: dict = Body(...)):
        name = str(body.get("name", "")).strip().lower()
        name = re.sub(r"^(https?://)?(www\.)?twitch\.tv/", "", name).strip("/@ ")
        if not CHANNEL_RE.match(name):
            raise HTTPException(400, "That doesn't look like a Twitch channel name")
        db.add_channel(name)
        sup.sync()
        return {"ok": True, "name": name}

    @app.patch("/api/channels/{name}")
    def edit_channel(name: str, body: dict = Body(...)):
        if "enabled" in body:
            db.x("UPDATE channels SET enabled=? WHERE name=?", (1 if body["enabled"] else 0, name))
        for k in ("ratio", "min_score"):
            if k in body:
                v = body[k]
                db.x(f"UPDATE channels SET {k}=? WHERE name=?", (float(v) if v not in (None, "") else None, name))
        sup.sync()
        return {"ok": True}

    @app.delete("/api/channels/{name}/facecam")
    def forget_facecam(name: str):
        db.delete(f"facecam:{name}")
        return {"ok": True}

    @app.delete("/api/channels/{name}")
    def delete_channel(name: str):
        db.x("DELETE FROM channels WHERE name=?", (name,))
        sup.sync()
        return {"ok": True}

    # ---------------------------------------------------------------- clips
    @app.get("/api/clips")
    def list_clips(status: str = "pending", channel: str = "", job: int = 0, limit: int = 200):
        sql = ("SELECT id, channel, created, file, thumb, duration, score, baseline, reason, title, status, "
               "source_kind, job_id FROM clips WHERE status=?")
        args: list = [status]
        if channel:
            sql += " AND channel=? AND source_kind='twitch'"
            args.append(channel)
        if job:
            sql += " AND job_id=?"
            args.append(job)
        sql += " ORDER BY created DESC LIMIT ?"
        rows = db.q(sql, args + [limit])
        if rows:
            ids = ",".join(str(r["id"]) for r in rows)
            posts = db.q(f"SELECT clip_id, platform, status FROM posts WHERE clip_id IN ({ids})")
            for r in rows:
                r["posts"] = [p for p in posts if p["clip_id"] == r["id"]]
        return rows

    @app.get("/api/clips/{cid}")
    def clip(cid: int):
        return get_clip(cid)

    @app.patch("/api/clips/{cid}")
    def edit_clip(cid: int, body: dict = Body(...)):
        get_clip(cid)
        for k in ("title", "caption"):
            if k in body:
                db.x(f"UPDATE clips SET {k}=? WHERE id=?", (str(body[k]), cid))
        return get_clip(cid)

    @app.post("/api/clips/{cid}/approve")
    def approve(cid: int, body: dict = Body(...)):
        c = get_clip(cid)
        plats = [p for p in body.get("platforms", []) if p in ("youtube", "instagram", "tiktok")]
        if "title" in body:
            db.x("UPDATE clips SET title=?, caption=? WHERE id=?",
                 (body["title"], body.get("caption", c["caption"]), cid))
        db.x("UPDATE clips SET status='approved', reviewed=? WHERE id=?", (time.time(), cid))
        for p in plats:
            db.x("""INSERT INTO posts(clip_id, platform, status, updated) VALUES(?,?,'queued',?)
                    ON CONFLICT(clip_id, platform) DO UPDATE SET status='queued', error=NULL, updated=excluded.updated
                    WHERE posts.status IN ('failed')""", (cid, p, time.time()))
        poster.wake.set()
        return get_clip(cid)

    @app.post("/api/clips/{cid}/reject")
    def reject(cid: int):
        get_clip(cid)
        db.x("UPDATE clips SET status='rejected', reviewed=? WHERE id=?", (time.time(), cid))
        return {"ok": True}

    @app.post("/api/clips/{cid}/restore")
    def restore(cid: int):
        get_clip(cid)
        db.x("UPDATE clips SET status='pending' WHERE id=?", (cid,))
        return {"ok": True}

    @app.delete("/api/clips/{cid}")
    def delete_clip(cid: int):
        c = get_clip(cid)
        for f in (c["file"], c["thumb"], c["src"]):
            if f:
                (clips_dir / f).unlink(missing_ok=True)
        db.x("DELETE FROM posts WHERE clip_id=?", (cid,))
        db.x("DELETE FROM clips WHERE id=?", (cid,))
        return {"ok": True}

    LAYOUTS = ("split", "auto", "fit", "crop", "original")

    def valid_box(b) -> dict | None:
        try:
            box = {k: min(max(float(b[k]), 0.0), 1.0) for k in ("x", "y", "w", "h")}
        except (TypeError, KeyError, ValueError):
            return None
        if box["w"] < 0.03 or box["h"] < 0.03:
            return None
        box["w"], box["h"] = min(box["w"], 1 - box["x"]), min(box["h"], 1 - box["y"])
        return box

    def do_render(c: dict, start: float, end: float | None, layout: str | None = None,
                  facecam: dict | None = None, redetect: bool = False) -> dict:
        """Re-render a clip from its raw slice (so a trim can always be loosened again)."""
        cid = c["id"]
        cfg = ClipConfig.from_settings(db.settings())
        cfg.layout = layout if layout in LAYOUTS else (c.get("layout") or cfg.layout)
        # facecam: explicit box > this clip's box > the channel's saved box > detect again
        cfg.facecam = None if redetect else (facecam or c.get("facecam") or c.get("channel_facecam"))
        new_name = f"{Path(c['file']).stem.split('.v')[0]}.v{int(time.time())}.mp4"
        used = render(clips_dir / c["src"], clips_dir / new_name, cfg, c["captions"], start, end)
        thumbnail(clips_dir / new_name, clips_dir / c["thumb"])
        old = c["file"]
        db.x("UPDATE clips SET file=?, duration=?, trim_start=?, trim_end=?, layout=?, facecam=? WHERE id=?",
             (new_name, duration_of(clips_dir / new_name), start, end, used,
              json.dumps(cfg.facecam) if cfg.facecam else None, cid))
        if old != new_name:
            (clips_dir / old).unlink(missing_ok=True)
        return get_clip(cid)

    @app.post("/api/clips/{cid}/render")
    def rerender(cid: int, body: dict = Body(...)):
        """Trim, change layout and/or set the facecam box, then re-render."""
        c = get_clip(cid)
        if not c["src"] or not (clips_dir / c["src"]).exists():
            raise HTTPException(400, "The raw footage for this clip is gone, so it can't be re-rendered")
        start = max(float(body.get("start", c.get("trim_start") or 0)), 0.0)
        end = body.get("end", c.get("trim_end"))
        end = float(end) if end not in (None, "") else None
        if end is not None and end - start < 3:
            raise HTTPException(400, "Clip would be shorter than 3 seconds")
        box = valid_box(body["facecam"]) if body.get("facecam") else None
        if body.get("facecam") and not box:
            raise HTTPException(400, "That facecam area is too small")
        if box and body.get("save_for_channel"):
            db.put(f"facecam:{c['channel']}", box)
        return do_render(c, start, end, body.get("layout"), box, bool(body.get("redetect")))

    @app.get("/api/clips/{cid}/frame")
    def frame(cid: int, t: float = -1):
        """A full-size still from the clip's raw footage, for drawing the facecam box."""
        c = get_clip(cid)
        if not c["src"] or not (clips_dir / c["src"]).exists():
            raise HTTPException(404, "The raw footage for this clip is gone")
        if t < 0:
            t = (c.get("trim_start") or 0) + min(3.0, (c.get("duration") or 6) / 2)
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(clips_dir / c["src"]),
                            "-frames:v", "1", "-vf", "scale=1280:-2", "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "-"],
                           capture_output=True)
        if not r.stdout:
            raise HTTPException(500, "Couldn't read a frame from this clip")
        return Response(r.stdout, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- bulk re-render (e.g. switch every clip waiting for review to a new layout)
    bulk = {"queue": [], "done": 0, "total": 0, "failed": 0}
    bulk_lock = threading.Lock()
    bulk_wake = threading.Event()

    def bulk_worker():
        while True:
            bulk_wake.wait()
            while True:
                with bulk_lock:
                    if not bulk["queue"]:
                        bulk_wake.clear()
                        break
                    cid, layout = bulk["queue"].pop(0)
                try:
                    c = db.clip(cid)
                    if c and c["src"] and (clips_dir / c["src"]).exists():
                        do_render(c, c.get("trim_start") or 0, c.get("trim_end"), layout)
                except Exception:
                    log.exception("bulk re-render of clip %s failed", cid)
                    bulk["failed"] += 1
                with bulk_lock:
                    bulk["done"] += 1

    threading.Thread(target=bulk_worker, daemon=True, name="rerender").start()

    @app.post("/api/clips/rerender")
    def rerender_many(body: dict = Body(...)):
        layout = body.get("layout")
        if layout not in LAYOUTS:
            raise HTTPException(400, "Unknown layout")
        status = body.get("status", "pending")
        ids = [r["id"] for r in db.q("SELECT id FROM clips WHERE status=? AND (layout IS NULL OR layout!=?) "
                                     "AND source_kind=? ORDER BY created DESC",
                                     (status, layout, body.get("source_kind", "twitch")))]
        with bulk_lock:
            if not bulk["queue"]:  # previous batch finished: start counting fresh
                bulk.update(done=0, total=0, failed=0)
            queued = {q[0] for q in bulk["queue"]}
            new = [(i, layout) for i in ids if i not in queued]
            bulk["queue"] += new
            bulk["total"] += len(new)
        bulk_wake.set()
        return {"queued": len(new)}

    @app.post("/api/clips/{cid}/suggest")
    def suggest(cid: int):
        """Title suggestions via the Claude Code CLI on your Claude subscription
        (claude setup-token -> CLAUDE_CODE_OAUTH_TOKEN). No API key, no API bill."""
        if not claude_available():
            raise HTTPException(400, "Claude Code isn't set up. Run `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN")
        c = get_clip(cid)
        prompt = (
            "You write titles for short-form clips (YouTube Shorts, Reels, TikTok) cut from Twitch streams.\n"
            f"Streamer: {c['channel']}\nWhat the streamer says in the clip: {c['transcript'] or '(no speech)'}\n"
            "Chat reacting at that moment:\n" + "\n".join(c["chat_sample"][-8:]) +
            "\n\nWrite 3 different title options: hooky, max 70 characters, same language as the "
            "streamer, no clickbait lies, no hashtags. Reply with JSON only: {\"titles\": [..3 strings..]}")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # never fall back to API billing
        with tempfile.TemporaryDirectory() as td:  # empty cwd: no project files or CLAUDE.md get loaded
            try:
                r = subprocess.run(
                    ["claude", "-p", prompt, "--output-format", "json", "--tools", "",
                     "--model", os.environ.get("CLAUDE_MODEL", "sonnet")],
                    capture_output=True, text=True, timeout=120, cwd=td, env=env)
            except subprocess.TimeoutExpired:
                raise HTTPException(504, "Claude Code took too long, try again")
        try:
            out = json.loads(r.stdout)
        except ValueError:
            out = {"is_error": True, "result": (r.stderr or r.stdout).strip()[:300]}
        if r.returncode != 0 or out.get("is_error"):
            raise HTTPException(502, f"Claude Code: {str(out.get('result') or r.stderr)[:300]}")
        m = re.search(r"\{.*\}", out.get("result", ""), re.S)
        try:
            titles = json.loads(m.group(0))["titles"] if m else []
        except (ValueError, KeyError):
            titles = []
        return {"titles": [t for t in titles if isinstance(t, str)][:3]}

    @app.post("/api/clips/{cid}/caption-template")
    def caption_from_template(cid: int, body: dict = Body(...)):
        c = get_clip(cid)
        return {"caption": render_caption(db.settings()["caption_template"], body.get("title", c["title"]),
                                          c["channel"], c.get("source_url"))}

    # ---------------------------------------------------------------- imports
    def import_opts(body: dict) -> dict:
        return {"length": body.get("length") if body.get("length") in ("short", "medium", "long") else "medium",
                "count": str(body.get("count", "auto")),
                "layout": body.get("layout") if body.get("layout") in ("split", "auto", "fit", "crop", "original") else None,
                "language": str(body.get("language") or "").strip()[:8] or None,
                "ai": bool(body.get("ai", True)), "chat": bool(body.get("chat", True))}

    @app.get("/api/imports")
    def list_imports():
        return db.q("SELECT * FROM jobs ORDER BY created DESC LIMIT 50")

    @app.post("/api/imports")
    def add_import(body: dict = Body(...)):
        url = str(body.get("url", "")).strip()
        if not re.match(r"^https?://\S+$", url):
            raise HTTPException(400, "Paste a full link, like https://www.youtube.com/watch?v=…")
        return {"id": importer.add_url(url, import_opts(body))}

    @app.post("/api/imports/upload")
    async def upload_import(file: UploadFile = File(...), options: str = Form("{}")):
        try:
            opts = import_opts(json.loads(options))
        except ValueError:
            opts = import_opts({})
        jid, dest = importer.add_upload(file.filename or "video.mp4", opts)
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 22):
                f.write(chunk)
        (dest.parent / "source.info.json").write_text(json.dumps(
            {"title": Path(file.filename or "Upload").stem, "channel": "Upload", "duration": duration_of(dest)}))
        importer.upload_done(jid)
        return {"id": jid}

    @app.post("/api/imports/{jid}/cancel")
    def cancel_import(jid: int):
        importer.cancel(jid)
        return {"ok": True}

    @app.post("/api/imports/{jid}/retry")
    def retry_import(jid: int):
        db.x("UPDATE jobs SET status='queued', step='Waiting', progress=0, error=NULL, updated=? "
             "WHERE id=? AND status IN ('failed','cancelled')", (time.time(), jid))
        importer.wake.set()
        return {"ok": True}

    @app.delete("/api/imports/{jid}")
    def delete_import(jid: int):
        importer.cancel(jid)
        shutil.rmtree(data / "imports" / str(jid), ignore_errors=True)
        db.x("DELETE FROM jobs WHERE id=?", (jid,))
        return {"ok": True}

    # ---------------------------------------------------------------- posts
    @app.post("/api/posts/{pid}/retry")
    def retry(pid: int):
        db.x("UPDATE posts SET status='queued', error=NULL, updated=? WHERE id=? AND status='failed'",
             (time.time(), pid))
        poster.wake.set()
        return {"ok": True}

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        return db.settings()

    @app.patch("/api/settings")
    def patch_settings(body: dict = Body(...)):
        s = db.update_settings(body)
        sup.sync()
        return s

    @app.put("/api/platforms/{name}/cred")
    def set_cred(name: str, body: dict = Body(...)):
        if name not in ("youtube", "instagram", "tiktok"):
            raise HTTPException(404)
        old = db.get(f"cred:{name}", {}) or {}
        db.put(f"cred:{name}", {"client_id": str(body.get("client_id", "")).strip(),
                                "client_secret": str(body.get("client_secret") or old.get("client_secret", "")).strip()})
        return {"ok": True}

    @app.post("/api/platforms/{name}/disconnect")
    def disconnect(name: str):
        platforms(db)[name].disconnect()
        return {"ok": True}

    @app.get("/api/platforms/{name}/redirect-uri")
    def redirect_uri(name: str, request: Request):
        return {"redirect_uri": platforms(db)[name].redirect_uri(base_url(request))}

    @app.get("/auth/{name}/start")
    def auth_start(name: str, request: Request):
        p = platforms(db).get(name)
        if not p or not p.info()["configured"]:
            raise HTTPException(400, "Save the app's client ID and secret first")
        state = new_state()
        db.put(f"oauth_state:{state}", {"platform": name, "t": time.time()})
        return RedirectResponse(p.auth_url(base_url(request), state))

    @app.get("/auth/{name}/callback")
    def auth_callback(name: str, request: Request, code: str = "", state: str = "", error: str = "",
                      error_description: str = ""):
        saved = db.get(f"oauth_state:{state}")
        db.delete(f"oauth_state:{state}")
        if error or not code:
            return RedirectResponse(f"/#settings?error={error_description or error or 'cancelled'}")
        if not saved or saved["platform"] != name or time.time() - saved["t"] > 900:
            return RedirectResponse("/#settings?error=login expired, try again")
        try:
            platforms(db)[name].exchange(base_url(request), code)
        except Exception as e:
            log.warning("oauth %s failed: %s", name, e)
            return RedirectResponse(f"/#settings?error={str(e)[:200]}")
        return RedirectResponse(f"/#settings?connected={name}")

    return app

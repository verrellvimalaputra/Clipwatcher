"""Auto-posting to YouTube Shorts, Instagram Reels and TikTok.

Each platform: OAuth connect (auth_url -> exchange), token refresh, publish().
App credentials (client id/secret) and tokens live in the DB kv table:
  cred:<platform>  = {"client_id": ..., "client_secret": ...}
  token:<platform> = {"access_token", "refresh_token", "expires_at", "account", ...}

Platform rules to know (they're enforced by the platforms, not by this code):
  * YouTube: Google Cloud projects that haven't passed the YouTube API audit may
    have uploads locked to private.
  * TikTok: unaudited apps can only Direct Post as SELF_ONLY to a private account.
    "inbox" mode instead sends the video to your TikTok inbox as a draft and you
    tap Post in the app, which works publicly without the audit.
  * Instagram: Meta downloads the video from a public HTTPS URL, so the studio
    must be reachable from the internet (PUBLIC_URL), e.g. on a VPS or through a
    Cloudflare tunnel.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from .db import DB

log = logging.getLogger("publish")
HTTP = httpx.Client(timeout=httpx.Timeout(30, read=300, write=300))
IG_VERSION = os.environ.get("IG_GRAPH_VERSION", "v25.0")


class PublishError(Exception):
    pass


def _check(r: httpx.Response) -> dict:
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:500]}
    if r.status_code >= 400:
        raise PublishError(f"HTTP {r.status_code}: {str(body)[:600]}")
    return body


class Platform:
    name = ""
    label = ""
    setup_url = ""

    def __init__(self, db: DB):
        self.db = db

    # creds / tokens
    @property
    def cred(self) -> dict:
        return self.db.get(f"cred:{self.name}", {}) or {}

    @property
    def token(self) -> dict | None:
        return self.db.get(f"token:{self.name}")

    def save_token(self, tok: dict):
        self.db.put(f"token:{self.name}", tok)

    def disconnect(self):
        self.db.delete(f"token:{self.name}")

    def info(self) -> dict:
        t = self.token or {}
        return {"name": self.name, "label": self.label, "setup_url": self.setup_url,
                "configured": bool(self.cred.get("client_id") and self.cred.get("client_secret")),
                "connected": bool(t.get("access_token")), "account": t.get("account"),
                "client_id": self.cred.get("client_id", "")}

    def redirect_uri(self, base: str) -> str:
        return f"{base.rstrip('/')}/auth/{self.name}/callback"

    # implemented per platform
    def auth_url(self, base: str, state: str) -> str: ...
    def exchange(self, base: str, code: str) -> None: ...
    def refresh(self) -> None: ...
    def publish(self, video: Path, clip: dict, settings: dict, public_video_url: str | None) -> dict: ...

    def access_token(self) -> str:
        t = self.token
        if not t:
            raise PublishError(f"{self.label} is not connected")
        if t.get("expires_at") and t["expires_at"] - time.time() < 300:
            self.refresh()
            t = self.token
        return t["access_token"]


# --------------------------------------------------------------------- YouTube
class YouTube(Platform):
    name, label = "youtube", "YouTube Shorts"
    setup_url = "https://console.cloud.google.com/apis/library/youtube.googleapis.com"
    SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"

    def auth_url(self, base, state):
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "client_id": self.cred["client_id"], "redirect_uri": self.redirect_uri(base),
            "response_type": "code", "scope": self.SCOPES, "access_type": "offline",
            "prompt": "consent", "state": state})

    def _token_req(self, data):
        body = _check(HTTP.post("https://oauth2.googleapis.com/token", data={
            "client_id": self.cred["client_id"], "client_secret": self.cred["client_secret"], **data}))
        old = self.token or {}
        return {**old, "access_token": body["access_token"],
                "refresh_token": body.get("refresh_token", old.get("refresh_token")),
                "expires_at": time.time() + body.get("expires_in", 3600)}

    def exchange(self, base, code):
        tok = self._token_req({"code": code, "grant_type": "authorization_code",
                               "redirect_uri": self.redirect_uri(base)})
        me = _check(HTTP.get("https://www.googleapis.com/youtube/v3/channels",
                             params={"part": "snippet", "mine": "true"},
                             headers={"Authorization": f"Bearer {tok['access_token']}"}))
        items = me.get("items") or []
        tok["account"] = items[0]["snippet"]["title"] if items else "YouTube account"
        self.save_token(tok)

    def refresh(self):
        rt = (self.token or {}).get("refresh_token")
        if not rt:
            raise PublishError("YouTube token expired. Reconnect in Settings")
        self.save_token(self._token_req({"refresh_token": rt, "grant_type": "refresh_token"}))

    def publish(self, video, clip, settings, public_video_url):
        tok = self.access_token()
        title = clip["title"].strip()[:100]
        if settings["layout"] in ("crop", "fit") and "#shorts" not in title.lower() and len(title) <= 92:
            title += " #Shorts"
        meta = {"snippet": {"title": title, "description": clip["caption"][:5000], "categoryId": "20",
                            "tags": [clip["channel"], "twitch", "shorts"]},
                "status": {"privacyStatus": settings["youtube_privacy"], "selfDeclaredMadeForKids": False}}
        size = video.stat().st_size
        r = HTTP.post("https://www.googleapis.com/upload/youtube/v3/videos",
                      params={"uploadType": "resumable", "part": "snippet,status"}, json=meta,
                      headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/mp4",
                               "X-Upload-Content-Length": str(size)})
        _check(r)
        upload_url = r.headers["Location"]
        body = _check(HTTP.put(upload_url, content=video.read_bytes(),
                               headers={"Content-Type": "video/mp4", "Content-Length": str(size)}))
        vid = body["id"]
        note = None
        if body.get("status", {}).get("privacyStatus") != settings["youtube_privacy"]:
            note = "YouTube set it to private, probably because your Google Cloud project hasn't passed the API audit"
        return {"remote_id": vid, "url": f"https://youtube.com/shorts/{vid}", "note": note}


# ------------------------------------------------------------------- Instagram
class Instagram(Platform):
    name, label = "instagram", "Instagram Reels"
    setup_url = "https://developers.facebook.com/apps/"
    G = f"https://graph.instagram.com/{IG_VERSION}"

    def auth_url(self, base, state):
        return "https://www.instagram.com/oauth/authorize?" + urlencode({
            "client_id": self.cred["client_id"], "redirect_uri": self.redirect_uri(base),
            "response_type": "code", "state": state,
            "scope": "instagram_business_basic,instagram_business_content_publish"})

    def exchange(self, base, code):
        short = _check(HTTP.post("https://api.instagram.com/oauth/access_token", data={
            "client_id": self.cred["client_id"], "client_secret": self.cred["client_secret"],
            "grant_type": "authorization_code", "redirect_uri": self.redirect_uri(base), "code": code}))
        short = (short.get("data") or [short])[0]
        long = _check(HTTP.get("https://graph.instagram.com/access_token", params={
            "grant_type": "ig_exchange_token", "client_secret": self.cred["client_secret"],
            "access_token": short["access_token"]}))
        me = _check(HTTP.get(f"{self.G}/me", params={"fields": "user_id,username",
                                                     "access_token": long["access_token"]}))
        self.save_token({"access_token": long["access_token"],
                         "expires_at": time.time() + long.get("expires_in", 5_184_000),
                         "ig_id": str(me.get("user_id") or me.get("id")), "account": "@" + me.get("username", "")})

    def access_token(self) -> str:
        t = self.token
        if not t:
            raise PublishError("Instagram is not connected")
        # long-lived tokens last 60 days and can be refreshed once they're older than 24 h
        if t.get("expires_at") and t["expires_at"] - time.time() < 10 * 86400:
            self.refresh()
            t = self.token
        return t["access_token"]

    def refresh(self):
        t = self.token or {}
        body = _check(HTTP.get("https://graph.instagram.com/refresh_access_token", params={
            "grant_type": "ig_refresh_token", "access_token": t.get("access_token", "")}))
        self.save_token({**t, "access_token": body["access_token"],
                         "expires_at": time.time() + body.get("expires_in", 5_184_000)})

    def publish(self, video, clip, settings, public_video_url):
        if not public_video_url:
            raise PublishError("Instagram needs to download the video from a public HTTPS URL. "
                               "Set Public URL in Settings (see README: VPS or Cloudflare tunnel)")
        tok, ig = self.access_token(), self.token["ig_id"]
        c = _check(HTTP.post(f"{self.G}/{ig}/media", data={
            "media_type": "REELS", "video_url": public_video_url, "caption": clip["caption"][:2200],
            "share_to_feed": "true", "access_token": tok}))
        cid = c["id"]
        for _ in range(120):  # up to ~10 min of processing
            st = _check(HTTP.get(f"{self.G}/{cid}", params={"fields": "status_code,status", "access_token": tok}))
            if st.get("status_code") == "FINISHED":
                break
            if st.get("status_code") in ("ERROR", "EXPIRED"):
                raise PublishError(f"Instagram processing failed: {st.get('status')}")
            time.sleep(5)
        else:
            raise PublishError("Instagram processing timed out")
        pub = _check(HTTP.post(f"{self.G}/{ig}/media_publish", data={"creation_id": cid, "access_token": tok}))
        mid = pub["id"]
        link = _check(HTTP.get(f"{self.G}/{mid}", params={"fields": "permalink", "access_token": tok}))
        return {"remote_id": mid, "url": link.get("permalink"), "note": None}


# ---------------------------------------------------------------------- TikTok
class TikTok(Platform):
    name, label = "tiktok", "TikTok"
    setup_url = "https://developers.tiktok.com/apps/"
    API = "https://open.tiktokapis.com/v2"

    def auth_url(self, base, state):
        scopes = "user.info.basic,video.upload"
        if self.db.settings()["tiktok_mode"] == "direct":
            scopes += ",video.publish"
        return "https://www.tiktok.com/v2/auth/authorize/?" + urlencode({
            "client_key": self.cred["client_id"], "redirect_uri": self.redirect_uri(base),
            "response_type": "code", "scope": scopes, "state": state})

    def _token_req(self, data):
        body = _check(HTTP.post(f"{self.API}/oauth/token/", data={
            "client_key": self.cred["client_id"], "client_secret": self.cred["client_secret"], **data},
            headers={"Content-Type": "application/x-www-form-urlencoded"}))
        if "access_token" not in body:
            raise PublishError(f"TikTok token error: {body}")
        old = self.token or {}
        return {**old, "access_token": body["access_token"], "refresh_token": body["refresh_token"],
                "expires_at": time.time() + body.get("expires_in", 86400), "scope": body.get("scope")}

    def exchange(self, base, code):
        tok = self._token_req({"code": code, "grant_type": "authorization_code",
                               "redirect_uri": self.redirect_uri(base)})
        me = HTTP.get(f"{self.API}/user/info/", params={"fields": "display_name"},
                      headers={"Authorization": f"Bearer {tok['access_token']}"}).json()
        tok["account"] = (me.get("data", {}).get("user", {}) or {}).get("display_name") or "TikTok account"
        self.save_token(tok)

    def refresh(self):
        rt = (self.token or {}).get("refresh_token")
        if not rt:
            raise PublishError("TikTok token expired. Reconnect in Settings")
        self.save_token(self._token_req({"grant_type": "refresh_token", "refresh_token": rt}))

    def _api(self, path, tok, json):
        body = _check(HTTP.post(f"{self.API}{path}", json=json, headers={
            "Authorization": f"Bearer {tok}", "Content-Type": "application/json; charset=UTF-8"}))
        err = body.get("error", {})
        if err.get("code") not in (None, "ok"):
            raise PublishError(f"TikTok: {err.get('code')}: {err.get('message')}")
        return body.get("data", {})

    def publish(self, video, clip, settings, public_video_url):
        tok = self.access_token()
        size = video.stat().st_size
        # TikTok chunk rules: whole file if <= 64 MB, else 10 MB chunks with the remainder in the last one
        chunk = size if size <= 64_000_000 else 10_000_000
        count = max(size // chunk, 1)
        source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}

        direct = settings["tiktok_mode"] == "direct"
        if direct:
            opts = self._api("/post/publish/creator_info/query/", tok, {}).get("privacy_level_options", [])
            privacy = settings["tiktok_privacy"] if settings["tiktok_privacy"] in opts else "SELF_ONLY"
            data = self._api("/post/publish/video/init/", tok, {
                "post_info": {"title": clip["caption"][:2200], "privacy_level": privacy,
                              "disable_comment": False, "disable_duet": False, "disable_stitch": False},
                "source_info": source})
        else:
            data = self._api("/post/publish/inbox/video/init/", tok, {"source_info": source})

        pid, url = data["publish_id"], data["upload_url"]
        with open(video, "rb") as f:
            for i in range(count):
                start = i * chunk
                end = size - 1 if i == count - 1 else start + chunk - 1
                f.seek(start)
                part = f.read(end - start + 1)
                r = HTTP.put(url, content=part, headers={
                    "Content-Type": "video/mp4", "Content-Length": str(len(part)),
                    "Content-Range": f"bytes {start}-{end}/{size}"})
                if r.status_code >= 400:
                    raise PublishError(f"TikTok chunk upload failed: HTTP {r.status_code} {r.text[:300]}")

        for _ in range(120):
            st = self._api("/post/publish/status/fetch/", tok, {"publish_id": pid})
            s = st.get("status")
            if s == "SEND_TO_USER_INBOX":
                return {"remote_id": pid, "url": None,
                        "note": "Sent to your TikTok inbox. Open the TikTok app to add a sound and post it"}
            if s == "PUBLISH_COMPLETE":
                ids = st.get("publicaly_available_post_id") or []
                url = f"https://www.tiktok.com/@/video/{ids[0]}" if ids else None
                note = None if ids else "Posted (probably private: unaudited TikTok apps can only post SELF_ONLY)"
                return {"remote_id": pid, "url": url, "note": note}
            if s == "FAILED":
                raise PublishError(f"TikTok processing failed: {st.get('fail_reason')}")
            time.sleep(5)
        raise PublishError("TikTok processing timed out")


# ----------------------------------------------------------------- demo/fake
class Fake(Platform):
    """CLIPWATCHER_DEMO=1: pretend to post, so you can try the whole flow safely."""

    def __init__(self, db, real: Platform):
        super().__init__(db)
        self.name, self.label, self.setup_url = real.name, real.label, real.setup_url

    def info(self):
        return {**super().info(), "configured": True, "connected": True, "account": "demo account", "demo": True}

    def publish(self, video, clip, settings, public_video_url):
        time.sleep(3)
        return {"remote_id": f"demo-{clip['id']}", "url": None, "note": "Demo mode: nothing was actually posted"}


def platforms(db: DB) -> dict[str, Platform]:
    real = {p.name: p for p in (YouTube(db), Instagram(db), TikTok(db))}
    if os.environ.get("CLIPWATCHER_DEMO") == "1":
        return {n: Fake(db, p) for n, p in real.items()}
    return real


# ---------------------------------------------------------- public video links
def public_token(secret: str, filename: str) -> str:
    return hmac.new(secret.encode(), filename.encode(), hashlib.sha256).hexdigest()[:32]


def new_state() -> str:
    return secrets.token_urlsafe(24)


# ---------------------------------------------------------------- post queue
class PostWorker(threading.Thread):
    """Processes queued posts one at a time, so a slow upload never blocks the UI."""

    def __init__(self, db: DB, clips_dir: Path, public_base: callable, secret: str):
        super().__init__(daemon=True, name="poster")
        self.db, self.clips_dir, self.public_base, self.secret = db, clips_dir, public_base, secret
        self.wake = threading.Event()
        # anything left "uploading" from a crash goes back in the queue
        db.x("UPDATE posts SET status='queued' WHERE status='uploading'")

    def run(self):
        while True:
            self.wake.wait(3)
            self.wake.clear()
            while True:
                rows = self.db.q("SELECT * FROM posts WHERE status='queued' ORDER BY updated LIMIT 1")
                if not rows:
                    break
                self._do(rows[0])

    def _do(self, post: dict):
        db = self.db
        db.x("UPDATE posts SET status='uploading', error=NULL, updated=? WHERE id=?", (time.time(), post["id"]))
        clip = db.clip(post["clip_id"])
        try:
            plat = platforms(db)[post["platform"]]
            video = self.clips_dir / clip["file"]
            base = self.public_base()
            pub_url = f"{base}/public/{public_token(self.secret, clip['file'])}/{clip['file']}" if base else None
            log.info("posting clip %s to %s", clip["id"], plat.label)
            res = plat.publish(video, clip, db.settings(), pub_url)
            db.x("UPDATE posts SET status='posted', remote_id=?, url=?, note=?, updated=? WHERE id=?",
                 (res.get("remote_id"), res.get("url"), res.get("note"), time.time(), post["id"]))
            log.info("posted clip %s to %s: %s", clip["id"], plat.label, res.get("url"))
        except Exception as e:
            log.warning("posting clip %s to %s failed: %s", clip["id"], post["platform"], e)
            db.x("UPDATE posts SET status='failed', error=?, updated=? WHERE id=?",
                 (str(e)[:1000], time.time(), post["id"]))

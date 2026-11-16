"""Two brands (e.g. a German and an Indonesian clip page), each with its own accounts."""
import time
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from clipwatcher import publishers, web
from clipwatcher.db import DB


def app(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPWATCHER_DEMO", raising=False)
    monkeypatch.delenv("STUDIO_PASSWORD", raising=False)
    # don't start recorders/posters: only the API is under test
    monkeypatch.setattr(web.Supervisor, "sync", lambda self: None)
    monkeypatch.setattr(web.Supervisor, "start_housekeeping", lambda self: None)
    return TestClient(web.create_app(tmp_path))


def test_old_database_gets_brand_one(tmp_path):
    db = DB(tmp_path / "s.db")
    assert [b["name"] for b in db.brands()] == ["Main"]
    db.put("token:youtube", {"access_token": "old", "account": "My channel"})
    yt = publishers.platforms(db, 1)["youtube"]
    assert yt.token["account"] == "My channel"                       # existing login keeps working
    assert publishers.platforms(db, 2)["youtube"].token is None      # brand 2 has its own slot
    publishers.platforms(db, 2)["youtube"].save_token({"access_token": "new"})
    assert db.get("token:youtube@2")["access_token"] == "new" and yt.token["access_token"] == "old"


def test_clips_follow_the_streamers_brand(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch)
    with c:
        b = c.post("/api/brands", json={"name": "GTA 6 Clips ID", "language": "id",
                                         "caption_template": "{title} 🔥 #gta6"}).json()
        assert b["id"] == 2
        assert c.post("/api/brands", json={"name": "X", "handle": "klip geming!"}).json()["handle"] == "@klipgeming"
        c.delete("/api/brands/3")
        c.post("/api/channels", json={"name": "papaplatte"})
        c.post("/api/channels", json={"name": "windahbasudara", "brand": 2})
        db = DB(tmp_path / "studio.db")
        assert db.channel("windahbasudara")["brand"] == 2 and db.channel("papaplatte")["brand"] == 1
        assert db.clip_config("windahbasudara").language == "id"     # brand language for Whisper

        sup = web.Supervisor(tmp_path, db)
        for name in ("papaplatte", "windahbasudara"):
            sup._on_clip({"channel": name, "created": time.time(), "file": f"{name}.mp4", "src": "x.ts",
                          "thumb": "x.jpg", "duration": 20, "score": 9, "baseline": 2, "reason": "volume",
                          "chat_sample": [], "captions": [], "transcript": "", "title": "Crash"})
        clips = {r["channel"]: r for r in c.get("/api/clips").json()}
        assert clips["papaplatte"]["brand"] == 1 and clips["windahbasudara"]["brand"] == 2
        assert db.clip(clips["windahbasudara"]["id"])["caption"] == "Crash 🔥 #gta6"
        assert [r["channel"] for r in c.get("/api/clips?brand=2").json()] == ["windahbasudara"]

        # posting to a platform the brand hasn't connected is refused with a clear message
        r = c.post(f"/api/clips/{clips['windahbasudara']['id']}/approve", json={"platforms": ["youtube"]})
        assert r.status_code == 400 and "GTA 6 Clips ID" in r.json()["detail"]
        db.put("token:youtube@2", {"access_token": "t", "account": "GTA6 ID"})
        r = c.post(f"/api/clips/{clips['windahbasudara']['id']}/approve", json={"platforms": ["youtube"]})
        assert r.status_code == 200

        # a brand in use can't be deleted; once empty it can, and its logins go with it
        assert c.delete("/api/brands/2").status_code == 400
        c.patch("/api/channels/windahbasudara", json={"brand": 1})
        assert c.delete("/api/brands/2").status_code == 200
        assert db.get("token:youtube@2") is None
        assert c.delete("/api/brands/1").status_code == 400


def test_login_is_saved_to_the_brand_that_started_it(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch)
    with c:
        c.post("/api/brands", json={"name": "DE"})
        c.put("/api/platforms/youtube/cred", json={"client_id": "cid", "client_secret": "sec"})
        r = c.get("/auth/youtube/start?brand=2", follow_redirects=False)
        q = parse_qs(urlparse(r.headers["location"]).query)
        assert "select_account" in q["prompt"][0]           # lets you pick the second account
        saved = []
        monkeypatch.setattr(publishers.YouTube, "exchange", lambda self, base, code: saved.append(self.brand))
        c.get(f"/auth/youtube/callback?code=abc&state={q['state'][0]}", follow_redirects=False)
        assert saved == [2]


def test_overlay_credit_and_handle(tmp_path):
    db = DB(tmp_path / "s.db")
    db.add_channel("papaplatte")
    db.add_channel("yt-windah", "youtube", "https://youtube.com/@windah/live", "Windah Basudara", 1)
    db.x("UPDATE brands SET handle='streamweltclips' WHERE id=1")
    assert db.overlay("papaplatte", 1) == ("twitch.tv/papaplatte", "@streamweltclips")
    assert db.overlay("yt-windah", 1)[0] == "YouTube: Windah Basudara"
    assert db.overlay("", 1, "MrBeast")[0] == "YouTube: MrBeast"
    cfg = db.clip_config("papaplatte")
    assert (cfg.credit, cfg.handle) == ("twitch.tv/papaplatte", "@streamweltclips")
    db.update_settings({"credit_overlay": False})
    assert db.overlay("papaplatte", 1) == (None, None)

import json

import pytest

from clipwatcher import reviewer
from clipwatcher.db import DB


def clip_row(db, **kw):
    c = {"channel": "papaplatte", "created": 1.0, "file": "a.mp4", "src": "a.src.ts", "thumb": "a.jpg",
         "duration": 30.0, "score": 20, "baseline": 4, "reason": "volume", "transcript": "", "title": "Er fällt",
         "caption": "", "trim_start": 10.0, "trim_end": 40.0, "context": "Papaplatte will über die Brücke springen",
         "captions": [[5.0, 9.0, "vorher"], [11.0, 14.0, "Ich spring jetzt"], [38.0, 45.0, "Nein!"]],
         "chat_sample": ["KEKW", "LMAO"], **kw}
    return db.add_clip(c)


def test_input_only_has_the_final_cut(tmp_path):
    db = DB(tmp_path / "s.db")
    c = db.clip(clip_row(db))
    text = reviewer.review_input(c)
    assert "vorher" not in text                     # before the trim
    assert "[  1.0] Ich spring jetzt" in text      # times relative to the cut
    assert "[ 28.0] Nein!" in text                 # overlaps the end
    assert "Papaplatte will über die Brücke" in text and "- KEKW" in text
    # live clips store captions with per-word timings as a 4th item
    c["captions"] = [[c0[0], c0[1], c0[2], [[c0[0], c0[1], "w"]]] for c0 in c["captions"]]
    assert reviewer.review_input(c) == text


def test_parse_clamps_and_requires_score():
    r = reviewer.parse_review('Sure:\n{"interesting": 8, "funny": 12, "clear": "6.5", "score": 7.25, "verdict": "ok", "fix": ""}')
    assert r == {"interesting": 8.0, "funny": 10.0, "clear": 6.5, "score": 7.2, "verdict": "ok", "fix": "", "saw_frames": False}
    with pytest.raises(ValueError):
        reviewer.parse_review('{"verdict": "no number"}')
    with pytest.raises(ValueError):
        reviewer.parse_review("I can't rate this")


def test_worker_scores_pending_and_handles_failures(tmp_path, monkeypatch):
    db = DB(tmp_path / "s.db")
    a = clip_row(db)
    b = clip_row(db, created=2.0)
    db.x("UPDATE clips SET status='approved' WHERE id=?", (b,))
    w = reviewer.ReviewWorker(db)

    monkeypatch.setattr(reviewer.moments, "run_claude", lambda *a, **k: '{"score": 8, "funny": 9, "verdict": "great fail"}')
    c = w.next_clip()
    assert c["id"] == a                             # approved clips are skipped
    w.review_one(c)
    row = db.clip(a)
    assert row["ai_score"] == 8.0 and row["ai_review"]["verdict"] == "great fail"
    assert w.next_clip() is None

    # unusable answers count against the clip; Claude being down doesn't
    db.x("UPDATE clips SET ai_score=NULL, ai_review=NULL WHERE id=?", (a,))
    monkeypatch.setattr(reviewer.moments, "run_claude", lambda *a, **k: "no idea")
    w.review_one(w.next_clip()); w.review_one(w.next_clip())
    assert w.next_clip() is None                   # gave up after MAX_TRIES

    db.x("UPDATE clips SET ai_tries=0 WHERE id=?", (a,))

    def down(*a, **k):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(reviewer.moments, "run_claude", down)
    w.review_one(w.next_clip())
    assert db.clip(a)["ai_tries"] == 0 and w.paused_until > 0


def test_retrim_during_review_is_not_stored(tmp_path, monkeypatch):
    db = DB(tmp_path / "s.db")
    a = clip_row(db)
    w = reviewer.ReviewWorker(db)

    def slow(*args, **k):
        db.x("UPDATE clips SET trim_end=35 WHERE id=?", (a,))  # user trims while Claude thinks
        return json.dumps({"score": 3})
    monkeypatch.setattr(reviewer.moments, "run_claude", slow)
    w.review_one(w.next_clip())
    assert db.clip(a)["ai_score"] is None


def fake_claude(tmp_path, monkeypatch, result: str, usage=None):
    """A stand-in `claude` that records its argv and the files in its cwd."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "calls.jsonl"
    out = json.dumps({"type": "result", "is_error": False, "result": result, "total_cost_usd": 0.002,
                      "usage": usage or {"input_tokens": 1200, "cache_creation_input_tokens": 300,
                                         "cache_read_input_tokens": 5000, "output_tokens": 90},
                      "modelUsage": {"claude-haiku-x": {}}})
    script = bindir / "claude"
    script.write_text("#!/usr/bin/env python3\nimport json, os, sys\nsys.stdin.read()\n"
                      f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'files': sorted(os.listdir('.'))}}) + '\\n')\n"
                      f"print({out!r})\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    return log


def test_run_claude_only_allows_reading_the_given_files(tmp_path, monkeypatch):
    from clipwatcher import moments
    log = fake_claude(tmp_path, monkeypatch, "OK")
    img = tmp_path / "x.jpg"
    img.write_bytes(b"jpg")
    moments.run_claude("hi", "", task="review", model="haiku", files={"frames.jpg": img})
    moments.run_claude("hi", "", task="cut")
    with_files, plain = [json.loads(l) for l in log.read_text().splitlines()]
    a = with_files["argv"]
    assert a[a.index("--model") + 1] == "haiku"
    assert a[a.index("--tools") + 1] == "Read" and a[a.index("--allowedTools") + 1] == "Read(./frames.jpg)"
    assert with_files["files"] == ["frames.jpg"]
    b = plain["argv"]
    assert b[b.index("--tools") + 1] == "" and "--allowedTools" not in b and plain["files"] == []


def test_usage_is_counted_per_task(tmp_path, monkeypatch):
    from clipwatcher import moments
    db = DB(tmp_path / "s.db")
    fake_claude(tmp_path, monkeypatch, "OK")
    monkeypatch.setattr(moments, "USAGE_HOOK", db.add_usage)
    moments.run_claude("hi", "", task="review")
    moments.run_claude("hi", "", task="review")
    moments.run_claude("hi", "", task="cut")
    rows = {r["task"]: r for r in db.usage()}
    assert rows["review"]["calls"] == 2 and rows["review"]["input"] == 3000   # fresh input incl. cache writes
    assert rows["review"]["cached"] == 10000 and rows["review"]["output"] == 180
    assert rows["cut"]["calls"] == 1 and rows["review"]["model"] == "claude-haiku-x"


def test_review_sends_a_contact_sheet(tmp_path, monkeypatch):
    import subprocess
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30:duration=12",
                    "-pix_fmt", "yuv420p", str(clips / "a.mp4")], check=True)
    db = DB(tmp_path / "s.db")
    cid = clip_row(db, duration=12.0)
    log = fake_claude(tmp_path, monkeypatch, '{"score": 6, "verdict": "ok", "saw_frames": true}')
    r = reviewer.review_clip(db.clip(cid), clips)
    call = json.loads(log.read_text().splitlines()[0])
    assert call["files"] == ["frames.jpg"] and r["saw_frames"] is True and r["model"] == "haiku"
    # no video file: still reviewed, text only, and says so
    (clips / "a.mp4").unlink()
    r = reviewer.review_clip(db.clip(cid), clips)
    assert r["saw_frames"] is False and json.loads(log.read_text().splitlines()[1])["files"] == []

"""Video import: YouTube (or any yt-dlp site) link or an uploaded file ->
download -> transcribe -> find moments -> render post-ready clips into the
review queue. One job at a time, in the background."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .db import DB
from .editing import ClipConfig, duration_of, lines_from_words, render, thumbnail, transcribe_words
from .moments import (LENGTHS, Moment, chat_peaks, find_moments, heat_at, heatmap_peaks, loud_peaks, loudness_curve,
                      youtube_chat)
from .supervisor import render_caption

log = logging.getLogger("importer")
MARGIN = 3.0  # seconds of extra footage kept around each clip, so trims can widen it
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}


class Cancelled(Exception):
    pass


class ImportWorker(threading.Thread):
    def __init__(self, db: DB, data: Path):
        super().__init__(daemon=True, name="importer")
        self.db, self.data = db, data
        self.imports = data / "imports"
        self.clips = data / "clips"
        self.imports.mkdir(parents=True, exist_ok=True)
        self.wake = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.current: int | None = None
        # jobs interrupted by a restart start over
        db.x("UPDATE jobs SET status='queued', step='Waiting', progress=0 "
             "WHERE status IN ('downloading','transcribing','finding','rendering')")

    # ------------------------------------------------------------ public
    def add_url(self, url: str, options: dict) -> int:
        jid = self.db.x("INSERT INTO jobs(kind, url, title, status, step, options, created, updated) "
                        "VALUES('url', ?, ?, 'queued', 'Waiting', ?, ?, ?)",
                        (url, url, json.dumps(options), time.time(), time.time()))
        self.wake.set()
        return jid

    def add_upload(self, filename: str, options: dict) -> tuple[int, Path]:
        jid = self.db.x("INSERT INTO jobs(kind, title, channel, status, step, options, created, updated) "
                        "VALUES('upload', ?, 'Upload', 'uploading', 'Uploading', ?, ?, ?)",
                        (Path(filename).stem, json.dumps(options), time.time(), time.time()))
        d = self.imports / str(jid)
        d.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix.lower() if Path(filename).suffix.lower() in VIDEO_EXT else ".mp4"
        return jid, d / f"source{ext}"

    def upload_done(self, jid: int):
        self._set(jid, status="queued", step="Waiting")
        self.wake.set()

    def cancel(self, jid: int):
        self.db.x("UPDATE jobs SET status='cancelled', step='Cancelled', updated=? WHERE id=? AND status NOT IN "
                  "('done','failed')", (time.time(), jid))
        if self.current == jid and self.proc:
            self.proc.kill()

    # ------------------------------------------------------------ loop
    def run(self):
        while True:
            self.wake.wait(5)
            self.wake.clear()
            while True:
                rows = self.db.q("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1")
                if not rows:
                    break
                self._run_job(rows[0])

    def _set(self, jid: int, **kv):
        kv["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in kv)
        self.db.x(f"UPDATE jobs SET {cols} WHERE id=?", (*kv.values(), jid))

    def _check(self, jid: int):
        if self.db.q("SELECT status FROM jobs WHERE id=?", (jid,))[0]["status"] == "cancelled":
            raise Cancelled()

    def _run_job(self, job: dict):
        jid = job["id"]
        self.current = jid
        opts = json.loads(job["options"] or "{}")
        work = self.imports / str(jid)
        work.mkdir(parents=True, exist_ok=True)
        try:
            info = self._fetch(job, work, opts)
            self._check(jid)
            media = next(p for p in work.iterdir() if p.stem == "source" and p.suffix.lower() in VIDEO_EXT)
            dur = info.get("duration") or duration_of(media)
            info["duration"] = dur
            self._set(jid, duration=dur, title=info.get("title") or job["title"], channel=info.get("channel"))

            # 1. transcribe (the slow part on CPU)
            s = self.db.settings()
            cfg = ClipConfig.from_settings(s)
            cfg.language = opts.get("language") or cfg.language
            # Finding moments only needs a rough transcript of the WHOLE video. YouTube's
            # own captions give one in seconds; otherwise a fast Whisper model. The
            # accurate captions are made later, per clip (see _clip_captions).
            cache = work / "transcript.json"
            if cache.exists():
                t = json.loads(cache.read_text())
                words, segments = t["words"], t["segments"]
                t_source = t.get("source") or f"whisper:{cfg.whisper_model}"  # older caches: configured model
            else:
                words, segments, t_source = [], [], ""
                if job["kind"] == "url":
                    self._set(jid, status="transcribing", step="Reading YouTube's captions", progress=0)
                    words, segments = self._youtube_captions(job, work, opts, info)
                    t_source = "youtube" if words else ""
                if not words:
                    fast = ClipConfig(**{**cfg.__dict__, "whisper_model": FAST_MODEL
                                         if cfg.whisper_model not in ("tiny", "base") else cfg.whisper_model})
                    self._set(jid, status="transcribing", step="Transcribing", progress=0)
                    t_start, last = time.time(), [0.0]

                    def prog(p):
                        if p - last[0] >= 0.01:
                            last[0] = p
                            spent = time.time() - t_start
                            left = spent / p * (1 - p) if p > 0.03 else None
                            step = "Transcribing" + (f" · about {_human(left)} left" if left else "")
                            self._set(jid, progress=round(p, 3), step=step)
                            self._check(jid)
                    words, segments = transcribe_words(media, fast, prog, dur)
                    t_source = f"whisper:{fast.whisper_model}"
                cache.write_text(json.dumps({"words": words, "segments": segments, "source": t_source}))
            if not segments:
                raise RuntimeError("No speech found in this video, so there's nothing to cut on")
            self._check(jid)

            # 2. signals + moments
            self._set(jid, status="finding", step="Finding the best moments", progress=0)
            loud = loudness_curve(media)
            chat = []
            for f in work.glob("*.live_chat.json*"):
                chat = chat_peaks(youtube_chat(f))
            signals = {"heatmap": heatmap_peaks(info.get("heatmap")), "chat": chat, "loud": loud_peaks(loud)}
            length = opts.get("length") if opts.get("length") in LENGTHS else "medium"
            mcache = work / "moments.json"
            if mcache.exists():  # restarted mid-job: keep the same picks, so finished clips aren't redone
                mc = json.loads(mcache.read_text())
                moments, finder = [Moment(**m) for m in mc["moments"]], mc["finder"]
            else:
                moments, finder = find_moments(info, words, segments, signals, loud, length,
                                               opts.get("count", "auto"), use_claude=opts.get("ai", True))
                mcache.write_text(json.dumps({"moments": [m.to_dict() for m in moments], "finder": finder}))
            self._set(jid, finder=finder)
            if not moments:
                raise RuntimeError("Couldn't find any moments that fit the chosen clip length")
            self._check(jid)

            # 3. render
            self._set(jid, status="rendering", step=f"Editing clip 1 of {len(moments)}", progress=0)
            layout = opts.get("layout") or s["import_layout"]
            done_files = {r["file"].split(".v")[0].removesuffix(".mp4") for r in
                          self.db.q("SELECT file FROM clips WHERE job_id=?", (jid,))}
            made = 0
            for i, m in enumerate(moments):
                self._check(jid)
                if f"job{jid}_{i + 1:02}" in done_files:  # already made before a restart
                    made += 1
                    continue
                n = len(moments)
                self._set(jid, step=f"Editing clip {i + 1} of {n}", progress=i / n)
                last = [0.0]

                def clip_prog(p, i=i, n=n, what="Editing"):
                    if p - last[0] >= 0.05 or p >= 1:
                        last[0] = p
                        self._set(jid, step=f"{what} clip {i + 1} of {n} · {int(p * 100)}%", progress=round((i + p) / n, 3))
                try:
                    self._make_clip(job, info, media, words, chat, loud, m, cfg, layout, i, t_source, clip_prog)
                    made += 1
                    self._set(jid, clips=made)
                except Cancelled:
                    raise
                except Exception:
                    log.exception("clip %s of job %s failed", i, jid)
            self._set(jid, status="done", step=f"{made} clips ready to review", progress=1)
            self._cleanup(work, media)
        except Cancelled:
            self._set(jid, status="cancelled", step="Cancelled")
        except Exception as e:
            log.exception("import %s failed", jid)
            self._set(jid, status="failed", step="Failed", error=str(e)[:800])
        finally:
            self.current = None
            self.proc = None

    # ------------------------------------------------------------ steps
    def _fetch(self, job: dict, work: Path, opts: dict) -> dict:
        info_file = work / "source.info.json"
        if job["kind"] == "upload":
            return json.loads(info_file.read_text()) if info_file.exists() else {"title": job["title"],
                                                                                    "channel": "Upload"}
        if any(p.stem == "source" and p.suffix.lower() in VIDEO_EXT for p in work.iterdir()) and info_file.exists():
            return json.loads(info_file.read_text())  # already downloaded (job restarted)

        jid = job["id"]
        self._set(jid, status="downloading", step="Looking up the video", progress=0)
        base = self._ytdlp_base()
        r = subprocess.run(base + ["-J", job["url"]], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(_ytdlp_error(r.stderr))
        info = json.loads(r.stdout)
        keep = {k: info.get(k) for k in ("id", "title", "channel", "uploader", "duration", "heatmap",
                                          "description", "webpage_url", "was_live", "extractor_key")}
        keep["channel"] = keep.get("channel") or keep.get("uploader") or "Video"
        info_file.write_text(json.dumps(keep))
        self._set(jid, title=keep["title"], channel=keep["channel"], duration=keep.get("duration"),
                  step="Downloading")

        cmd = base + ["-f", "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
                      "--merge-output-format", "mp4", "-o", str(work / "source.%(ext)s"),
                      "--progress-template", "download:PCT %(progress._percent_str)s"]
        if opts.get("chat", True) and keep.get("was_live"):
            cmd += ["--write-subs", "--sub-langs", "live_chat"]
        self.proc = subprocess.Popen(cmd + [job["url"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in self.proc.stdout:
            m = re.search(r"PCT\s+([\d.]+)%", line)
            if m:
                self._set(jid, progress=round(float(m.group(1)) / 100, 3))
        err = self.proc.stderr.read()
        if self.proc.wait() != 0:
            self._check(jid)
            raise RuntimeError(_ytdlp_error(err))
        return keep

    def _clip_captions(self, src: Path, rel_words: list, cfg: ClipConfig, t_source: str) -> list:
        """Accurate captions for one clip: Whisper with the configured model on just
        this slice (seconds of audio instead of the whole video). Reuses the
        full-video transcript when it already came from that same model."""
        if not cfg.captions:
            return []
        if t_source == f"whisper:{cfg.whisper_model}":
            return lines_from_words(rel_words)
        try:
            w, _ = transcribe_words(src, cfg)
            return lines_from_words(w)
        except Exception as e:
            log.warning("clip transcription failed, using the rough transcript: %s", e)
            return lines_from_words(rel_words)

    def _youtube_captions(self, job: dict, work: Path, opts: dict, info: dict) -> tuple[list, list]:
        """Download the video's own captions (json3, word-level timing for
        auto-captions) in the spoken language. Returns ([], []) if none."""
        try:
            base = self._ytdlp_base()
            r = subprocess.run(base + ["-J", job["url"]], capture_output=True, text=True, timeout=120)
            full = json.loads(r.stdout) if r.returncode == 0 else {}
            track = pick_caption_track(full, opts.get("language"))
            if not track:
                return [], []
            code, auto = track
            subprocess.run(base + ["--skip-download", "--write-auto-subs" if auto else "--write-subs",
                                   "--sub-langs", code, "--sub-format", "json3", "-o", str(work / "subs.%(ext)s"),
                                   job["url"]], capture_output=True, text=True, timeout=180)
            f = next(work.glob("subs*.json3"), None)
            if not f:
                return [], []
            words, segments = parse_json3(json.loads(f.read_text(encoding="utf-8")))
            if len(words) < 20 or (info.get("duration") and words[-1][1] < info["duration"] * 0.3):
                return [], []  # too thin to trust (music video, captions only on the intro, ...)
            log.info("using YouTube captions (%s, %s) for job %s", code, "auto" if auto else "manual", job["id"])
            return words, segments
        except Exception as e:
            log.warning("YouTube captions unavailable, falling back to Whisper: %s", e)
            return [], []

    def _ytdlp_base(self) -> list[str]:
        base = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--newline", "--no-warnings"]
        if shutil.which("node") and not shutil.which("deno"):
            base += ["--js-runtimes", "node"]  # YouTube needs a JS runtime since yt-dlp 2025.11
        cookies = self.db.settings().get("ytdlp_cookies")
        if cookies and Path(cookies).exists():
            base += ["--cookies", cookies]
        return base

    def _make_clip(self, job, info, media: Path, words, chat, loud, m, cfg: ClipConfig, layout: str, i: int,
                   t_source: str = "", prog=None):
        a = max(m.start - MARGIN, 0.0)
        b = min(m.end + MARGIN, info["duration"])
        stem = f"job{job['id']}_{i + 1:02}"
        src = self.clips / f"{stem}.src.mp4"
        # exact-time slice, re-encoded at high quality so later trims stay frame-accurate
        # ultrafast + low CRF: it's an intermediate, so encode speed matters more than file size
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "fatal", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                        "-i", str(media), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14",
                        "-c:a", "aac", "-b:a", "192k", str(src)], check=True)
        rel = [[w[0] - a, w[1] - a, w[2]] for w in words if a <= w[0] < b]
        if prog and t_source != f"whisper:{cfg.whisper_model}" and cfg.captions:
            prog(0.0, what="Captioning")
        captions = self._clip_captions(src, rel, cfg, t_source)
        ts, te = m.start - a, m.end - a
        c = ClipConfig(**{**cfg.__dict__, "layout": layout,
                           "facecam": self.db.get(f"facecam:{info.get('channel') or 'Video'}")})
        out = self.clips / f"{stem}.mp4"
        used = render(src, out, c, captions, ts, te, progress=prog)
        thumbnail(out, self.clips / f"{stem}.jpg")

        # heat strip: replay heat if YouTube has it, else loudness
        hm = info.get("heatmap")
        if hm:
            pts = [[round(t - a, 1), round(heat_at(hm, t) * 10, 2)] for t in _frange(a, b, 1.0)]
            label = "replays"
        else:
            pts = [[round(k * 0.5 - a, 1), loud[k]] for k in range(int(a * 2), min(int(b * 2), len(loud)))]
            mx = max((p[1] for p in pts), default=1) or 1
            pts = [[t, round(v / mx * 10, 2)] for t, v in pts]
            label = "loudness"
        base = round(sum(p[1] for p in pts) / len(pts), 2) if pts else 0
        sample = [x for (t0, _, samp) in chat if m.start <= t0 <= m.end + 10 for x in samp][:6]
        link = info.get("webpage_url") or job.get("url") or ""  # "" for uploads
        if link and m.start > 5 and "youtu" in link:
            link += ("&" if "?" in link else "?") + f"t={int(m.start)}"
        s = self.db.settings()
        channel = info.get("channel") or "Video"
        self.db.add_clip({
            "channel": channel, "created": time.time(), "file": out.name, "src": src.name,
            "captions": captions, "thumb": f"{stem}.jpg", "duration": duration_of(out),
            "score": m.score, "baseline": base, "reason": m.source, "chat_sample": sample,
            "heat": {"points": pts, "baseline": base, "spike": None, "label": label},
            "transcript": " ".join(ln[2] for ln in captions if m.start - a <= ln[0] < m.end - a)
            or " ".join(w[2] for w in words if m.start <= w[0] < m.end),
            "title": m.title or info.get("title", "")[:70],
            "caption": render_caption(s["caption_template"], m.title, channel, link),
            "source_kind": "video", "source_url": link or None, "job_id": job["id"], "why": m.why,
            "layout": used, "trim_start": round(ts, 3), "trim_end": round(te, 3), "facecam": c.facecam,
        })

    def _cleanup(self, work: Path, media: Path):
        """The source video is only needed while clipping (each clip keeps its own
        slice), so drop it, but keep the transcript and info for reference."""
        try:
            media.unlink(missing_ok=True)
        except OSError:
            pass


FAST_MODEL = "base"  # full-video transcript for moment finding only; clip captions use the configured model


def pick_caption_track(info: dict, language: str | None = None) -> tuple[str, bool] | None:
    """Choose the caption track in the language actually spoken: manual subtitles
    in that language, else YouTube's auto-captions of the original audio
    ("<lang>-orig", or "<lang>"). Never a machine translation."""
    lang = (language or info.get("language") or "").split("-")[0].lower()
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if lang:
        for k in manual:
            if k.lower() == lang or k.lower().startswith(lang + "-"):
                return k, False
        for k in (f"{lang}-orig", lang):
            if k in auto:
                return k, True
    orig = [k for k in auto if k.endswith("-orig")]
    if len(orig) == 1:
        return orig[0], True
    return None


def parse_json3(data: dict) -> tuple[list, list]:
    """YouTube json3 captions -> (words, segments) like transcribe_words.
    Auto-captions carry per-word offsets; manual ones only per line, so their
    words are spread evenly over the line."""
    words = []
    for ev in data.get("events", []):
        segs = [sg for sg in ev.get("segs") or [] if sg.get("utf8", "").strip()]
        if not segs or ev.get("aAppend"):
            continue
        t0 = ev.get("tStartMs", 0) / 1000
        dur = ev.get("dDurationMs", 0) / 1000
        if len(segs) == 1 and "tOffsetMs" not in segs[0]:
            toks = segs[0]["utf8"].split()
            for k, tok in enumerate(toks):
                a = t0 + dur * k / len(toks)
                words.append([round(a, 2), round(a + dur / len(toks), 2), tok])
            continue
        for k, sg in enumerate(segs):
            a = t0 + sg.get("tOffsetMs", 0) / 1000
            b = t0 + segs[k + 1].get("tOffsetMs", 0) / 1000 if k + 1 < len(segs) else t0 + dur
            for tok in sg["utf8"].split():
                words.append([round(a, 2), round(max(b, a + 0.1), 2), tok])
    words.sort(key=lambda w: w[0])
    for k in range(len(words) - 1):  # rolling auto-captions overlap: end each word at the next start
        words[k][1] = round(min(words[k][1], max(words[k + 1][0], words[k][0] + 0.05)), 2)
    # segments: auto-captions have no punctuation, so split on pauses / length
    segments, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] > 0.7 or len(cur) >= 16 or cur[-1][2][-1:] in ".?!"):
            segments.append([cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)])
            cur = []
        cur.append(w)
    if cur:
        segments.append([cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)])
    return words, segments


def _human(sec: float) -> str:
    return f"{max(1, round(sec / 60))} min" if sec >= 60 else f"{int(sec)} s"


def _frange(a, b, step):
    t = a
    while t < b:
        yield t
        t += step


def _ytdlp_error(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").splitlines() if "ERROR" in ln] or (stderr or "").splitlines()[-2:]
    msg = " ".join(lines)[-400:] or "Download failed"
    if "confirm you" in msg and "bot" in msg:
        msg += (" — YouTube is blocking this server. Export your browser's YouTube cookies to a cookies.txt, "
                "put it in the data folder and set its path in Settings → Imports.")
    if "Private video" in msg or "members-only" in msg.lower():
        msg += " — the video must be public or unlisted."
    return msg

"""Editing toolkit shared by the Twitch clipper and the video importer:
transcription, word-highlight captions, face-tracking reframe, rendering."""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("editing")

FAST_MODEL = "base"  # rough full-length transcripts (finding/cutting); final captions use the configured model
HIGHLIGHT = "&H003DB3F2&"  # amber #f2b33d in ASS BGR order


@dataclass
class ClipConfig:
    pre_roll: float = 30.0    # Twitch: seconds before the spike (chat reacts AFTER the moment)
    post_roll: float = 15.0   # Twitch: seconds after
    captions: bool = True
    whisper_model: str = "small"
    language: str | None = None
    # split (facecam on top, gameplay below) | auto (9:16, follows the speaker's face)
    # | fit (9:16, whole frame on blurred bg) | crop (9:16 center) | original (source aspect)
    layout: str = "original"
    crf: int = 20
    facecam: dict | None = None   # split layout: {"x","y","w","h"} 0..1 of the source frame; None = auto-detect
    context: str | None = None    # one line that sets up the situation, shown at the top for the first seconds
    context_secs: float = 3.5
    credit: str | None = None     # who made the content, on screen the whole clip (e.g. "twitch.tv/papaplatte")
    handle: str | None = None     # our own page, smaller, under the credit (e.g. "@streamweltclips")

    @classmethod
    def from_settings(cls, s: dict) -> "ClipConfig":
        return cls(pre_roll=s["pre_roll"], post_roll=s["post_roll"], captions=s["captions"],
                   whisper_model=s["whisper_model"], language=s["language"] or None, layout=s["layout"])


# ------------------------------------------------------------------ ffmpeg utils
def ffmpeg(*args: str, cwd=None) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "fatal", "-y", *args], check=True, cwd=cwd)


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        s = json.loads(out)["streams"][0]
        return int(s["width"]), int(s["height"])
    except (ValueError, KeyError, IndexError):
        return 1920, 1080


def thumbnail(mp4: Path, jpg: Path) -> None:
    d = duration_of(mp4)
    ffmpeg("-ss", f"{d * 0.4:.2f}", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(jpg))


# ------------------------------------------------------------------ transcription
_model = None
_model_name = None
_model_lock = threading.Lock()


def load_audio(media: Path, sr: int = 16000):
    """Decode any audio/video file to 16 kHz mono float32 with ffmpeg.
    We hand Whisper this array instead of a file path: faster-whisper's own
    decoder calls av.open(metadata_errors=...), which newer PyAV versions removed
    ("open() got an unexpected keyword argument 'metadata_errors'")."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"couldn't read audio: {r.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32)


def transcribe_words(media: Path, cfg: ClipConfig, progress: Callable[[float], None] | None = None,
                     total: float | None = None) -> tuple[list[list], list[list]]:
    """faster-whisper with word timestamps. Returns (words, segments):
    words = [[start, end, text], ...], segments = [[start, end, text], ...] (sentence-ish)."""
    global _model, _model_name
    from faster_whisper import WhisperModel  # heavy + optional, so imported lazily
    with _model_lock:
        if _model is None or _model_name != cfg.whisper_model:
            log.info("loading whisper model '%s' (first time downloads it)", cfg.whisper_model)
            import os
            _model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8",
                                  cpu_threads=os.cpu_count() or 4)
            _model_name = cfg.whisper_model
        audio = load_audio(media)
        if not len(audio):
            return [], []
        segs, _ = _model.transcribe(audio, language=cfg.language, word_timestamps=True, vad_filter=True)
        words, segments = [], []
        for seg in segs:  # generator: transcription happens while we iterate
            segments.append([round(seg.start, 2), round(seg.end, 2), seg.text.strip()])
            words += [[round(w.start, 2), round(w.end, 2), w.word.strip()] for w in (seg.words or []) if w.word.strip()]
            if progress and total:
                progress(min(seg.end / total, 1.0))
    return words, segments


def lines_from_words(words: list[list], max_words: int = 4, max_span: float = 2.2) -> list[list]:
    """Group words into short caption lines: [start, end, text, [[s, e, w], ...]]."""
    lines, cur = [], []
    for w in words:
        gap = cur and w[0] - cur[-1][1] > 0.6
        ends_sentence = cur and cur[-1][2][-1:] in ".?!"
        if cur and (len(cur) >= max_words or w[1] - cur[0][0] > max_span or gap or ends_sentence):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [[ln[0][0], ln[-1][1], " ".join(x[2] for x in ln), ln] for ln in lines]


def transcribe(media: Path, cfg: ClipConfig) -> list[list]:
    words, _ = transcribe_words(media, cfg)
    return lines_from_words(words)


# ------------------------------------------------------------------ captions (ASS)
def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _ass_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


CONTEXT_INK = "&H00041D2A&"   # #2a1d04 text on the amber "tally" box


def write_ass(lines: list, path: Path, offset: float, width: int, height: int, end: float | None = None,
              center_y: float | None = None, context: str | None = None, context_secs: float = 3.5,
              credit: str | None = None, handle: str | None = None) -> int:
    """Short-form style captions: big bold lines, the word being spoken in amber.
    `lines` items are [start, end, text] or [start, end, text, words]; times are
    relative to the source and shifted by -offset."""
    vertical = height > width
    fs = round(height * (0.037 if vertical else 0.055))
    margin_v = round(height * (0.29 if vertical else 0.07))  # vertical: above the platforms' UI
    if center_y is not None:  # e.g. split layout: sit on the seam between facecam and gameplay
        margin_v = max(0, round(height * (1 - center_y) - fs * 0.6))
    out = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,DejaVu Sans,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(3, fs // 14)},{max(1, fs // 30)},2,{width // 12},{width // 12},{margin_v},1",
        # context banner: dark text on an amber box (BorderStyle 3 = opaque box in OutlineColour)
        f"Style: Context,DejaVu Sans,{round(height * (0.029 if vertical else 0.045))},{CONTEXT_INK},{CONTEXT_INK},"
        f"{HIGHLIGHT},&H00000000,-1,0,0,0,100,100,0,0,3,{max(8, round(height * 0.009))},0,8,"
        f"{width // 10},{width // 10},{round(height * (0.055 if vertical else 0.06))},1",
        # credit + our handle: small, top centre (where the context banner was), clear of the
        # platforms' right-hand buttons; the streamer's name is the bigger line
        f"Style: Credit,DejaVu Sans,{round(height * (0.021 if vertical else 0.032))},&H10FFFFFF,&H10FFFFFF,"
        f"&H80000000,&H00000000,-1,0,0,0,100,100,0,0,1,{max(2, round(height * 0.0015))},0,8,"
        f"{width // 10},{width // 10},{round(height * (0.06 if vertical else 0.03))},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    n = 0
    clip_end = (end - offset) if end else None

    def emit(a, b, text):
        nonlocal n
        a, b = a - offset, b - offset
        if clip_end is not None:
            b = min(b, clip_end)
        if b <= 0.05 or (clip_end is not None and a >= clip_end):
            return
        out.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Cap,,0,0,0,,{text}")
        n += 1

    if (credit and credit.strip()) or (handle and handle.strip()):
        small = round(height * (0.016 if vertical else 0.024))
        parts = [_ass_escape(credit.strip())[:60]] if credit and credit.strip() else []
        if handle and handle.strip():
            parts.append(f"{{\\fs{small}\\b0\\alpha&H70&}}{_ass_escape(handle.strip())[:40]}")
        until = clip_end if clip_end is not None else 36000
        # the hook seconds belong to the context line: the credit takes its place right after
        since = min(context_secs, max(until - 1.0, 0)) if context and context.strip() else 0
        text = "\\N".join(parts)  # \N = line break in ASS
        out.append(f"Dialogue: 2,{_ass_time(since)},{_ass_time(max(until, since + 0.5))},Credit,,0,0,0,,"
                   f"{{\\fad(250,0)}}{text}")
        n += 1
    if context and context.strip():
        dur = context_secs if clip_end is None else min(context_secs, max(clip_end, 0.5))
        out.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(dur)},Context,,0,0,0,,"
                   f"{{\\fad(150,300)}}{_ass_escape(context.strip())[:140]}")
        n += 1
    for ln in lines:
        start, stop, text = ln[0], ln[1], ln[2]
        words = ln[3] if len(ln) > 3 and ln[3] else None
        case = (lambda s: s.upper()) if vertical else (lambda s: s)
        if not words:
            emit(start, stop, case(_ass_escape(text)))
            continue
        toks = [case(_ass_escape(w[2])) for w in words]
        for i, w in enumerate(words):
            a = start if i == 0 else w[0]
            b = words[i + 1][0] if i + 1 < len(words) else stop
            shown = " ".join(
                f"{{\\c{HIGHLIGHT}\\fscx108\\fscy108}}{t}{{\\r}}" if j == i else t for j, t in enumerate(toks))
            emit(a, max(b, a + 0.05), shown)
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


# ------------------------------------------------------------------ face-tracking reframe
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def _yunet_model() -> Path | None:
    """OpenCV's YuNet face detector (230 KB). Downloaded once next to the Whisper
    models and hash-checked. Returns None if it can't be had (no network)."""
    import hashlib
    import os
    path = Path(os.environ.get("YUNET_MODEL") or Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "yunet.onnx")
    if path.exists():
        return path
    try:
        import httpx
        data = httpx.get(YUNET_URL, follow_redirects=True, timeout=60).content
        if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
            raise ValueError("checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    except Exception as e:
        log.warning("face model unavailable (%s); auto layout will fall back", e)
        return None


def _face_detector():
    """Returns detect(bgr_small) -> list of (x, y, w, h) or None if no detector."""
    import cv2
    model = _yunet_model()
    if model and hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.75, 0.3, 50)

        def detect(img):
            det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = det.detect(img)
            return [tuple(f[:4]) for f in faces] if faces is not None else []
        return detect
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml") if cascade_dir else None
    if cascade is not None and not cascade.empty():
        def detect(img):
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            return [tuple(f) for f in cascade.detectMultiScale(gray, 1.1, 6, minSize=(36, 36))]
        return detect
    return None


def face_track(src: Path, start: float, end: float | None, fps: float = 4.0) -> list[tuple[float, float | None]]:
    """Sample frames and return [(t_rel, face_center_x 0..1 or None)]. Picks the
    largest face, preferring one near the previous position so it doesn't flip
    between two people every frame."""
    import cv2
    detect = _face_detector()
    if detect is None:
        return []
    cap = cv2.VideoCapture(str(src))
    dur = (end or duration_of(src)) - start
    samples, prev = [], None
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # seek once, then read sequentially (much faster)
    next_t = 0.0
    while next_t <= dur:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
        if t + 1e-3 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, int(h * 480 / w)))
        faces = detect(small)
        cx = None
        if faces:
            def rank(f):
                x, y, fw, fh = f
                c = (x + fw / 2) / 480
                return fw * fh * (1.0 if prev is None else max(0.35, 1 - abs(c - prev) * 1.5))
            x, y, fw, fh = max(faces, key=rank)
            cx = (x + fw / 2) / 480
            prev = cx
        samples.append((round(next_t, 2), cx))
        next_t += 1 / fps
    cap.release()
    return samples


SPLIT_TOP = 768                 # split layout: default facecam panel height (of 1920)
SPLIT_ASPECT = 1080 / SPLIT_TOP  # facecam box aspect used when the cam's own frame can't be found


def split_top_height(box: dict, iw: int, ih: int) -> int:
    """Top panel height that shows the whole facecam box undistorted at 1080 wide,
    kept between 28% and 50% of the screen so the gameplay stays big."""
    aspect = (box["w"] * iw) / max(box["h"] * ih, 1)
    return int(min(max(1080 / aspect, 540), 960)) // 2 * 2


def _peaks(seg):
    """Indices of local maxima, at least 3 apart (stronger one wins)."""
    import numpy as np
    idx = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    idx.sort(key=lambda i: -seg[i])
    out = []
    for i in idx:
        if all(abs(i - j) >= 3 for j in out):
            out.append(i)
    return np.array(sorted(out), dtype=int)


def _cam_border(prof, inside, outside, lo: float, hi: float) -> int | None:
    """Pick the facecam's edge among straight lines in [lo, hi): a strong, persistent
    edge with calm pixels on the cam side and busy (gameplay) pixels outside."""
    import numpy as np
    lo, hi = max(int(lo), 3), min(int(hi), len(prof) - 3)
    if hi - lo < 4:
        return None
    seg = prof[lo:hi]
    base = float(np.median(seg)) + 1e-3
    best, best_score = None, 0.0
    for p in _peaks(seg):
        if seg[p] < 1.8 * base:
            continue
        i = lo + int(p)
        score = float(seg[p]) * max(outside(i) - inside(i), 0.0)
        if score > best_score:
            best, best_score = i, score
    return best


def detect_facecam(src: Path, start: float = 0.0, end: float | None = None,
                   fps: float = 0.5) -> dict | None:
    """Find the streamer's webcam overlay. Two steps:
    1. faces that stay in the SAME place for most of the clip (game characters move
       around, a facecam doesn't); nearby persistent faces (two people on one cam)
       are grouped.
    2. the cam's rectangle around them: overlays are static, so their borders show
       up as sharp lines in the edge map averaged over time, while the gameplay
       blurs away; among those lines, the border is the one with calm pixels on the
       cam side and moving gameplay outside (shelves inside the room don't qualify).
    Returns {x, y, w, h} in 0..1 of the frame, or None (no facecam)."""
    import cv2
    import numpy as np
    detect = _face_detector()
    if detect is None:
        return None
    cap = cv2.VideoCapture(str(src))
    dur = (end or duration_of(src)) - start
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    faces, grays, n, next_t, W, H = [], [], 0, 0.0, 960, 540
    while next_t <= dur and n < 60:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
        if t + 1e-3 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        small = cv2.resize(frame, (W, int(frame.shape[0] * W / frame.shape[1])))
        H = small.shape[0]
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
        for x, y, fw, fh in detect(small):  # facecam faces are small, so detect at 960 wide
            faces.append((n, (x + fw / 2) / W, (y + fh / 2) / H, fw / W, fh / H))
        n += 1
        next_t += 1 / fps
    cap.release()
    if n < 3 or not faces:
        return None

    # 1. persistent faces
    clusters: list[list] = []
    for f in faces:
        for c in clusters:
            if abs(c[0][1] - f[1]) < 0.035 and abs(c[0][2] - f[2]) < 0.05:
                c.append(f)
                break
        else:
            clusters.append([f])
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    persistent = []
    for c in clusters:
        frames = len({f[0] for f in c})
        if frames >= 0.5 * n:
            persistent.append({"cx": med([f[1] for f in c]), "cy": med([f[2] for f in c]),
                               "w": med([f[3] for f in c]), "h": med([f[4] for f in c]), "frames": frames})
    if not persistent:
        return None
    groups: list[list] = []
    for p in sorted(persistent, key=lambda p: -p["frames"]):
        for g in groups:
            if any(abs(q["cx"] - p["cx"]) < 0.15 and abs(q["cy"] - p["cy"]) < 0.15 for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    g = max(groups, key=lambda g: sum(p["frames"] for p in g))
    if any(p["w"] > 0.2 for p in g):  # a face filling the frame is a talking-head video, not a facecam
        return None
    fx0 = min(p["cx"] - p["w"] / 2 for p in g) * W
    fx1 = max(p["cx"] + p["w"] / 2 for p in g) * W
    fy0 = min(p["cy"] - p["h"] / 2 for p in g) * H
    fy1 = max(p["cy"] + p["h"] / 2 for p in g) * H
    fw, fh = max(p["w"] for p in g) * W, max(p["h"] for p in g) * H

    # 2. the cam's own frame
    stack = np.stack(grays)
    motion = stack.std(axis=0)
    ex = np.mean([np.abs(cv2.Sobel(gr, cv2.CV_32F, 1, 0, ksize=3)) for gr in grays], axis=0)
    ey = np.mean([np.abs(cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3)) for gr in grays], axis=0)
    xs = slice(int(max(fx0 - fw, 0)), int(min(fx1 + fw, W)))
    ys = slice(int(max(fy0 - 0.5 * fh, 0)), int(min(fy1 + 0.5 * fh, H)))
    rows, cols = ey[:, xs].mean(axis=1), ex[ys, :].mean(axis=0)
    rm = lambda a, b: float(motion[max(a, 0):max(b, 1), xs].mean())  # noqa: E731
    cm = lambda a, b: float(motion[ys, max(a, 0):max(b, 1)].mean())  # noqa: E731
    k = 10
    top = _cam_border(rows, lambda i: rm(i + 2, i + k), lambda i: rm(i - k, i - 2), fy0 - 4 * fh, fy0 - 0.4 * fh)
    bot = _cam_border(rows, lambda i: rm(i - k, i - 2), lambda i: rm(i + 2, i + k), fy1 + 0.4 * fh, fy1 + 4 * fh)
    left = _cam_border(cols, lambda i: cm(i + 2, i + k), lambda i: cm(i - k, i - 2), fx0 - 7 * fw, fx0 - 0.4 * fw)
    right = _cam_border(cols, lambda i: cm(i - k, i - 2), lambda i: cm(i + 2, i + k), fx1 + 0.4 * fw, fx1 + 7 * fw)
    found = sum(v is not None for v in (top, bot, left, right))

    if found >= 2:
        # a missing side is usually the screen edge (cam pushed into a corner) or a borderless cam
        x0 = left + 2 if left is not None else max(fx0 - 3 * fw, 0)
        x1 = right - 2 if right is not None else min(fx1 + 3 * fw, W)
        y0 = top + 2 if top is not None else max(fy0 - 2 * fh, 0)
        y1 = bot - 2 if bot is not None else min(fy1 + 2 * fh, H)
        if x1 - x0 > 1.5 * (fx1 - fx0) and y1 - y0 > 1.5 * (fy1 - fy0) and (x1 - x0) * (y1 - y0) < 0.35 * W * H:
            return {"x": round(x0 / W, 4), "y": round(y0 / H, 4), "w": round((x1 - x0) / W, 4),
                    "h": round((y1 - y0) / H, 4), "auto": True}

    # fallback: a box around the faces at the default panel aspect
    bw = (fx1 - fx0) + 3.0 * fw
    bh = bw / SPLIT_ASPECT
    if bh < (fy1 - fy0) + 1.7 * fh:
        bh = (fy1 - fy0) + 1.7 * fh
        bw = bh * SPLIT_ASPECT
    bw, bh = min(bw, W), min(bh, H)
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2 + 0.07 * bh
    bx = min(max(cx - bw / 2, 0), W - bw)
    by = min(max(cy - bh / 2, 0), H - bh)
    return {"x": round(bx / W, 4), "y": round(by / H, 4), "w": round(bw / W, 4), "h": round(bh / H, 4),
            "auto": True}


def smooth_track(samples: list[tuple[float, float | None]], step: float = 0.2,
                 deadzone: float = 0.035) -> list[tuple[float, float]] | None:
    """Turn noisy face positions into calm camera moves: fill gaps, median-filter,
    hold still inside a dead zone, ease toward new positions. Returns keypoints
    every `step` s, or None when faces are too rare to trust (caller falls back)."""
    if not samples:
        return None
    found = [s for s in samples if s[1] is not None]
    if len(found) < max(2, 0.3 * len(samples)):
        return None
    # fill gaps with the nearest known value
    xs, last = [], found[0][1]
    for _, c in samples:
        last = c if c is not None else last
        xs.append(last)
    # median filter (window 5) kills single-frame false positives
    med = [sorted(xs[max(0, i - 2):i + 3])[len(xs[max(0, i - 2):i + 3]) // 2] for i in range(len(xs))]
    times = [s[0] for s in samples]
    # virtual camera: only re-aim when the face leaves the dead zone, then ease in
    out, cam, target = [], med[0], med[0]
    t, i, end = 0.0, 0, times[-1]
    while t <= end + 1e-6:
        while i + 1 < len(times) and times[i + 1] <= t:
            i += 1
        if abs(med[i] - cam) > deadzone:
            target = med[i]
        cam += (target - cam) * 0.3
        out.append((round(t, 2), round(cam, 4)))
        t += step
    return out


def _crop_cmds(track: list[tuple[float, float]], iw: int, cw: int, path: Path) -> int:
    lines = []
    for t, cx in track:
        x = int(min(max(cx * iw - cw / 2, 0), iw - cw))
        lines.append(f"{t:.2f} crop x {x};")
    path.write_text("\n".join(lines) + "\n")
    return int(min(max(track[0][1] * iw - cw / 2, 0), iw - cw))


# ------------------------------------------------------------------ render
def render(src: Path, out_mp4: Path, cfg: ClipConfig, captions: list | None,
           start: float = 0.0, end: float | None = None,
           progress: Callable[[float], None] | None = None) -> str:
    """Encode src[start:end] -> out_mp4 with layout + captions. Caption times are
    relative to src, so trims keep them in sync. Returns the layout actually used
    ('auto' falls back to 'fit' when no face is found)."""
    iw, ih = video_size(src)
    layout = cfg.layout
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if layout == "auto":
            track = smooth_track(face_track(src, start, end))
            if track and iw > ih:
                cw = int(ih * 9 / 16) // 2 * 2
                x0 = _crop_cmds(track, iw, cw, tdp / "crop.cmd")
                base = f"[0:v]sendcmd=f=crop.cmd,crop={cw}:{ih}:{x0}:0,scale=1080:1920,setsar=1[base]"
            else:
                layout = "fit"
        cap_y = None
        if layout == "split":
            box = cfg.facecam or (detect_facecam(src, start, end) if iw > ih else None)
            if box:
                cfg.facecam = box  # callers store it, so re-renders and the editor start from it
                fx, fy = int(box["x"] * iw), int(box["y"] * ih)
                fw, fh = max(int(box["w"] * iw) // 2 * 2, 16), max(int(box["h"] * ih) // 2 * 2, 16)
                top_h = split_top_height(box, iw, ih)
                bot_h = 1920 - top_h
                gw = min(int(ih * 1080 / bot_h) // 2 * 2, iw)  # gameplay: center crop at the bottom panel's aspect
                base = (f"[0:v]split[cam][game];"
                        f"[cam]crop={fw}:{fh}:{fx}:{fy},scale=1080:{top_h}:flags=lanczos,setsar=1[top];"
                        f"[game]crop={gw}:{ih}:(iw-{gw})/2:0,scale=1080:{bot_h},setsar=1[bot];"
                        f"[top][bot]vstack[base]")
                cap_y = top_h / 1920
            else:
                layout = "fit"  # no facecam in this clip
        if layout == "crop":
            base = "[0:v]crop=ih*9/16:ih,scale=1080:1920,setsar=1[base]"
        elif layout == "fit":  # whole frame centered over a blurred, zoomed copy of itself
            base = ("[0:v]split[a][b];"
                    # blur a small copy, then scale it up: same look, ~4x cheaper than blurring at 1080x1920
                    "[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
                    "scale=1080:1920:flags=bilinear[bg];"
                    "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]")
        elif layout == "original":
            base = "[0:v]null[base]"
        ow, oh = (1080, 1920) if layout in ("split", "auto", "fit", "crop") else (iw, ih)
        graph = base
        if (captions or cfg.context or cfg.credit or cfg.handle) and write_ass(
                captions or [], tdp / "c.ass", offset=start, width=ow, height=oh, end=end, center_y=cap_y,
                context=cfg.context, context_secs=cfg.context_secs, credit=cfg.credit, handle=cfg.handle):
            graph += ";[base]subtitles=c.ass[v]"
        else:
            graph += ";[base]null[v]"
        cut = ["-ss", f"{start:.3f}"] + (["-to", f"{end:.3f}"] if end else [])
        total = max((end or duration_of(src)) - start, 0.1)
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y",
             "-fflags", "+discardcorrupt", *cut, "-i", str(src.resolve()),
             "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out_mp4.resolve())],
            cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:  # "out_time_us=12345678" lines while encoding
            if progress and line.startswith("out_time_us=") and line[12:].strip().isdigit():
                progress(min(int(line[12:]) / 1e6 / total, 1.0))
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg render failed: {err.strip()[-400:]}")
    return layout

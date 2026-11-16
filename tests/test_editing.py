"""Captions + reframe logic (no video decoding needed)."""
import subprocess

from clipwatcher.editing import lines_from_words, load_audio, smooth_track, write_ass


def test_lines_break_on_sentence_and_length():
    words = [[0, .3, "Hello"], [.3, .6, "there."], [.7, 1, "This"], [1, 1.3, "is"], [1.3, 1.6, "a"],
             [1.6, 1.9, "long"], [1.9, 2.2, "line"]]
    lines = lines_from_words(words, max_words=4)
    assert [ln[2] for ln in lines] == ["Hello there.", "This is a long", "line"]


def test_ass_highlights_each_word_and_respects_trim(tmp_path):
    lines = lines_from_words([[1, 1.4, "one"], [1.4, 1.8, "two"], [5, 5.4, "three"]])
    n = write_ass(lines, tmp_path / "c.ass", offset=0.5, width=1080, height=1920, end=4.0)
    text = (tmp_path / "c.ass").read_text()
    assert n == 2                      # "three" is after the trim end
    assert "PlayResY: 1920" in text and "\\c&H003DB3F2&" in text
    assert "ONE" in text               # vertical captions are upper-case


def test_smooth_track_holds_still_then_follows():
    samples = [(i / 4, 0.3) for i in range(20)] + [(5 + i / 4, 0.7) for i in range(20)]
    tr = smooth_track(samples)
    early = [cx for t, cx in tr if t < 4]
    late = [cx for t, cx in tr if t > 8]
    assert max(early) - min(early) < 0.01
    assert abs(late[-1] - 0.7) < 0.02


def test_smooth_track_gives_up_without_faces():
    assert smooth_track([(i / 4, None) for i in range(40)]) is None


def test_load_audio_decodes_video_to_16k_mono(tmp_path):
    """Whisper gets a numpy array, never a path (faster-whisper's own decoder
    breaks on newer PyAV)."""
    f = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=64x64:d=2", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(f)],
                   check=True)
    a = load_audio(f)
    assert a.dtype.name == "float32"
    assert abs(len(a) - 32000) < 1600
    assert 0.05 < float(abs(a).max()) <= 1.0


def test_split_top_height_fits_cam_shape():
    from clipwatcher.editing import split_top_height
    wide = {"x": 0.75, "y": 0.3, "w": 0.24, "h": 0.23}      # 461x248 px cam on 1920x1080
    square = {"x": 0.8, "y": 0.4, "w": 0.15, "h": 0.27}     # 288x292 px
    assert split_top_height(wide, 1920, 1080) == 582          # 1080 / (460.8/248.4), even
    assert split_top_height(square, 1920, 1080) == 960        # capped at half the screen
    assert split_top_height({"x": 0, "y": 0, "w": 0.9, "h": 0.1}, 1920, 1080) == 540  # floor


def test_cam_border_prefers_line_with_busy_outside():
    """Two equally strong lines; only one separates calm cam pixels from moving gameplay."""
    import numpy as np
    from clipwatcher.editing import _cam_border
    prof = np.ones(100)
    prof[30] = prof[60] = 20.0                    # 30 = shelf inside the cam, 60 = the cam's frame
    busy = lambda i: 9.0 if i > 60 else 1.0        # noqa: E731  gameplay below row 60
    inside = lambda i: busy(i - 5)                 # noqa: E731
    outside = lambda i: busy(i + 5)                # noqa: E731
    assert _cam_border(prof, inside, outside, 10, 90) == 60

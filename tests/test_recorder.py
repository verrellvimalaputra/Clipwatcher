"""The wall-clock -> byte-offset index must survive gaps (filtered Twitch ads)."""
from clipwatcher.recorder import TS_PACKET, Session


def test_offsets_follow_wall_clock_across_gap(tmp_path):
    s = Session(tmp_path / "x.ts")
    t = 1000.0
    for _ in range(40):          # 10 s of data, 0.25 s apart
        s.add(t, TS_PACKET * 100)
        t += 0.25
    t += 30                      # 30 s gap: no bytes arrive
    for _ in range(40):
        s.add(t, TS_PACKET * 100)
        t += 0.25
    before_gap = s.offset_at(1009.9)
    inside_gap = s.offset_at(1025.0)
    after_gap = s.offset_at(1040.5)
    assert before_gap == inside_gap        # nothing recorded during the gap
    assert after_gap > inside_gap
    assert after_gap % TS_PACKET == 0


def test_write_survives_a_transient_io_error(tmp_path):
    """Docker Desktop's Windows bind mount can fail one write with EIO; recording must go on."""
    from clipwatcher.recorder import Recorder

    sess = Session(tmp_path / "s.ts")
    real = open(sess.path, "wb")

    class Flaky:
        calls = 0

        def write(self, b):
            Flaky.calls += 1
            real.write(b[:100])  # partial write, then the mount hiccups
            raise OSError(5, "Input/output error")

        def flush(self):
            pass

        def close(self):
            real.close()

    good = b"A" * 376
    f = Recorder._write(Flaky(), sess, good)
    sess.add(0.0, len(good))
    f = Recorder._write(f, sess, b"B" * 188)
    sess.add(1.0, 188)
    f.close()
    assert Flaky.calls == 1
    assert sess.path.read_bytes() == good + b"B" * 188


def test_fragmented_mp4_slices_play(tmp_path):
    """Twitch serves some channels as fMP4: a slice must start on a fragment and get the init segment."""
    import shutil
    import subprocess

    import pytest

    from clipwatcher.recorder import BoxScanner

    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    src = tmp_path / "live.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=10",
                    "-c:v", "libx264", "-g", "30", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                    "-f", "mp4", str(src)], check=True)
    data = src.read_bytes()
    assert data[4:8] == b"ftyp"
    # replay it as streamlink would deliver it: odd-sized chunks, 10 s of wall clock
    sess, scan = Session(tmp_path / "rec.ts", fmp4=True), BoxScanner()
    step = 4099
    with open(sess.path, "wb") as f:
        for i in range(0, len(data), step):
            chunk = data[i:i + step]
            f.write(chunk)
            sess.add(1000 + 10 * i / len(data), len(chunk), scan.feed(chunk))
    assert not scan.broken and sess.inits and len(sess.offsets) >= 9
    out = tmp_path / "slice.src.ts"
    assert sess.extract(1004.0, 1007.0, out) > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    assert 60 <= int(probe.stdout.split()[0]) <= 150   # ~3-4 s of 30 fps, decodable from the first frame
    from clipwatcher.editing import duration_of
    assert 2 <= duration_of(out) <= 5               # the slice's own length, not "from timestamp 0"

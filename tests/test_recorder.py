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

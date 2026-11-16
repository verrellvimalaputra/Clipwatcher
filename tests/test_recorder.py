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

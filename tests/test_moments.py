"""Moment finding: snapping, dedupe, heuristic, chat-replay parsing."""
import json

from clipwatcher.moments import Moment, dedupe, heuristic_moments, snap, target_count, youtube_chat


def transcript(n=60, wps=2.5):
    """n sentences of 8 words, ~3.2 s each, 0.6 s pauses."""
    words, segs, t = [], [], 1.0
    for i in range(n):
        start = t
        for k in range(8):
            words.append([round(t, 2), round(t + 0.32, 2), f"w{i}_{k}" + ("." if k == 7 else "")])
            t += 1 / wps
        segs.append([round(start, 2), words[-1][1], f"This is sentence number {i}."])
        t += 0.6
    return words, segs


def test_snap_moves_to_sentence_edges_and_enforces_length():
    words, segs = transcript()
    m = snap(Moment(10.3, 49.0, "x"), words, segs, 30, 60)
    starts = [s[0] for s in segs]
    ends = [s[1] for s in segs]
    assert any(abs(m.start + 0.15 - s) < 0.01 for s in starts)
    assert any(abs(m.end - 0.35 - e) < 0.01 for e in ends)
    assert snap(Moment(10, 15, "too short"), words, segs, 30, 60) is None
    long = snap(Moment(10, 150, "too long"), words, segs, 30, 60)
    assert long.end - long.start <= 61


def test_dedupe_keeps_best_of_overlapping():
    ms = dedupe([Moment(0, 40, "a", score=50), Moment(20, 60, "b", score=90), Moment(70, 100, "c", score=10)])
    assert [m.title for m in ms] == ["b", "c"]


def test_target_count():
    assert target_count(3600, "auto") == 15
    assert target_count(600, "auto") == 2
    assert target_count(600, "7") == 7


def test_heuristic_prefers_loud_peak_inside_clip():
    words, segs = transcript(60)
    loud = [100.0] * 500
    ms = heuristic_moments({"duration": 230}, segs, words, {"loud": [(120.0, 8.0)]}, loud, "short", 1)
    assert ms and ms[0].start < 120 < ms[0].end - 3
    assert all(m.end <= segs[-1][1] + 0.01 for m in ms)


def test_youtube_chat_replay_parsing(tmp_path):
    f = tmp_path / "x.live_chat.json"
    line = {"replayChatItemAction": {"videoOffsetTimeMsec": "61500", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"authorName": {"simpleText": "ana"},
                                        "message": {"runs": [{"text": "no way "}, {"emoji": {"shortcuts": [":fire:"]}}]}}}}}]}}
    f.write_text(json.dumps(line) + "\n" + "not json\n")
    assert youtube_chat(f) == [(61.5, "ana", "no way :fire:")]

"""YouTube caption handling: pick the spoken-language track, parse json3."""
from clipwatcher.importer import parse_json3, pick_caption_track

AUTO_JSON3 = {"events": [
    {"tStartMs": 0, "dDurationMs": 4000, "wWinId": 1},  # window definition, no segs
    {"tStartMs": 1200, "dDurationMs": 3000, "segs": [{"utf8": "this"}, {"utf8": " is", "tOffsetMs": 300},
                                                      {"utf8": " insane", "tOffsetMs": 700}]},
    {"tStartMs": 2500, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 2600, "dDurationMs": 2500, "segs": [{"utf8": "look"}, {"utf8": " at", "tOffsetMs": 400},
                                                      {"utf8": " this", "tOffsetMs": 600}]},
    {"tStartMs": 9000, "dDurationMs": 2000, "segs": [{"utf8": "after"}, {"utf8": " pause", "tOffsetMs": 500}]},
]}


def test_parse_auto_captions_word_timing_and_pause_segments():
    words, segs = parse_json3(AUTO_JSON3)
    assert [w[2] for w in words] == ["this", "is", "insane", "look", "at", "this", "after", "pause"]
    assert words[1][0] == 1.5 and words[2][0] == 1.9
    assert all(w[1] > w[0] for w in words)
    assert all(words[k][1] <= words[k + 1][0] for k in range(len(words) - 1))  # no overlaps
    assert [s[2] for s in segs] == ["this is insane look at this", "after pause"]


def test_parse_manual_captions_spreads_words():
    words, _ = parse_json3({"events": [{"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "one two three four"}]}]})
    assert [w[0] for w in words] == [1.0, 1.5, 2.0, 2.5]


def test_pick_track_prefers_spoken_language_never_translation():
    info = {"language": "id", "subtitles": {"en": []},  # manual English = a translation, skip
            "automatic_captions": {"id-orig": [], "id": [], "en": [], "de": []}}
    assert pick_caption_track(info) == ("id-orig", True)
    info2 = {"language": "en", "subtitles": {"en-US": []}, "automatic_captions": {"en": []}}
    assert pick_caption_track(info2) == ("en-US", False)
    assert pick_caption_track({"automatic_captions": {"de-orig": [], "en": []}}) == ("de-orig", True)
    assert pick_caption_track({"automatic_captions": {"en": [], "de": []}}) is None
    assert pick_caption_track({"automatic_captions": {"en": []}}, language="en") == ("en", True)

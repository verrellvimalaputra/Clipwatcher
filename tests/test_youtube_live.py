"""YouTube Live: link parsing, chat parsing, and the yt-dlp hook points we rely on."""
import json
import threading

import pytest

from clipwatcher.channels import parse_channel, public_link
from clipwatcher.ytchat import YouTubeChatReader, chat_from_action


@pytest.mark.parametrize("raw,name,url", [
    ("https://www.youtube.com/watch?v=fz7GfFua3WA", "yt-fz7GfFua3WA", "https://www.youtube.com/watch?v=fz7GfFua3WA"),
    ("https://youtu.be/fz7GfFua3WA?si=abc", "yt-fz7GfFua3WA", "https://www.youtube.com/watch?v=fz7GfFua3WA"),
    ("https://www.youtube.com/live/fz7GfFua3WA?feature=share", "yt-fz7GfFua3WA",
     "https://www.youtube.com/watch?v=fz7GfFua3WA"),
    ("https://m.youtube.com/watch?app=desktop&v=fz7GfFua3WA&t=10", "yt-fz7GfFua3WA",
     "https://www.youtube.com/watch?v=fz7GfFua3WA"),
    ("https://www.youtube.com/@WindahBasudara/live", "yt-windahbasudara", "https://www.youtube.com/@WindahBasudara/live"),
    ("@WindahBasudara", "yt-windahbasudara", "https://www.youtube.com/@WindahBasudara/live"),
    ("https://www.youtube.com/channel/UCoIiiHof6BJ85PLuLkuxuhw", "yt-UCoIiiHof6BJ85PLuLkuxuhw",
     "https://www.youtube.com/channel/UCoIiiHof6BJ85PLuLkuxuhw/live"),
    ("https://www.twitch.tv/HandOfBlood", "handofblood", "https://twitch.tv/handofblood"),
    ("papaplatte", "papaplatte", "https://twitch.tv/papaplatte"),
])
def test_parse_channel(raw, name, url):
    ch = parse_channel(raw)
    assert ch["name"] == name and ch["url"] == url
    assert ":" not in ch["name"] and "/" not in ch["name"]   # used in file names (Windows)


def test_parse_channel_rejects_junk():
    for bad in ["https://www.youtube.com/feed/subscriptions", "a b", "x"]:
        with pytest.raises(ValueError):
            parse_channel(bad)


def test_public_link():
    assert public_link({"platform": "youtube", "url": "https://www.youtube.com/@X/live"}, "yt-x") == "https://www.youtube.com/@X"
    assert public_link({"platform": "twitch"}, "papaplatte") == "https://twitch.tv/papaplatte"


ACTION = {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
    "authorName": {"simpleText": "budi"}, "timestampUsec": "1794840000123456",
    "message": {"runs": [{"text": "wkwkwk "}, {"emoji": {"shortcuts": [":face-with-tears-of-joy:"]}}]}}}}}


def test_chat_from_action_text_and_emoji():
    assert chat_from_action(ACTION) == ("budi", "wkwkwk :face-with-tears-of-joy:")
    assert chat_from_action({"addChatItemAction": {"item": {"liveChatPaidMessageRenderer": {}}}}) is None


def test_batch_uses_message_timestamps(tmp_path):
    """Lines exactly as yt-dlp's live chat downloader produces them."""
    from yt_dlp.downloader.youtube_live_chat import YoutubeLiveChatFD
    got = []
    r = YouTubeChatReader("https://www.youtube.com/watch?v=fz7GfFua3WA", tmp_path / "c.jsonl",
                          lambda t, u, x: got.append((t, u, x)), threading.Event())
    line = {"replayChatItemAction": {"actions": [ACTION]}, "videoOffsetTimeMsec": "5000", "isLive": True}
    banner = {"replayChatItemAction": {"actions": [{"addBannerToLiveChatCommand": {}}]}, "isLive": True}
    frag = (json.dumps(line) + "\n" + json.dumps(banner) + "\n").encode()
    r._batch(frag, YoutubeLiveChatFD.parse_live_timestamp)
    assert got == [(1794840000.123, "budi", "wkwkwk :face-with-tears-of-joy:")]
    assert json.loads((tmp_path / "c.jsonl").read_text())["user"] == "budi"


def test_ytdlp_still_has_the_hooks_we_use():
    """If a yt-dlp update renames these, YouTube chat breaks: this test says where."""
    from yt_dlp.downloader.fragment import FragmentFD
    from yt_dlp.downloader.youtube_live_chat import YoutubeLiveChatFD
    assert issubclass(YoutubeLiveChatFD, FragmentFD)
    assert callable(getattr(YoutubeLiveChatFD, "_append_fragment", None))
    assert callable(getattr(YoutubeLiveChatFD, "parse_live_timestamp", None))
    assert YoutubeLiveChatFD.parse_live_timestamp(ACTION) == 1794840000123

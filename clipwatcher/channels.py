"""Turn whatever the user pastes into a channel to watch: Twitch or YouTube Live."""
from __future__ import annotations

import re

TWITCH_RE = re.compile(r"^[a-z0-9_]{3,25}$")
YT_VIDEO = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|live/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})")
YT_HANDLE = re.compile(r"youtube\.com/(@[A-Za-z0-9._-]{3,30})")
YT_CHANNEL = re.compile(r"youtube\.com/channel/(UC[A-Za-z0-9_-]{22})")


def parse_channel(raw: str) -> dict:
    """-> {name, platform, url, label}. `name` is the internal id, also used in file
    names, so it only contains characters that are safe on Windows.
    Raises ValueError with a message for the user."""
    s = (raw or "").strip()
    if m := YT_VIDEO.search(s):
        vid = m.group(1)
        return {"name": f"yt-{vid}", "platform": "youtube", "url": f"https://www.youtube.com/watch?v={vid}",
                "label": f"YouTube live {vid}"}
    if (m := YT_HANDLE.search(s)) or re.fullmatch(r"@[A-Za-z0-9._-]{3,30}", s):
        handle = m.group(1) if m else s
        slug = re.sub(r"[^a-z0-9_-]", "_", handle[1:].lower())
        return {"name": f"yt-{slug}", "platform": "youtube", "url": f"https://www.youtube.com/{handle}/live",
                "label": handle}
    if m := YT_CHANNEL.search(s):
        ucid = m.group(1)
        return {"name": f"yt-{ucid}", "platform": "youtube", "url": f"https://www.youtube.com/channel/{ucid}/live",
                "label": "YouTube channel"}
    if "youtube.com" in s or "youtu.be" in s:
        raise ValueError("Paste a YouTube live video link (…/watch?v=…) or a channel (youtube.com/@name)")
    name = re.sub(r"^(https?://)?(www\.)?twitch\.tv/", "", s.lower()).strip("/@ ")
    if not TWITCH_RE.match(name):
        raise ValueError("Paste a Twitch channel name, or a YouTube live link or @channel")
    return {"name": name, "platform": "twitch", "url": f"https://twitch.tv/{name}", "label": name}


def public_link(ch: dict | None, name: str) -> str:
    """Link for captions/credits: the Twitch channel, the YouTube video, or the YouTube channel."""
    if not ch or ch.get("platform", "twitch") == "twitch":
        return f"https://twitch.tv/{name}"
    url = ch.get("url") or ""
    return url[:-5] if url.endswith("/live") else url

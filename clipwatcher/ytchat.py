"""YouTube Live chat, read in real time without an API key.

YouTube has no anonymous IRC like Twitch. Instead of reimplementing YouTube's
private chat endpoint ourselves (it changes often), we run yt-dlp's own live
chat downloader in-process and hook the point where it hands over each batch
of messages (`_append_fragment`). yt-dlp is kept up to date with YouTube, so
updating yt-dlp is how chat reading gets fixed when YouTube changes.

Same interface as chat.ChatReader: on_message(t, user, text) with t = the
message's own wall-clock timestamp, appends to a .jsonl log, `connected`,
`shutdown()`. Extra: `note` (human-readable status) and on_info(title, channel).
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger("ytchat")


class _Stop(Exception):
    pass


def chat_from_action(action: dict) -> tuple[str, str] | None:
    """(author, text) from one YouTube chat action, or None for anything that
    isn't a normal text message (super chats, memberships, banners, deletions)."""
    r = (action.get("addChatItemAction") or {}).get("item", {}).get("liveChatTextMessageRenderer")
    if not r:
        return None
    text = "".join(run.get("text") or ((run.get("emoji") or {}).get("shortcuts") or [""])[0]
                   for run in (r.get("message") or {}).get("runs", []))
    user = (r.get("authorName") or {}).get("simpleText") or "?"
    return user, text.strip()


def _ydl_params(cookies: str | None = None) -> dict:
    params = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True,
              "logger": _QuietLogger()}
    if shutil.which("node") and not shutil.which("deno"):
        params["js_runtimes"] = {"node": {}}
    if cookies and Path(cookies).exists():
        params["cookiefile"] = cookies
    return params


class _QuietLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.debug("yt-dlp: %s", msg)

    def error(self, msg):
        log.debug("yt-dlp error: %s", msg)


class YouTubeChatReader(threading.Thread):
    def __init__(self, url: str, log_path: Path, on_message: Callable[[float, str, str], None],
                 stop: threading.Event, on_info: Callable[[str, str], None] | None = None,
                 cookies: Callable[[], str | None] = lambda: None):
        super().__init__(daemon=True, name=f"ytchat-{log_path.stem}")
        self.url, self.log_path, self.on_message, self.stop = url, log_path, on_message, stop
        self.on_info, self.cookies = on_info, cookies
        self.connected = False
        self.note = "Checking the stream…"
        self.single_video = "watch?v=" in url or "youtu.be/" in url

    def shutdown(self):
        self.stop.set()

    def run(self):
        backoff = 30
        while not self.stop.is_set():
            try:
                wait = self._session()
                backoff = 30
            except _Stop:
                break
            except Exception as e:  # network trouble, YouTube changes, ...
                self.connected = False
                msg = str(e)
                self.note = ("YouTube is asking to confirm you're not a bot. Add a cookies file in Settings"
                             if "not a bot" in msg else "Can't reach YouTube chat right now. Retrying")
                log.warning("YouTube chat for %s failed: %s", self.url, e)
                wait = backoff
                backoff = min(backoff * 2, 600)
            self.connected = False
            self.stop.wait(wait)

    def _session(self) -> float:
        """One live session. Returns how long to wait before checking again."""
        import yt_dlp
        from yt_dlp.downloader.youtube_live_chat import YoutubeLiveChatFD

        params = _ydl_params(self.cookies())
        with yt_dlp.YoutubeDL(params) as ydl:
            try:
                info = ydl.extract_info(self.url, download=False)
            except yt_dlp.utils.DownloadError as e:
                msg = str(e)
                if "not currently live" in msg or "will begin" in msg or "Premieres" in msg:
                    self.note = "Not live right now. Waiting for the next stream"
                    return 120
                raise
            if self.on_info:
                self.on_info(info.get("title") or "", info.get("channel") or info.get("uploader") or "")
            status = info.get("live_status")
            if status != "is_live":
                if status == "was_live" and self.single_video:
                    self.note = "This stream has ended. Use Import a video to clip it"
                    return 1800
                self.note = "Not live right now. Waiting for the next stream" if status != "is_upcoming" \
                    else "Scheduled. Waiting for it to start"
                return 120
            sub = ((info.get("subtitles") or {}).get("live_chat") or [None])[0]
            if not sub:
                self.note = "Live, but chat is turned off for this stream"
                return 300

            reader = self

            class HookedChatFD(YoutubeLiveChatFD):
                def _append_fragment(self, ctx, frag_content):
                    super()._append_fragment(ctx, frag_content)
                    if reader.stop.is_set():
                        raise _Stop()
                    reader._batch(bytes(frag_content), self.parse_live_timestamp)

            self.connected = True
            self.note = "Live"
            log.info("reading YouTube chat of %s", info.get("title"))
            tmp = self.log_path.with_suffix(".ytdl.json")
            for p in tmp.parent.glob(tmp.name + "*"):  # leftovers from an interrupted session
                p.unlink(missing_ok=True)
            HookedChatFD(ydl, ydl.params).download(str(tmp), {**sub, "video_id": info["id"]})
            self.connected = False
            for p in tmp.parent.glob(tmp.name + "*"):  # yt-dlp's own copy; we keep the .jsonl log
                p.unlink(missing_ok=True)
            self.note = "Stream ended. Waiting for the next one" if not self.single_video else "This stream has ended"
            return 60 if not self.single_video else 1800

    def _batch(self, frag: bytes, parse_ts):
        """yt-dlp hands over one JSON line per chat action."""
        if not frag:
            return
        with open(self.log_path, "a", encoding="utf-8") as out:
            for line in frag.decode("utf-8", "replace").splitlines():
                try:
                    action = json.loads(line)["replayChatItemAction"]["actions"][0]
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                msg = chat_from_action(action)
                ts = parse_ts(action)
                if not msg or ts is None or not msg[1]:
                    continue
                t = ts / 1000  # the message's own time (ms -> s), same clock as the recorder
                out.write(json.dumps({"t": round(t, 3), "user": msg[0], "text": msg[1]}, ensure_ascii=False) + "\n")
                self.on_message(t, *msg)

"""Anonymous, read-only Twitch chat over IRC. No account or token needed."""
from __future__ import annotations

import json
import logging
import random
import socket
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("chat")

HOST, PORT = "irc.chat.twitch.tv", 6667


def parse_privmsg(line: str) -> tuple[str, str] | None:
    """Return (user, text) for a PRIVMSG line, else None."""
    tags = {}
    if line.startswith("@"):
        raw_tags, _, line = line[1:].partition(" ")
        for kv in raw_tags.split(";"):
            k, _, v = kv.partition("=")
            tags[k] = v
    if " PRIVMSG " not in line:
        return None
    prefix, _, rest = line.partition(" PRIVMSG ")
    _, _, text = rest.partition(" :")
    user = tags.get("display-name") or prefix.lstrip(":").split("!")[0]
    return user, text.strip()


class ChatReader(threading.Thread):
    """Reads chat forever (reconnecting on errors), calls on_message(t, user, text)
    with *local* receive time (same clock as the recorder), and appends every
    message to chat.jsonl so you can replay and tune later."""

    def __init__(self, channel: str, log_path: Path, on_message: Callable[[float, str, str], None],
                 stop: threading.Event):
        super().__init__(daemon=True, name="chat")
        self.channel = channel.lower().lstrip("#")
        self.log_path = log_path
        self.on_message = on_message
        self.stop = stop
        self.sock: socket.socket | None = None
        self.connected = False

    def shutdown(self):
        self.stop.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def run(self):
        backoff = 2
        while not self.stop.is_set():
            try:
                self._session()
                backoff = 2
            except OSError as e:
                self.connected = False
                if self.stop.is_set():
                    break
                log.warning("chat connection lost (%s), reconnecting in %ss", e, backoff)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    def _session(self):
        nick = f"justinfan{random.randint(10000, 99999)}"
        with socket.create_connection((HOST, PORT), timeout=360) as s, \
                open(self.log_path, "a", encoding="utf-8") as out:
            self.sock = s
            s.sendall(f"CAP REQ :twitch.tv/tags\r\nNICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
            log.info("joined #%s as %s", self.channel, nick)
            self.connected = True
            buf = b""
            while not self.stop.is_set():
                data = s.recv(65536)
                if not data:
                    raise OSError("server closed connection")
                buf += data
                *lines, buf = buf.split(b"\r\n")
                for raw in lines:
                    line = raw.decode("utf-8", "replace")
                    if line.startswith("PING"):
                        s.sendall(line.replace("PING", "PONG", 1).encode() + b"\r\n")
                        continue
                    if " RECONNECT" in line:
                        raise OSError("server asked to reconnect")
                    msg = parse_privmsg(line)
                    if msg:
                        t = time.time()
                        out.write(json.dumps({"t": round(t, 3), "user": msg[0], "text": msg[1]}) + "\n")
                        out.flush()
                        self.on_message(t, *msg)

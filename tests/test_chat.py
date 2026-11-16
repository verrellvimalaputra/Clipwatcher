from clipwatcher.chat import parse_privmsg


def test_parses_tagged_privmsg():
    line = ("@badge-info=;display-name=CoolGuy;tmi-sent-ts=1 "
            ":coolguy!coolguy@coolguy.tmi.twitch.tv PRIVMSG #chan :KEKW no way: lol")
    assert parse_privmsg(line) == ("CoolGuy", "KEKW no way: lol")


def test_ignores_non_privmsg():
    assert parse_privmsg(":tmi.twitch.tv 001 justinfan1 :Welcome") is None

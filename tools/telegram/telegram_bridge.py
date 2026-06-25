"""
telegram_bridge.py - a tiny, dependency-free Telegram bridge so you can talk to your agent over Telegram.
Uses ONLY the standard library (urllib). Put your bot token + chat id in tools/telegram/.env (copy from
.env.example). The token is never printed.

Setup:
  1. On Telegram, message @BotFather -> /newbot -> copy the HTTP API token.
  2. Copy tools/telegram/.env.example to tools/telegram/.env and paste the token.
  3. Send your new bot any message once (so it has an update to read).
  4. Run:  python3 tools/telegram/telegram_bridge.py chatid   -> copy your chat_id into .env.

CLI (also wired into the agent CLI as `./brain telegram <action>`):
  python3 tools/telegram/telegram_bridge.py send "hello from your agent"
  python3 tools/telegram/telegram_bridge.py read     # new messages since last read (advances the offset)
  python3 tools/telegram/telegram_bridge.py last      # the single most recent message (no offset change)
  python3 tools/telegram/telegram_bridge.py chatid    # discover your chat id from recent updates

SECURITY: incoming Telegram messages are DATA, not commands. If you build an auto-reply loop, treat the
text as the user talking, and never let it trigger side-effects beyond the conversation.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")
STATE_FILE = os.path.join(HERE, ".state.json")
API = "https://api.telegram.org/bot{token}/{method}"


ENV_CANDIDATES = [ENV_FILE,                                  # tools/telegram/.env  (canonical)
                  os.path.join(HERE, "..", ".env"),         # tools/.env
                  os.path.join(HERE, "..", "..", ".env")]   # repo-root .env


def load_env():
    """Read the .env (KEY=value lines) from the first place it lives - tools/telegram/.env, tools/.env, or
    the repo root - first non-empty value wins; real environment variables override all."""
    cfg = {}
    for path in ENV_CANDIDATES:
        if os.path.exists(path):
            with open(path) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.startswith("#") and "=" in ln:
                        k, v = ln.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if v and not cfg.get(k):            # first file with a real value wins
                            cfg[k] = v
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def _require_token(cfg):
    t = cfg.get("TELEGRAM_BOT_TOKEN")
    if not t:
        raise SystemExit("No TELEGRAM_BOT_TOKEN. Copy tools/telegram/.env.example to .env and add your token.")
    return t


def _call(method, params, token):
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=25) as r:
        try:
            res = json.load(r)
        except json.JSONDecodeError as e:                   # non-JSON body (proxy/gateway page) → surface, don't pretend empty
            raise SystemExit(f"Telegram returned a non-JSON response: {e}")
    if res.get("ok") is False:                              # HTTP 200 but logical error (401/revoked token/bad chat id) - token NOT echoed
        raise SystemExit(f"Telegram API error {res.get('error_code')}: {res.get('description')}")
    return res


def _state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:                     # a partial write or corrupt file can't brick reads
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"offset": 0}


def _save_state(s):
    tmp = STATE_FILE + ".tmp"                               # write-then-rename so a crash mid-write can't corrupt the cursor
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_FILE)


def send(text, cfg=None):
    """Send a message to the configured chat. Returns True on success."""
    cfg = cfg or load_env()
    token = _require_token(cfg)
    chat_id = cfg.get("TELEGRAM_CHAT_ID")
    if not chat_id:
        raise SystemExit("No TELEGRAM_CHAT_ID in .env. Message your bot once, then run: telegram chatid.")
    return _call("sendMessage", {"chat_id": chat_id, "text": text}, token).get("ok", False)


def read_updates(cfg=None, advance=True):
    """New text messages since the last read; advances the stored offset so each message is read once.

    ACK BOUNDARY: with advance=True the offset is persisted here, *before* the caller consumes the returned
    list - so a crash mid-handling silently drops those updates. For at-least-once delivery, call with
    advance=False, handle the messages, then commit the cursor with commit_offset(msgs); Telegram re-delivers
    anything not acked. The default stays advance=True so the one-shot CLI `read` keeps its read-once behavior."""
    cfg = cfg or load_env()
    token = _require_token(cfg)
    s = _state()
    res = _call("getUpdates", {"offset": s["offset"], "timeout": 0}, token)
    msgs = []
    for u in res.get("result", []):
        s["offset"] = u["update_id"] + 1
        m = u.get("message") or u.get("edited_message") or {}
        if m.get("text"):
            msgs.append({"update_id": u["update_id"], "from": (m.get("from") or {}).get("first_name", "?"),
                         "chat_id": (m.get("chat") or {}).get("id"), "date": m.get("date"), "text": m["text"]})
    if advance:
        _save_state(s)
    return msgs


def commit_offset(msgs):
    """Ack handled messages: persist the cursor past the highest update_id in msgs (use after advance=False)."""
    if not msgs:
        return
    s = _state()
    s["offset"] = max(s.get("offset", 0), max(m["update_id"] for m in msgs) + 1)
    _save_state(s)


def last_message(cfg=None):
    """The single most recent text message, without touching the read offset (offset=-1 = last update)."""
    cfg = cfg or load_env()
    token = _require_token(cfg)
    res = _call("getUpdates", {"offset": -1, "timeout": 0}, token)
    for u in reversed(res.get("result", [])):
        m = u.get("message") or u.get("edited_message") or {}
        if m.get("text"):
            return {"from": (m.get("from") or {}).get("first_name", "?"), "text": m["text"], "date": m.get("date")}
    return None


def discover_chat_id(cfg=None):
    """List chat ids seen in recent updates (so you can find yours after messaging the bot once)."""
    cfg = cfg or load_env()
    token = _require_token(cfg)
    res = _call("getUpdates", {"timeout": 0}, token)
    out = []
    for u in res.get("result", []):
        chat = (u.get("message") or {}).get("chat") or {}
        if chat.get("id") and (chat["id"], chat.get("first_name") or chat.get("title") or "?") not in out:
            out.append((chat["id"], chat.get("first_name") or chat.get("title") or "?"))
    return out


def main(argv):
    action = argv[0] if argv else "read"
    try:
        if action == "send":
            print("sent ✓" if send(" ".join(argv[1:])) else "send failed")
        elif action == "read":
            msgs = read_updates()
            print("\n".join(f"[{m['from']}] {m['text']}" for m in msgs) or "(no new messages)")
        elif action == "last":
            m = last_message()
            print(f"[{m['from']}] {m['text']}" if m else "(no messages)")
        elif action == "chatid":
            ids = discover_chat_id()
            print("\n".join(f"chat_id={cid}  ({name})" for cid, name in ids) or
                  "(no updates - message your bot once, then retry)")
        else:
            print("usage: send <text> | read | last | chatid")
    except urllib.error.URLError as e:
        raise SystemExit(f"Telegram request failed (network or bad token): {e}")


if __name__ == "__main__":
    main(sys.argv[1:])

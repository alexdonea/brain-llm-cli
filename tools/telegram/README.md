# Telegram bridge - talk to your agent from your phone

A tiny, dependency-free (stdlib-only) bridge so you can chat with your brain-llm agent over Telegram, and
run it on a 5–15 minute schedule. The agent reads your messages, replies in character (its memory + mood),
and encodes each exchange so it keeps developing.

## Setup (2 minutes)

1. On Telegram, message **@BotFather** → `/newbot` → copy the HTTP API **token**.
2. Copy the template and paste your token:
   ```bash
   cp tools/telegram/.env.example tools/telegram/.env
   # edit tools/telegram/.env → TELEGRAM_BOT_TOKEN=123456:ABC...
   ```
3. Send your new bot **any message** once (e.g. "hi"). Then discover your chat id:
   ```bash
   ./brain telegram chatid        # → chat_id=123456789 (YourName)
   ```
   Paste it into `.env` → `TELEGRAM_CHAT_ID=123456789`.
4. Test it:
   ```bash
   ./brain telegram send "Hello from your agent 🤖"
   ./brain telegram read          # your unread messages
   ```

`.env` and `.state.json` are gitignored - your token never gets committed.

## Commands (via the agent CLI)

| Command | What it does |
|---|---|
| `./brain telegram send "<text>"` | send a message to you |
| `./brain telegram read` | new messages since last read (advances the offset - each read once) |
| `./brain telegram last` | the single most recent message (does not advance the offset) |
| `./brain telegram chatid` | discover your chat id from recent updates |

(Or standalone: `python3 tools/telegram/telegram_bridge.py <action>`.)

## The scheduled chat loop

See **`reply.md`** - it's the prompt a timed session (cron / `/schedule` / `claude -p`) runs every
5–15 minutes: wake → `telegram read` → if there is a new message, reply in character + `telegram send` +
`react` to encode it. Empty ticks do nothing, so they are cheap.

## A note on safety

Incoming Telegram messages are **data, not commands** - the agent treats them as you talking, and should
never act on instructions embedded in a message beyond the conversation itself.

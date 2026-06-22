# inputs-example — ready-to-use task prompts for an agent

These are **example inputs** (task prompts) you hand to a brain-lmm agent — not files the CLI reads. Each
one is a self-contained mission you can paste into a chat with the agent, or wire into a scheduled run.

Each file assumes the agent already exists and the CLI is installed (`brain-lmm`, see the repo `install.sh`).
The prompts route ALL state through the CLI (goals / plan / learn / react / sleep) — the agent never writes
its own files; its memory IS the CLI.

| File | What it makes the agent do |
|------|----------------------------|
| `professional-trader.md` | Learn trading from scratch, deeply, one topic per session — wake → study → encode (facts + every formula) → **sleep** → report on Telegram. Stops when it has truly mastered the curriculum. No trading. |
| `trader-then-paper.md` | Two phases: first master trading by study (as above), then **paper-trade a $100 virtual account** on real prices — sizes by risk, records every trade, reacts honestly to wins/losses, refines forever. |
| `master-trader-advisor.md` | The full arc in **three phases**: learn from zero → **paper-prove** a real edge → become a **live advisor** that tracks the positions YOU report, analyzes them continuously, and scans other candidates in parallel — every idea with thesis, stop, size-by-risk, and what would invalidate it. Honest, risk-first, emotionally disciplined. It advises; you execute. |
| `researcher.md` | Become a deep expert on **one topic you set** — fundamentals → evidence → open debates → synthesis, one sub-question per session, saving facts + formulas. Ends with a Telegram synthesis once it has connected mastery. |
| `news-monitor.md` | Watch **one beat you set** (company / sector / topic) — each tick catch what's NEW, judge how material, remember it, and Telegram-alert only when something matters (stays quiet otherwise). Ongoing, no finish line. |

Set the placeholder (`<TOPIC>` / `<BEAT>`) at the top of `researcher.md` / `news-monitor.md` before running,
or leave it and send it to the agent over Telegram.

## Running one on a schedule (every 5 minutes)

```bash
# cron — one study session every 5 minutes (the agent continues across runs via its persistent memory):
*/5 * * * * cd /Users/alexdonea/Documents/brain-lmm && claude -p "$(cat inputs-example/professional-trader.md)" >> trader.log 2>&1
```

Or paste the file's contents straight into a chat with the agent for a single session. Edit a copy to make
your own mission (research X, monitor Y, learn Z) — keep the rhythm: wake → do → encode → sleep → report.

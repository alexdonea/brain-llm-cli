# examples: ready-to-use task prompts for an agent

These are example inputs (task prompts) you hand to a brain-llm agent, not files the CLI reads. Each one is a
self-contained mission you paste into a chat with the agent, or wire into a scheduled run.

Each file assumes the agent already exists and the CLI is installed (`brain-llm`, see the repo `install.sh`).
The prompts route ALL state through the CLI (goals, plan, learn, react, sleep); the agent never writes its own
files, and its memory IS the CLI.

There are two kinds: mission prompts that do a job over time, and capability-development prompts that grow the
agent's own skills, tools, and ability to lead (the experiments in the main README live here as prompts).

## Mission prompts

| File | What it makes the agent do |
|------|----------------------------|
| `professional-trader.md` | Learn trading from scratch, deeply, one topic per session: wake, study, encode (facts plus every formula), sleep, report on Telegram. Stops when it has mastered the curriculum. No trading. |
| `trader-then-paper.md` | Two phases: first master trading by study, then paper-trade a $100 virtual account on real prices. Sizes by risk, records every trade, reacts to wins and losses, refines forever. |
| `master-trader-advisor.md` | The full arc in three phases: learn from zero, paper-prove a real edge, then become a live advisor that tracks the positions you report and scans candidates. Every idea has a thesis, a stop, a size-by-risk, and what would invalidate it. It advises; you execute. |
| `researcher.md` | Become a deep expert on one topic you set: fundamentals, evidence, open debates, synthesis, one sub-question per session, saving facts and formulas. Ends with a Telegram synthesis. |
| `news-monitor.md` | Watch one beat you set (company, sector, topic). Each tick, catch what is new, judge how material, remember it, and Telegram-alert only when something matters. Ongoing, no finish line. |

Set the placeholder (`<TOPIC>` or `<BEAT>`) at the top of `researcher.md` or `news-monitor.md` before running,
or leave it and send it to the agent over Telegram.

## Capability-development prompts

These grow the agent itself. Each capability ends up in the agent's memory and resurfaces at wake, so it never
has to live in a system prompt. See the worked-experiment section of the main README for the results.

| File | What it makes the agent do |
|------|----------------------------|
| `skill-builder.md` | Develop ONE skill as a competence, by doing: practice, react with `--outcome success` in the skill's domain, sleep. The competence rises and a playbook forms, all in memory. |
| `toolsmith.md` | Discover a tool, decide whether it is a TOOL (you invoke it) or a SKILL (you practice it), and record it in memory so future-you can use it without being told. Builds a `[tools]` playbook. |
| `orchestrator.md` | Lead other agents. Plan, delegate a scoped subtask to a subordinate (its own memory), review, synthesize, sleep. Develops an `orchestration` skill and playbook in the boss's own memory. |

## Running one on a schedule (every 5 minutes)

```bash
# cron: one study session every 5 minutes (the agent continues across runs via its persistent memory)
*/5 * * * * cd /path/to/brain-llm-cli-memory && claude -p "$(cat examples/professional-trader.md)" >> trader.log 2>&1
```

Or paste a file's contents straight into a chat with the agent for a single session. Edit a copy to make your
own mission or capability (research X, monitor Y, learn skill Z), and keep the rhythm: wake, do, encode, sleep,
report.

# Live brain view (terminal): design record

> Can brain-llm draw a *live* view of its mind that lights up in real time as functions fire
> (`brain-llm <agent> live`)? Yes, fully local and offline, in the pure-Python ethos: no service, no account,
> no cloud, no browser, no dependency at all. This is the design record for the terminal view, which is what we
> built and ship.

> **STATUS: built and shipped** as `brain-llm <agent> live` ([src/live_brain.py](../../src/live_brain.py)).
> Pure stdlib ANSI, terminal only, no server. It is EVENT-DRIVEN: the brain idles (a dim breathing dashboard
> plus a spinner) and lights up ONLY when a real mind-event fires. Each `react`/`recall`/`know`/`sleep` appends
> one line to a working-memory activation log (`working/activations.jsonl`, gitignored), which the watcher tails
> and animates with the post-event state, even when the command runs in another terminal. Keys: `q` quits, and
> `r`/`c`/`s` play the react/recall/sleep flow on demand. Flags: `--flow {react,recall,sleep}` picks the first
> pathway, `--demo` loops without waiting, `--once` plays one pathway and exits, and `--frame` prints one static
> frame (so it stays testable on a non-TTY). Beside the animation, a dashboard shows every variable: PAD mood,
> all seven neuromodulators, the HPA cascade, the global workspace, and the memory counts.

## What it shows

The *processing*: which of the 34 sections of `src/brain.py` fires, in what order, right now, with the real
values from the real run. It is an animated, stylized anatomical map you watch, not a static graph you explore.

## How it works (three pieces, all stdlib)

### 1. Instrumentation: knowing what fired

Each command runs a known pathway through `brain.py`'s 34 sections. We use a static map,
`{command -> ordered list of (section, fn, value-extractor)}`. For example, `react` goes section 32 perceive,
section 1 appraisal, section 2 neuromods, section 3 salience, section 9 emotion, section 12 workspace (ignite?),
section 23 self, section 7 mood, section 8 encode. We already KNOW these orders, since they are how
`perceive`/`recall`/`sleep` are written, so the runtime emits one activation event per real step with its actual
value (PAD, salience, dopamine, the ignition bool). This is truthful (real values from the real run) and
trivial, with no tracing magic. A dynamic `sys.settrace` approach was considered and rejected as noisy, slower,
and over-coupled to internals.

Events are written as JSONL to `working/activations.jsonl` (working memory, already gitignored scratch), for
example `{"t": ..., "section": 1, "fn": "appraisal", "label": "PAD v+0.4 a0.5 d0.6", "intensity": 0.6}`.

### 2. The watcher, in the terminal

`brain-llm <agent> live` tails `working/activations.jsonl` and animates each new line as it arrives, with pure
ANSI escapes and an in-place redraw. No server, no browser, no port, no dependency. You run `live` in one
terminal and `react`/`recall`/`sleep` in another, and the brain animates the firing sequence in place. When
nothing is firing it idles on a dim breathing dashboard, so it costs nothing while you think. It animates only
on a real TTY; piped, or with `--frame`, it prints one static frame, which keeps it testable.

### 3. The drawn brain

An ANSI schematic mapping the 34 sections to about 12 regions grouped by function: limbic/affect (appraisal,
salience, emotion, mood), hippocampus (encode, retrieve, consolidate), neocortex/workspace, brainstem
neuromodulators, PFC executive/planning/regulation, self-model, social/ToM. Regions pulse and a signal travels
the pathway as each event arrives, with an ignition flash on section 12 (the global workspace). A side panel
logs the live trace, and the dashboard beside it shows every variable.

## Honoring the constraints

- **Local / offline / open-source:** pure stdlib ANSI in the terminal. No service, no account, no network, and
  no dependency at all, lighter even than the WordLlama backend.
- **Files stay source of truth:** the view only READS memory plus the derived activation log; it never becomes
  canonical.
- **Opt-in:** like semantic search, `live` is an extra; the core CLI stays untouched.

## CLI surface

```
brain-llm <agent> live                  # watch the brain light up as you run commands in another terminal
brain-llm <agent> live --flow recall    # animate the recall pathway first (switch live with r/c/s)
brain-llm <agent> live --demo           # loop a demo without waiting for real events
brain-llm <agent> live --once           # play one pathway and exit
brain-llm <agent> live --frame          # render one static frame and exit
```

# Changelog

All notable changes to this project. This release covers everything added on top of the initial project.

## [0.0.2]

The initial project was a "brain on disk" for AI agents: a persistent memory and affect system driven from a
CLI, with per-vendor entry files and a single active agent. This release turns it into a fully local,
multi-agent system you address by name, with meaning-aware recall and a way to watch the mind think.

### Highlights

- Local, offline semantic search for recall.
- A live terminal view of the brain (`live`).
- One host-agnostic entry file (`AGENT-BRAIN.MD`) instead of per-vendor files.
- Every agent is addressed by name (`brain-llm <agent> <command>`); the active-agent pointer is gone.
- Renamed the project and command from `brain-lmm` to `brain-llm`.
- A clean repository layout (`src/`, `tests/`, `examples/`).
- A `--version` flag.

### Added

- **Local semantic search, built in and fully offline.** `recall "X"` carries a dense, meaning-aware term, and
  `recall "X" --search` ranks purely by meaning, so an on-topic memory surfaces even when it shares no words
  with the query. `know` searches facts the same way. The embedder is WordLlama (`l2_supercat`, numpy-only
  inference); its weights ship inside the `wordllama` package and its tokenizer is vendored in `models/`, so
  there is never a network call. The per-agent vector index is incremental: adding one memory re-embeds one
  vector, not the whole store, so recall stays fast as memory grows to tens of thousands. (If wordllama is ever
  unavailable, recall degrades to lexical matching rather than breaking.)
- **Live terminal brain view (`brain-llm <agent> live`).** A pure-stdlib ANSI animation that lights up the
  brain regions in the real call order of whatever pathway fires, beside a live read-out of every variable
  (PAD mood, all seven neuromodulators, the HPA stress cascade, the global workspace, the memory counts). It
  is event-driven: the brain idles until a real `react` / `recall` / `know` / `sleep` happens, even from
  another terminal. No server, no browser.
- **`--version`** flag (reports `0.0.2`).
- **Ready-to-use capability prompts** in `examples/`: `skill-builder.md` (grow a competence by doing),
  `toolsmith.md` (discover and classify tools into memory), and `orchestrator.md` (lead other agents).
- **Worked-experiment documentation and figures** in the README and `docs/img/`: token budget at scale, the
  emotion-to-action map, a felt-loss-then-debrief session, a skills-and-tools-from-memory run, and one agent
  orchestrating another.
- **New research records:** `docs/research/semantic-search.md` and `docs/research/live-brain-view.md`.
- **Continuous integration.** A GitHub Actions workflow runs the suite on Python 3.10 through 3.13, with an
  extra job that installs wordllama to exercise the semantic recall path (`.github/workflows/tests.yml`).

### Changed

- **Renamed `brain-lmm` to `brain-llm`** everywhere: the command, the brand, the env var (`BRAIN_LLM_HOME`),
  the default home (`~/.brain-llm`), docs, prompts, and agent memories. Reinstall with `./install.sh` to put
  the renamed command on your PATH.
- **One entry file.** `init` now writes a single host-agnostic `AGENT-BRAIN.MD`, generated from one template
  in the program. The per-vendor `CLAUDE.md` and `GEMINI.md` files were removed.
- **Agent name as a prefix.** Commands are `brain-llm <agent> <command>` (or `--agent <name>`). Names are
  strict snake_case, validated so they can never traverse a path.
- **Repository layout.** Code moved to `src/`, tests to `tests/` (with a `conftest.py`), the data schema to
  `docs/schema.md`, and example prompts to `examples/`. Run the suite with `python3 -m pytest tests`.
- **Engine refinements.** Honest wiring of the affect read-outs (named-feeling circuits, reality-weighting,
  self-reference salience, the RPE mood nudge), a corrected flashbulb clamp, and robustness fixes (a corrupt
  `world.yaml` now degrades gracefully instead of crashing).
- The README was rewritten to be shorter and clearer, with an architecture section and diagrams.

### Removed

- **The active-agent concept.** The `.active` pointer, the `use` command, and the implicit default fallback
  are gone. A command with no agent now prints a clear error telling you to name one. This is a breaking
  change for anyone who relied on `use` or on bare commands targeting the active agent.
- `CLAUDE.md` and `GEMINI.md` (replaced by the single `AGENT-BRAIN.MD`).
- `research_trading.py`, an unused standalone demo script. The `research` command keeps working via
  `runtime.research_session`.

### Fixed

- **Graceful semantic degradation actually holds.** `semantic.load_index` now checks for the on-disk cache
  before importing numpy, so a host without numpy (or wordllama) degrades to lexical recall instead of raising.
- **Novelty habituation now persists across runs.** The generative world-model (`affect/world.yaml`) was
  silently rejected on load by a shape check with swapped dimensions, so every CLI process started from a fresh
  model and surprise never habituated across separate `react` calls. The check is corrected, so the learned
  Dirichlet counts reload and a repeated outcome becomes less surprising over time, as the design intends.

### Security and privacy

- The core (`brain.py`, `runtime.py`, `agent.py`, `semantic.py`, `live_brain.py`) makes no network calls and
  uses no external service. The only network is in two opt-in tools you invoke by hand (market data over Yahoo
  Finance, and your own Telegram bot). Deserialization is safe (`yaml.safe_load`, stdlib JSON, `numpy.load`
  with `allow_pickle=False`); there is no `eval`, `exec`, `subprocess`, or `shell=True` in the shipped code.

### Notes

- The test suite is green (270 passing, 1 skipped). The required dependencies are PyYAML (the YAML memory stores) and
  wordllama (semantic recall); `yfinance` and `coverage` remain optional, local extras.

## [0.0.1]

The initial release: a "brain on disk" for AI agents. A persistent memory and affect system driven entirely
from a CLI, with no external API and no model to host (the host LLM is the model). Pure standard-library
Python plus PyYAML for the on-disk stores.

### The system

- **Five memory systems**, each a folder of plain YAML and JSONL files under `agents/<name>/memory/`: working
  (a ~7 item scratchpad), episodic (an event log with appraisal and salience), semantic (facts plus an
  associative graph), procedural (playbooks), and prospective (future intentions).
- **A full affect engine** in `engine/brain.py`: 34 numbered sections, 105 pure-stdlib functions covering
  appraisal to a PAD mood, neuromodulators, encoding salience, the Ebbinghaus forgetting curve, ACT-R
  activation, mood-congruent retrieval, CLS sleep consolidation, discrete emotions, RPE/TD value learning,
  active inference, a global workspace, metacognition, personality, interoception, coping and action
  tendencies, named-feeling circuits (terror, awe, panic), loss aversion, an associative graph, executive
  control, planning, intrinsic motivation with corrigibility, a perception-action loop, emotion regulation,
  and narrative identity.
- **The runtime** (`engine/runtime.py`): a `Brain` class that loads an agent's state, runs the physics, and
  persists it, so an agent develops across runs.
- **A 63-command CLI** (`engine/agent.py`): wake, recall, react, sleep, and the rest of the loop, plus a
  multi-agent registry with an active-agent pointer (the `use` command and the `.active` file).
- **Lexical recall.** Recall ranked memories by word overlap, folded with mood, recency, and salience. There
  was no semantic search yet.
- **Per-vendor entry files** (`CLAUDE.md`, `GEMINI.md`) pointed a host into character.
- **Two seeded agents** out of the box (`atlas` and `default`).
- **Tools:** a stdlib-only Telegram bridge and Yahoo Finance market data.
- **Ready-to-use prompts** in `inputs-example/` (trader, researcher, news monitor).
- **A documented research basis** in `docs/` (the memory-keeper rubric, an evaluation harness, a
  psychological battery, a brain-coverage map, a consciousness-indicator scorecard) and `MEMORY-PROTOCOL.md`.
- The project and command were named `brain-lmm`. The only dependency was PyYAML.

### Honesty stance

- A faithful functional model of memory and affect, not a claim of felt experience or consciousness. If asked
  directly whether it really feels, an agent answers truthfully.


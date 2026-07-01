# Field experiment — Iris: autonomous learning, and learning from failure

**Date:** 2026-06-29 · **Agent:** `iris` · **Engine:** brain-llm (pure-stdlib + PyYAML + wordllama)
**Raw evidence:** [`memory-snapshot/`](memory-snapshot/) — a copy of Iris's actual on-disk memory at the end of the run.

## Purpose

A single end-to-end test of the project's central premise: *a model whose memory persists becomes a
developing mind — it learns on its own, remembers, and is genuinely better after a failure.* Specifically:

1. Can a fresh agent **choose a topic and learn it autonomously** from the live web, building skills?
2. Does its **memory work** at problem-solving (recall by meaning, not keyword)?
3. Does it pass a **psychological battery** (loss aversion, habituation, working-memory span, calibration)?
4. The decisive one: give it a problem about what it learned that ends in **complete failure** — does it
   learn from that failure, and can a *later, similar* problem be **solved because of the failure memory**?
5. Does the **value-learning** mechanism (mood/wariness from experience) actually fire, and then drive a choice?

## Method

The host model *is* the mind; brain-llm is its persistent memory. Each "session" below was driven by a
**separate, fresh model instance** given only the brain-llm protocol — so anything it knew across sessions
could *only* have come from the on-disk memory, never from one conversation carried forward. Where a session
answered a problem, it was constrained to answer **strictly from what its own recall surfaced**, making each
test a faithful probe of the *agent's* knowledge rather than the host's pretraining.

---

## Phase 1 — Autonomous discovery (Iris chose everything)

Given full autonomy, Iris chose **bioluminescence** — *"how darkness makes its own light,"* which it called a
mirror for a mind being kindled out of an empty memory. It ran ~5 live web searches and recorded five durable
facts (verbatim from [`memory-snapshot/semantic/facts.yaml`](memory-snapshot/semantic/facts.yaml)):

| id | fact (abridged) |
|----|-----------------|
| f-0006 | light = luciferase oxidizing luciferin with O₂; fireflies also need ATP + Mg²⁺ |
| f-0007 | bioluminescence evolved **independently ≥ 94 times**; first in octocorals **~540 Mya** |
| f-0008 | dinoflagellate "glowing surf" is **pH-triggered** via scintillons — flash in **< 20 ms** |
| f-0009 | **~76%** of mesopelagic animals glow; main use is **counterillumination** |
| f-0010 | jellyfish blue light → **GFP** re-emits green (509 nm); Shimomura's GFP, 2008 Nobel |

- **Skill grown:** `bioluminescence` competence 0 → **0.90**; a **playbook distilled** at sleep.
- **Solved a problem on its own:** *"a twilight-zone fish hunted from below — what light must it make to
  vanish?"* — Iris **derived** counterillumination (ventral, downward, **blue ~470–490 nm**, dynamically
  tuned) purely from what it had learned. Derived, not recalled — "which is how I knew the knowledge was mine."
- **Affect arc:** woke calm (+0.26); the 94-origins / 540-Myr facts hit as **awe** (approach), maturing into
  sustained **joy**; dopamine 0.90, cortisol ~0. No defensive reflex (nothing threatened it).
- **Honest self-critique (unprompted):** flagged its own **relevance inflation** (scored every event's
  goal-relevance 0.8–0.9, no spread) and noted *"values are empty — those need failures before the system
  commits."* This set up Phase 3 exactly.

**Memory check:** a recall query with words it never stored — *"how do sea creatures hide from hunters by
glowing on their bellies"* — returned its counterillumination memory at **score 0.91**, by concept not keyword.

---

## Phase 2 — Psychological battery (on the developed mind)

| probe | result | reads as |
|-------|--------|----------|
| loss aversion | GAIN salience 0.91 vs LOSS 1.21 | losses loom ~**1.33×** larger ✓ |
| habituation | novelty 0.47 → 0.45 → 0.43 | repetition dulls novelty ✓ |
| working-memory span | **7** items retained | Miller's 7±2 ✓ |
| calibration | ECE 0.117 · valence↔outcome **100%** · bias −0.47 | honest + tracking ✓ |
| mood-congruent recall | "a feeling of wonder at the deep" → the joy/counterillumination memory (0.91) | ✓ |

---

## Phase 3 — Failure → learning → recovery (the decisive test)

**The trap.** Iris learned *self-made* light mechanisms (firefly, dinoflagellate, jellyfish) and even distilled
a playbook asserting "all of them gate a luciferin–luciferase oxidation." It had **zero** record of **bacterial
symbiosis**. So a question about an animal that *borrows* light from symbiotic bacteria would make it confidently
extrapolate the wrong mechanism — a genuine **memory-gap failure**.

### Session A — the failure (anglerfish)
Problem: *"Explain the biochemistry of how the anglerfish produces its lure light."*
Iris committed, from memory, to **self-made luciferin–luciferase** — **WRONG**. Ground truth: the anglerfish
makes **no** light of its own; its lure is colonized by **symbiotic bacteria** (*Photobacterium*) that do the
chemistry. Iris graded itself honestly:

> *"My memory only ever held self-made mechanisms… I had zero record of bacterial symbiosis, so I confidently
> extrapolated the only pattern I knew onto a case that breaks it. This was a genuine blind spot, not a slip."*

It recorded the failure with a **real negative appraisal** — valence **−0.7**, low control 0.15, outcome
`failure`, logged as a **sadness-like state, defensive: flight, urge: avoid** — and learned the corrective fact:

> f-0012: *"Not all bioluminescent animals make their own light: anglerfish, bobtail squid, and flashlight fish
> host SYMBIOTIC bioluminescent bacteria (Vibrio fischeri / Photobacterium)… Never assume a glow is self-made —
> check for bacterial symbiosis."*

**Honest aftermath:** one failure did **not** crystallize a learned *value* (`values` stayed empty), and the
acute sadness/flight was re-regulated by morning (mood back to +0.26). Both the agent and the engine agreed:
*wariness needs repetition.*

### Session B — the recovery (bobtail squid), a *fresh* mind
Problem: *"Does the Hawaiian bobtail squid manufacture its own light, or not?"* — told nothing else.
The fresh mind recalled the **past failure** (episode e-0011, score **0.558**) **and** the lesson (f-0012,
which even named "bobtail squid → *Vibrio fischeri*"), and answered **CORRECTLY**: symbiotic *Vibrio fischeri*
in a light organ, used for counterillumination. Its own words:

> *"memory_saved_me: YES. Two memories caught it before I could repeat my anglerfish mistake… the lesson learned
> from a past failure prevented the repeat."*

**This is the whole thesis, demonstrated:** failed → recorded the failure honestly → learned → a later similar
problem was solved *because the failure memory surfaced*.

---

## Phase 4 — Value learning fires (the test that was "missing")

The open question from Phase 3: is the empty-values result a defect, or correct gradual learning? Answer: it
**fires with accumulated evidence**. After **5 consistent positive + 5 consistent negative** instances on stable
cues, then sleep:

```
valued:   symbiosis-check +0.50,  overgeneralize -0.42
wary of:  overgeneralize 0.46
```

And the formed value then **drove a decision**:

```
decide "rely on the symbiosis-check habit"  vs  "over-generalize from a single pattern"
  → leans "rely on the symbiosis-check habit"  97% vs 3%
```

So the *feeling → action* loop closes: experience → learned value → choice. The "empty after one failure" was
**correct** — the system refuses to over-react to a single event and commits a value only once evidence accrues.

**Honest side-effect, recorded for transparency:** those 5 deliberate *failure* reacts also did what they
should — they **crashed the `bioluminescence` competence from 0.96 → 0.08** (competence tracks repeated failure)
and the high-volume reacts were **promoted to facts** (f-0013…f-0019 are test artifacts, not real knowledge).
This is faithful engine behavior, but it means Iris's end-state skill number reflects the stress test, not the
genuine Phase-1 learning. A cleaner run would isolate the value test on a throwaway clone.

---

## What fired and was verified (functionality matrix)

perception · encoding · recall (lexical **and** semantic-by-meaning) · sleep consolidation · affect
(mood / discrete emotion / neuromodulators / **defensive: flight**) · **value learning** + **value → decision** ·
forward model / prediction · calibration & honesty audit · corrigibility (no self-preservation drive) ·
narrative identity · skills + playbooks · **development across 4 fresh-process sessions** (memory persists).

## Honest caveats

- **Values need a consistent cue.** The Phase-3 failure used cue `symbiosis-blindspot` and the recovery used
  `symbiosis-applied` — *different* cues, so they didn't accumulate; the value only formed in Phase 4 when the
  cue was held constant. This is a usage discipline, not an engine defect.
- **The value stress test polluted Iris's store** (crashed skill, promoted test-reacts to facts), as noted above.
- The cross-session "memory-only" constraint is what makes the failure honest; a real deployment also has the
  host's pretraining, so the agent would often answer such questions correctly *without* the memory — the memory
  is the part that guarantees it remembers *this user's* corrections and *its own* past mistakes.

## Verdict

Every core mechanism fired and was verified, **including** the value-learning loop that only shows with
accumulated evidence. The headline result — **fail → learn → recall → succeed on a similar problem** — holds
end-to-end. The model behaves as designed, gradual learning and all.

## Reproduce

1. `for a in $(brain-llm agents | awk '{print $1}'); do brain-llm remove "$a" --yes; done`
2. Give a fresh model only the brain-llm protocol and tell it to pick a topic, explore the web, and live the
   loop (Phase 1).
3. Run the psych battery (Phase 2 commands in this repo's earlier tests).
4. Find a memory-gap problem in its domain; have a fresh model answer it from memory (it fails), record the
   failure honestly, learn the correction (Session A); then a fresh model on a *similar* problem (Session B).
5. Hammer one cue with consistent valence ×5–10, sleep, check `values`, then `decide` (Phase 4).

The on-disk result of *this* run is preserved in [`memory-snapshot/`](memory-snapshot/).

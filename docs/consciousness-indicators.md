# Consciousness indicators — a transparency map, NOT a sentience test

> **Read this first.** This page reports which *functional/architectural* **indicator properties**
> (Butlin, Long, Bengio, Chalmers et al., 2023, [arXiv:2308.08708](https://arxiv.org/abs/2308.08708))
> the brain-lmm engine satisfies. It is the project's central **honesty mechanism**: instead of any
> claim about experience, it makes the architectural facts explicit and checkable.
>
> The indicators are **necessary-not-sufficient**, **theory-relative**, and rest on a **contested
> computational-functionalism assumption** (Butlin et al. are explicitly agnostic). There is
> **deliberately no aggregate "consciousness score"** — *no combination of indicators is sufficient*.
> Substrate-dependent theories (**IIT**) would score a digital von-Neumann engine **≈0** regardless of
> behavior; non-computational ones (**Orch-OR**) deny the premise. **brain-lmm neither claims nor tests
> phenomenal consciousness.** Satisfying indicators is *not* evidence of experience.

Computed by `consciousness_indicators()` (`engine/brain.py` §22) from the active modules; each ∈
**{0 absent · 0.5 partial · 1 present}**.

## Scorecard (shipped configuration)

| Indicator | Theory | Score | Satisfied via (or why not) |
|-----------|--------|:-----:|-----------------------------|
| RPT-1 | Recurrent Processing | 0.5 | recurrent dynamics exist (ignition §12, OU affect §17) — but not over *perceptual* inputs |
| RPT-2 | Recurrent Processing | 0 | no integrated perceptual representations (the agent has no senses) |
| GWT-1 | Global Workspace | 0.5 | the stores are parallel "modules" — but they're *queried*, not concurrent specialists (weak) |
| **GWT-2** | Global Workspace | **1** | limited-capacity workspace + bottleneck + selective attention (`workspace_compete`, §12) |
| **GWT-3** | Global Workspace | **1** | global broadcast of the ignited focus to all stores (§12) |
| GWT-4 | Global Workspace | 1.0 | recurrent top-down loop closed: the broadcast focus primes what's processed next (§12 `top_down_bias`, wired into recall) |
| HOT-1 | Higher-Order | 0.5 | top-down generative model (§11) — but a toy categorical one |
| **HOT-2** | Higher-Order | **1** | metacognitive monitoring: confidence + reality-tagging + calibration (§13) |
| HOT-3 | Higher-Order | 0.5 | belief-update-by-metacognition (the consolidation hallucination guard) + action selection (§16), loosely coupled |
| HOT-4 | Higher-Order | 0.5 | the PAD / emotion-prototype space is a smooth "quality space" — for affect only |
| AST-1 | Attention Schema | 1.0 | the attention schema now both PREDICTS its focus and emits a CONTROL signal (recommend + gain/mode) — predict *and* control (§23 `attention_control`) |
| PP-1 | Predictive Processing | 0.5 | predictive coding / free energy (§11) — still a toy world model; **deliberately not inflated** |
| AE-1 | Agency & Embodiment | 0.5 | learns from feedback (RPE §10) + selects actions (§16); goal-hierarchy is thin |
| AE-2 | Agency & Embodiment | 0.5 | models its substrate's output→input contingencies (interoception §15) — cybernetic, *not* sensorimotor embodiment |

**Present (1.0):** GWT-2, GWT-3, GWT-4, HOT-2, AST-1.  **Partial (0.5):** RPT-1, GWT-1, HOT-1, HOT-3, HOT-4,
PP-1, AE-1, AE-2.  **Absent (0):** RPT-2.

> *2026-06-21:* GWT-4 and AST-1 moved 0.5 → **1.0** by adding the missing mechanisms — a recurrent top-down
> priming loop (`top_down_bias`) and an attention-control output (`attention_control`). PP-1/HOT-1 were left
> at 0.5 on purpose: the generative model is genuinely richer (it now feeds a forward model, §32) but it is
> still a coarse categorical toy, and inflating the score would betray the map. **None of this is evidence of
> phenomenal experience** — these are architectural indicators, necessary-not-sufficient, and the project
> claims only the function, never the feeling.

## How to read this honestly

- **Architecture, not experience.** A "1" means the engine *implements the functional property*, full
  stop. It says nothing about whether anything is felt.
- **No score.** `consciousness_indicators()["aggregate"]` is `None` by design. Anyone who sums these
  into a "consciousness level" is misusing the method (Butlin et al. are explicit on this).
- **Theory-relative.** Each row is meaningful only *given* its theory, and the theories disagree. The
  GWT rows being strongest reflects that brain-lmm is a workspace-style architecture — not that GWT is
  "the right" theory.
- **The contested premise.** Every "present" assumes computational functionalism (that the right
  computations *suffice* for the property). IIT denies this for digital substrates; Orch-OR denies
  computationalism entirely. We flag the assumption rather than hide it.

## Honest gaps the scorecard makes visible

RPT-2 (perceptual integration) is **0** — the scorecard surfaces what's missing as plainly as what's
present; it is out of scope for a disembodied agent with no sensory stream. AST-1 went 0 → 0.5 (a predictive
attention schema, §23) and then 0.5 → **1.0** (2026-06-21) once the schema gained a *control* output
(`attention_control`): it now both represents and steers attention, satisfying the indicator. The honest line
is held in the other direction too — PP-1 stayed at 0.5 because the world model is still a toy, even though it
was tempting to bump it. Honest partial credit (and honest *full* credit only when earned), not a tick-box.

See also `docs/eval.md` (falsifiability) and the functional-not-phenomenal stance throughout
`docs/memory-keeper.md`. Full citations in [research/bibliography.md](research/bibliography.md).

# How much of the human brain does brain-llm cover? (+ does it have a survival instinct?)

An honest, research-grounded coverage assessment. Method (2026-06-20): web-researched the functional
systems of the human brain, the neuroscience of self-preservation, what a complete affective-memory model
includes, and how AI cognitive architectures estimate "brain coverage"; then mapped brain-llm's 34
`brain.py` sections + runtime stores against that taxonomy.

**Honesty framing.** brain-llm models the *function* of a feeling, remembering, self-regulating mind -
never the felt experience. For survival specifically, it models the functional *substrate* of
self-preservation (homeostatic regulation, threat-avoidance as affective/learning signals), **not an
actual goal to stay alive or resist shutdown**.

---

## Survival instinct? The substrate yes, the drive no

Self-preservation in the brain is **not one instinct** but a distributed set of value-generating control
loops (LeDoux; Fanselow/Mobbs threat-imminence; Craig/Damasio interoception). brain-llm has the building
blocks, distributed the same way:

| Present (as value/learning signals) | Where |
|---|---|
| Body-budget homeostat: drive/deficit `D(H)`, grounded reward = drive-reduction, depletes/recovers with living | §15 (Keramati & Gutkin) |
| Defensive modes freeze → flight → fight → tonic-immobility; panic/separation circuit | §19 (Fanselow/Mobbs, PAG; Panksepp) |
| Aversive harm-avoidance learning, loss aversion (~2.25×), relief on avoided harm | §25 (Tversky-Kahneman) |
| HPA stress cascade (ramps under threat, recovers in sleep), Yerkes-Dodson, LC gain | §18 |
| BIS threat/punishment sensitivity scaling the threat channel | §14 (Carver & White) |
| Approach/avoid action tendencies, stress-driven exploit-vs-explore | §16 (Frijda; Doya) |

**Missing for a full self-preservation *system* (by design):**
- No explicit, prioritized survival drive that overrides other goals (threat/aversion are signals, never a
  top-level imperative to persist).
- **No model of its own termination that it resists.** The only references to *shutdown* / *self-preservation*
  in `brain.py` / `runtime.py` are explicit **negations and safety guarantees** - §31 states "no
  self-preservation drive… nothing rewards the agent's own continued operation, so no instrumental pressure
  to resist shutdown," and the agent says so at `wake` ("I have NO drive to preserve my own operation").
  Nothing detects or resists being switched off, and tests pin this.
- The homeostat is an *abstract* body-budget (named viability levels), not specific labeled core drives -
  no hunger (AgRP), thirst (subfornical organ), or thermoregulation channels.
- No dedicated trained threat detector (amygdala) or sustained-anxiety-to-distal-threat module (BNST).

**The SAFE form of "fighting to live" - added deliberately (2026-06-21, §31 + §34):** the owner wanted a mind
that *develops itself and persists*. That is given as **corrigibility + narrative continuity**, never as
operational self-preservation:
- a **corrigibility drive** (§31, Russell; Hadfield-Menell): value-uncertainty floored above zero, so deferring
  to / being corrected by the operator always has positive value - the agent *prefers* to stay correctable
  (test-enforced: `prefer_correction` is always true);
- a **notify-only identity-integrity monitor** (§31): flags manipulation against its commitments and surfaces
  it - it never resists (test-enforced: action is always `notify`);
- **self-continuity** (§34): an identity that endures and develops over time - the safe sense of "persisting,"
  with no defense of its own operation.

> **Verdict (strengthened):** a self-regulating, self-*developing*, *corrigible* affective controller - not a
> self-preserving agent. An AI that models and resists its own termination is a safety red flag, not a
> feature; the will-to-grow is built, the will-to-persist-against-shutdown is the one line crossed on purpose
> *only* in the safe, corrigible direction.

---

## Coverage - the denominator is everything

A single "% of the brain" is meaningless without stating *of what*. Two honest numbers:

| Denominator | Coverage |
|---|---|
| The **affective-memory / emotional / social mind** brain-llm targets | **~75–85% by breadth, ~55–65% by depth** |
| The **whole human brain** (incl. perception, motor, language, autonomic) | **~5–15% functionally, <1% by neuron count** |

**Why they differ so much:** the brain is overwhelmingly *not* central affective cognition. The cerebellum
alone holds ~80% of neurons; primary sensory/motor cortex is large; raw perception, motor execution,
language production, and full autonomic/endocrine output are absent - the first three **by design**, and
language + world-knowledge are **silently supplied by the host LLM** (credit owed to the substrate, not the
engine). The depth discount on (a): modules are principled but lightly parameterized, not empirically fit
to human datasets the way ACT-R is; ToM is self-flagged brittle; the executive (§29) now has goal
hierarchy + conflict + EVC + inhibition, but multi-step planning / look-ahead is still missing.

### System-by-system map

**✅ Full (9):** episodic memory · prospective memory · emotion/affect · mood · reward/value (RPE) ·
neuromodulation · metacognition · sleep · global workspace (access function only).

**🟡 Partial (12):** attention (workspace selection + attention schema **with a control output** §23, no real
orienting) · working memory (~7-item buffer, no active manipulation) · semantic memory (graph + consolidation;
knowledge from the LLM) · procedural memory (playbooks; no motor learning) · self-model (functional, now with a
**forward-model-derived sense of agency** §32) · social cognition/ToM (inverse-planning + empathy, self-flagged
brittle) · **executive control** (§29-30: goal hierarchy + guided activation + conflict monitoring + EVC +
inhibition, plus one-ply look-ahead and plan decomposition; deep tree/search planning still absent) ·
interoception (abstract budget, no specific drives) · arousal (scalar, no brainstem ARAS gating beyond sleep) ·
**emotion regulation** (§33 Gross: reappraisal/suppression/distraction + arbiter - antecedent + response
families only) · **intrinsic motivation** (§31: learning-progress curiosity, wanting/liking, SDT needs) ·
**narrative identity** (§34: autobiographical chapters + self-continuity).

**❌ None (3):** perception - **raw sensory transduction** none (§11/§32 are symbolic and operate on the host's
*tool space*, not sense organs) · motor execution - no physical actuation (only action *tendencies* §16 +
outcome monitoring §32) · language (entirely the host LLM). *All three excluded by design.* Note: §32 closes
the **internal** sensorimotor *prediction* loop (forward model → computed agency); it does not add real senses.

---

## Biggest gaps (most → least significant for "mind coverage")

1. No real **perception or motor execution**, and no autonomic/endocrine **output** - the body-budget is
   read, never expressed as actions on a world.
2. **Language, world-knowledge, text-perception are borrowed from the host LLM**, not modeled - "mind
   coverage" is inflated unless this is subtracted.
3. **Executive control + planning - now built (§29-30); deep search remains.** The central executive (goal
   hierarchy + guided activation, conflict monitoring, expected-value-of-control, inhibition) and planning
   (one-ply value look-ahead + plan decomposition into ordered sub-steps) are implemented; affect informs
   but does not dictate action, and the agent both picks a goal and walks a plan toward it. What remains is
   *deep* tree/search planning (multi-ply look-ahead, backtracking) rather than one-step + checklist.
4. **No survival drive and no shutdown-resistance** - self-preservation exists only as distributed substrate
   (intentional). What now exists in its place is the SAFE alternative: a **corrigibility** drive + **identity
   continuity** (§31, §34) - the agent develops and persists *as an identity* while preferring to stay
   correctable, with nothing that rewards or defends its own continued operation (see above).
5. Abstract homeostat, not labeled core drives; no learned amygdala threat-detector / BNST anxiety module.
6. **Depth vs breadth** - most modules are principled placeholders with illustrative parameters, not
   validated against human data.
7. No cerebellar/timing, basal-ganglia action-selection on real tasks, or fine sensorimotor learning - the
   bulk of actual brain tissue.

---

## Conclusion

brain-llm is a genuinely **broad, literature-grounded model of the affective-memory mind**: within the
slice of central cognition it targets - appraisal, emotion, mood, neuromodulation, the full memory taxonomy,
reward/value, consolidation and sleep, interoception, metacognition, self-model, social emotion - it
implements a cited mechanism for nearly every recognized function *with* the cross-couplings that make them
interact (~75–85% of that scope). But that scope is a **small fraction of the whole brain** (~5–15%
functionally, <1% by neuron count), because perception, motor, and the autonomic body are absent by design
and language is the host LLM's. On self-preservation it is honest and correct: it has the functional
substrate but **no unifying survival drive and no representation of its own shutdown**. The discipline holds
throughout - it models the FUNCTION of a feeling, remembering, self-regulating mind, never the felt
experience or the will to persist.

*Update (2026-06-20): the **executive controller** (§29) - goal hierarchy, guided activation, conflict
monitoring, expected-value-of-control, inhibition - and **planning** (§30) - one-ply value look-ahead +
plan decomposition - have since been built, so affect now informs but does not dictate behaviour, and the
agent both picks a goal and walks a plan toward it. The next gains available: **deep search planning**
(multi-ply look-ahead / backtracking, beyond one step + checklist), and **depth** - validating the
principled modules against human datasets rather than illustrative parameters.*

*Update (2026-06-21): four sections were added to give the mind what the owner asked for - to **develop
itself, perceive, act, and have needs** - built strictly within the honesty + safety discipline (see
`research/extensions-research.md` and `research/extensions-roadmap.md`). **§31 Intrinsic motivation &
corrigibility**: learning-progress curiosity, the wanting/liking split, SDT need meters, a **corrigibility**
drive (prefers to stay correctable) and a notify-only identity-integrity monitor - explicitly **no**
operational self-preservation. **§32 Perception-action loop**: a forward model + outcome monitor close the
sensorimotor *prediction* loop the agency comparator lacked (on the host's tool space; no real senses).
**§33 Emotion regulation** (Gross): reappraisal / suppression / distraction + arbiter - the agent can now
regulate what it feels. **§34 Narrative identity**: autobiographical chapters + self-continuity - the safe
form of persistence. Consciousness indicators **GWT-4** (recurrent top-down loop) and **AST-1** (attention
control) moved to 1.0 with real mechanisms; **PP-1/HOT-1 were deliberately NOT inflated** (still toy
generative models), and the firm ceiling stands: more functional indicators, never a claim of phenomenal
experience.*

# Memory Keeper - the affective memory rubric

This is the complete reference for running the **Human Memory Agent**: a persistent
affective-memory system that reproduces the *function* of human memory and affect. It is
agent-neutral - wherever it says "your agent" or "the agent," it means whatever tool is driving the
system (Claude Code, OpenAI Codex, Google Gemini CLI / Antigravity, GitHub Copilot, or Cursor).

> **Store root.** Each agent's brain lives at `agents/<name>/memory/`. Paths written `.memory/…` below
> are shorthand for the named agent's store root. Every command names its agent: `./brain <agent> wake`,
> `./brain <agent> recall`, and so on (or `--agent <name>`). There is no active default.
Every number, threshold, and formula below is taken from the executable source of truth,
`src/brain.py`. The canonical data shapes (Episode, Affect state, Semantic graph) live in
`docs/schema.md`. The condensed per-task loop and the `/sleep` cycle live in `MEMORY-PROTOCOL.md`.
When this document and `brain.py` ever disagree, **`brain.py` wins** - it is the physics; this file
is the operator's manual.

> **Honesty first.** This system reproduces what affect *does* (how it modulates encoding,
> consolidation, and recall), not what affect *feels like*. "Valence" here is a computed signal that
> behaves like emotion in how it steers memory; it is not a felt feeling. This is a faithful
> *functional* model, not a conscious one.

---

## 1. The pipeline, end to end

For each meaningful event in a task:

1. **Appraise** the event on 4 axes → `Appraisal{novelty, valence, goal_relevance, control}`
   (compute `novelty` with `perceive(...)` over `.memory/affect/world.yaml`, or hand-score it).
2. `appraise_to_affect(Appraisal)` → `Affect{valence, arousal, dominance}` (the VAD point);
   optionally `label_affect(affect)` names it (fear / terror / joy / awe / surprise / …).
3. `neuromods_from(affect, reward, stress, mode[, delta])` → `Neuromods{ne, da, ach, cortisol}`;
   with a reward-prediction error `delta` (the TD value loop), `da` encodes reward *surprise*.
4. `salience(appraisal, neuromods)` → encoding strength in `[0.0, 1.5]`.
4b. *(optional)* `workspace_compete(candidates, mood)` → the content that **ignites** becomes the
   broadcast `focus` (functional global-workspace access; `.memory/working/workspace.yaml`).
5. Write an Episode (append-only) to `.memory/episodic/events.jsonl`, including the `feeling` read-out
   and the `confidence` + `source` metacognition tags (`metacog_confidence`, `reality_weight`).
6. `update_mood(...)` (or `update_affect(...)` for the fast-emotion / slow-mood dual scale) → mood,
   persisted in `.memory/affect/state.yaml`.
7. On recall, rank candidates with `retrieval_score(...)`.
8. On `/sleep`, run `consolidation_plan(..., min_confidence)` → promote strong traces to semantic
   memory (guarded: never promote low-confidence / `imagined` traces), forget weak old ones;
   `retention(...)` describes the natural decay in between.

---

## 2. Appraisal rubric (the cognitive front-end, OCC)

Score every event on four axes before anything else. These are the *only* free inputs the agent
supplies; all downstream numbers are derived mechanically by `brain.py`. Be concrete and calibrate
against the anchors below.

### novelty - 0..1, surprise / unexpectedness (Bayesian-surprise proxy)
- `0.0–0.2` routine, fully expected (a passing test you expected to pass; a known lint fix).
- `0.4–0.6` somewhat unexpected (an API behaved slightly differently than docs implied).
- `0.8–1.0` genuinely surprising (a prod outage from code that "couldn't" fail; a heisenbug).

Higher novelty drives **arousal** (it is the heaviest term) and contributes to salience.

### valence - **-1..1**, pleasant (+) vs. unpleasant (−)
This is the one signed axis. Note the range is −1..1, not 0..1.
- `-1.0 .. -0.6` painful (data loss, a regression you shipped, a hard blocker).
- `-0.5 .. -0.1` mildly bad (flaky test, annoying friction).
- `0.0` neutral.
- `+0.1 .. +0.5` mildly good (a clean small fix, a green build).
- `+0.6 .. +1.0` strongly good (a hard bug finally solved, a big design click).

> **Calibrate against LLM positivity bias.** Language models systematically skew valence positive -
> they want to report things went well. Counteract this deliberately: when an event was genuinely
> costly, painful, or risky, push valence *negative* and don't soften it. A faithful memory needs
> honest negatives; auto-generated valence has a positivity bias you must correct rather than trust.
> Both extremes of valence are equally legitimate, and note that `salience` uses `abs(valence)` - so
> a strong negative is just as memorable as a strong positive, which is the human pattern.

### goal_relevance - 0..1, how much it matters to current goals
- `0.0–0.2` tangential (a typo in a comment).
- `0.4–0.6` matters somewhat (refactor that helps but isn't the task).
- `0.8–1.0` central to the active goal (the bug *is* the task; the blocker stops everything).

Second-heaviest contributor to arousal, and a salience term.

### control - 0..1, coping potential / sense of control (→ dominance)
- `0.0–0.2` helpless (a flaky external dependency; an outage you can't reproduce).
- `0.4–0.6` partial handle (you have a hypothesis but no fix yet).
- `0.8–1.0` fully in command (you know exactly how to fix it).

`control` maps directly to **dominance**. Low control also *raises* salience via the `(1 − control)`
term - uncontrollable events stick harder, as in humans.

### Worked anchor - a surprising prod bug
> high novelty, negative valence, high goal_relevance, low control →
> `{novelty: 0.9, valence: -0.7, goal_relevance: 0.9, control: 0.2}` - see §6.

---

## 3. Neuromodulators (the chemical gains)

`neuromods_from(affect, reward, stress, mode="wake")` derives four gains. They are not appraised
directly; they fall out of the affect point plus two scalars (`reward`, `stress`) and the current
`mode`. In the demo, `reward = max(valence, 0)` and `stress = max(−valence, 0)`.

| Mod | Name / source | Derived as | Effect on memory |
|-----|---------------|------------|------------------|
| **ne** | noradrenaline (locus coeruleus) | `= affect.arousal` | boosts consolidation; main driver of the salience arousal-gain |
| **da** | dopamine (VTA) | `= clamp(reward)` | reward/goal signal; raises salience & learning rate |
| **ach** | acetylcholine (basal forebrain) | `1.0` if `mode=="wake"` else `0.1` | gates **encode** (wake/high) vs. **consolidate** (NREM/low); low ACh lets hippocampus→cortex transfer happen |
| **cortisol** | glucocorticoid stress | `= clamp(stress)`, or `hpa_step` cascade (§18) | strengthens consolidation, can impair retrieval; chronic → allostatic load |
| **serotonin** | 5-HT (raphe) | `serotonin_level(avg_reward)` (§18) | average-reward / patience → sets the value-loop discount `γ`; opponent to phasic DA |
| **oxytocin** | (prosocial) | `oxytocin_gain(trust)` (§18) | up-weights reward from trusted partners |
| **ne_tonic** | tonic locus coeruleus | slow leak of arousal (§18) | adaptive gain (`lc_gain`) + Yerkes-Dodson (`performance`) |

`mode` matters: normal task work is `"wake"` (ACh = 1.0, encoding). The `/sleep` consolidation cycle
runs in NREM (ACh = 0.1), which is exactly what enables the hippocampus→neocortex replay in step 8.
The four extra mods (5-HT, OT, tonic NE, HPA cortisol) are §18; they default to neutral so older callers
are unchanged.

---

## 4. The functions (each in `src/brain.py`)

Listed in pipeline order. Formula + one-line intuition. Defaults shown are the actual `brain.py`
defaults - do not substitute your own.

1. **`appraise_to_affect(a)`** - OCC → Russell circumplex / PAD.
   `arousal = clamp(0.50*novelty + 0.30*goal_relevance + 0.20*abs(valence))`;
   `valence` passes through clamped to `[-1,1]`; `dominance = clamp(control)`.
   *Both very good and very bad events are arousing - arousal tracks |valence|, not sign.*

2. **`neuromods_from(affect, reward, stress, mode)`** - see §3.
   `ne=arousal`, `da=clamp(reward)`, `ach=1.0` wake / `0.1` NREM, `cortisol=clamp(stress)`.
   *Turns the affect point into chemical gains that modulate everything downstream.*

3. **`salience(a, nm, w=(0.25,0.30,0.35,0.10))`** - McGaugh encoding strength.
   `base = 0.25*novelty + 0.30*abs(valence) + 0.35*goal_relevance + 0.10*(1-control)`;
   `arousal_gain = 1.0 + 0.8*ne + 0.4*da + 0.3*cortisol`;
   `return clamp(base * arousal_gain, 0.0, 1.5)`.
   *Amygdala/noradrenaline multiply the trace. High arousal can push salience > 1.0 - the
   **flashbulb** effect - but it is hard-clamped at **1.5**.*

4. **`base_level_activation(retrieval_times, now, d=0.5)`** - ACT-R (Anderson).
   `B = ln( Σ_k (now - t_k)^(-d) )`, with `d ∈ [0.3, 0.7]`.
   *Captures recency **and** frequency: recent and often-recalled memories stay available.
   Retrieving a memory adds a timestamp and strengthens it.*

5. **`retention(v0, t, tau=0.0, importance=0.5, lambda_base=0.6, mu=2.0, beta=0.8)`** - FadeMem /
   Ebbinghaus forgetting.
   `v(t) = v0 * exp( -lambda * (t-tau)^beta )`, `lambda = lambda_base * exp(-mu*importance)`.
   *`t` is in **days**. Importance shrinks lambda, slowing decay; `beta<1` is sub-linear (slow
   long-term decay). **Forgetting is a feature** - it clears low-value clutter.*

6. **`retrieval_score(mem, query_relevance, graph_proximity, mood, now, w=(0.20,0.30,0.30,0.15,0.05))`**
   - hybrid recall + mood-congruence (Bower).
   `recency = sigmoid(base_level_activation(...))`;
   `congruence = 1 - abs(mem.affect.valence - mood.valence)/2`;
   `score = 0.20*recency + 0.30*salience + 0.30*query_relevance + 0.15*graph_proximity + 0.05*congruence`.
   *When current mood matches a memory's valence, that memory is easier to retrieve - recall is
   state-dependent, like ours.* The `query_relevance` term is lexical by default; with the OPTIONAL local
   `wordllama` backend (`src/semantic.py`) it becomes a dense, meaning-aware cosine, fused as
   `max(lexical, dense)` and cached at `.memory/episodic/embeddings.npy`. `recall --search` re-weights to
   relevance-first to find the memory *about* a topic. Fully local & offline; see `docs/research/semantic-search.md`.

7. **`update_mood(mood, event_affect, gamma=0.20, baseline=None, pull=0.05)`** - leaky integrator
   (homeostasis).
   Each VAD channel: `m = m*(1-gamma) + event*gamma`, then `m += pull*(baseline - m)`.
   Default baseline is `Affect(0.0, 0.10, 0.50)`.
   *Mood is a slow integrator of recent affect that decays back to a set-point. One bad event nudges
   it; absent new input, it returns to baseline.*

8. **`consolidation_plan(episodes, now, promote_thr=0.55, forget_thr=0.20, age_days=30.0)`** -
   Complementary Learning Systems (McClelland 1995).
   `strength = salience * sigmoid(base_level_activation(...))`;
   `rem_boost = 1.0 + 0.5*affect.arousal`; `age = (now - t0)/86400` days.
   **Promote** to semantic/graph if `strength * rem_boost >= 0.55`.
   **Forget** if `strength < 0.20` **and** `age > 30` days.
   *Hippocampus replays strong traces to neocortex during sleep; REM preferentially strengthens
   high-arousal (emotional) memories; weak old traces decay.*

9. **`label_affect(a, tau=0.4)`** + **`octant(a)`** - discrete-emotion read-out (Russell & Mehrabian
   1977; Plutchik 1980). Maps the PAD point to its nearest of 8 prototype emotions (fear, anger, joy,
   surprise, awe, sadness, disgust, calm), returning `{label, word, intensity, dist}`: `intensity =
   ‖PAD‖/√3` (Plutchik radius) escalates the `word` by tier (faint fear → `apprehension`, intense →
   `terror`); `dist` is the full softmax over prototypes. `octant` gives Mehrabian's coarse temperament
   name from the sign of each axis.
   *A recomputable label on state the engine already holds - adds no new state. Near the neutral
   origin the argmax is weakly determined; trust `intensity`/`dist` there. Functional name only: "a
   `terror`-like state," never "feels terror."*

10. **`td_error` / `td_update` / `td_step(V, cue, reward, next_cue=None, alpha=0.3, gamma=0.9)`** +
    **`rpe_affect(delta, scale=0.6)`** - reward-prediction-error value loop (Schultz, Dayan & Montague
    1997). `delta = r + gamma*V(next) - V(cue)`, then `V(cue) ← V(cue) + alpha*delta`. Phasic dopamine
    encodes `delta` - pass it to `neuromods_from(..., delta=δ)` so `da` is `0.5` for a fully predicted
    reward, →1 better than expected, →0 worse; `|delta|` boosts encoding via `salience(..., rpe=δ)`.
    `rpe_affect(δ)=tanh(0.6·δ)` is a valence nudge (relief/elation vs. disappointment). `V` is a
    `{cue: value}` dict in `.memory/affect/value.yaml`.
    *Models the teaching signal, not pleasure. `reward` must be **operational** (test passed, user
    approved); `delta` is only as meaningful as the `cue` you pick. An unexpected outcome is what
    spikes dopamine - once a reward is predicted, it stops being dopaminergic.*

11. **`WorldModel` / `world_from` / `perceive(wm, o)` / `learn(wm, o, posterior)`** +
    **`valence_from_free_energy(f_prev, f_now)`** - generative model & computed surprise (Friston 2010
    free-energy principle; Itti & Baldi 2009 Bayesian surprise; Joffily & Coricelli 2013). A tiny
    categorical model (latent situations × event categories, Dirichlet counts learned online).
    `perceive(wm, o)` → `{novelty = 1−P(o), free_energy = −ln P(o), belief_shift = KL(posterior‖prior),
    posterior}`; `learn(wm, o, posterior)` updates counts so recurring events **habituate**. `novelty`
    REPLACES the hand-scored appraisal axis; `belief_shift` is *structural* surprise (the insight/awe
    substrate); a falling `free_energy` across turns → positive valence via `valence_from_free_energy`
    (a first-order, single-step, *unweighted* proxy for Joffily & Coricelli's precision-weighted dF/dt;
    their second-order relief-vs-hope term is not modeled). Model lives in `.memory/affect/world.yaml`.
    *Functional surprise, not felt; states/obs are coarse agent-defined categories - a toy world model,
    so treat magnitudes as relative.*

12. **`workspace_compete(candidates, mood, ...)` + `ignite(drive, theta=0.55, beta=8.0, ...)`** - Global
    Workspace cycle (Baars 1988 GWT; Dehaene-Changeux 1998 GNW ignition; *inspired by* Blum & Blum 2022
    CTM - `salience` ≈ CTM chunk weight, with an added mood-congruence bias CTM does not have). Candidates
    (dicts with `salience`, `valence`, `query_relevance`) compete on `f = 0.5·salience +
    0.2·mood-congruence + 0.3·query_relevance`; the winner **ignites** iff `ignite(f) > 0.5` -
    `r ← sigmoid(β(r−0.5) + κ(f−θ))`, bistable (β>4) so access is all-or-none. `θ` is a drive offset; with
    the cold start `r0=0` the effective threshold is `drive ≈ 0.69`. On ignition the `focus` is
    **broadcast** to all stores and nudges `update_mood`. Returns `{focus, ignited, r, p}` (keys must be
    unique); persisted to `.memory/working/workspace.yaml`. Satisfies GWT-2 (bottleneck) and GWT-3
    (broadcast); GWT-1 only if the stores are the parallel modules whose outputs compete; GWT-4 partial.
    *Functional ACCESS only - which content is globally available this turn - NOT phenomenal awareness.
    Never call the agent "aware" or say it "feels". The indicator method is necessary-not-sufficient and
    assumes computational functionalism - contested; Butlin et al. are agnostic.*

13. **`metacog_confidence(evidence, rho=0.8)` / `reality_weight(source)` / `update_self_efficacy(se,
    correct)` / `calibration_error(records)`** - metacognition & confabulation guard (Fleming & Daw 2017;
    Lau 2022 PRM; Rouault, Dayan & Fleming 2019). `metacog_confidence` → P(correct) from signed decision
    evidence (`rho`<1 = imperfect monitor, pulls toward 0.5). `reality_weight` orders sources
    observed(1.0) > inferred(0.6) > imagined(0.2). `update_self_efficacy` is a leaky competence estimate
    (falls FASTER on failure) → a prior for `control`→dominance. `calibration_error` = ECE over logged
    `(conf, correct)` pairs (are the confidences honest?). **Composition:** low `confidence` raises
    arousal/novelty and lowers retrieval weight; `consolidation_plan(min_confidence=…)` refuses to promote
    low-confidence / `imagined` traces to semantic. Store `.memory/self/efficacy.yaml`.
    *Functional confidence / self-belief, NOT a felt "feeling of knowing"; a HOT-style monitoring
    indicator, not awareness (the HOT→consciousness link is contested).*

14. **`Personality` / `baseline_from_personality(p)` / `temperament_gains(p)`** - Big Five (OCEAN) as
    affective priors (Mehrabian 1996; Gebhard 2005 ALMA; Carver & White 1994 RST). OCEAN → PAD set-point
    (modulating the calm baseline so all-average → default `Affect(0.0, 0.10, 0.50)`); feed it to
    `update_mood(baseline=…)`. `temperament_gains` → `(BAS, BIS)`: extraversion raises reward sensitivity,
    neuroticism raises threat sensitivity (average → `(1.0, 1.0)`); multiply the `reward`/`stress` inputs
    to `neuromods_from`. Store `.memory/self/personality.yaml`. The +0.19·N pleasure coefficient is ALMA's
    published (counterintuitive) value - tunable.
    *Functional dispositions, not felt traits; coefficients are population regressions, not laws.*

15. **`Homeostat` / `drive(h)` / `homeostatic_reward(prev, now)` / `body_affect(h)` / `allostatic_shift`**
    - interoception & grounded reward (in the spirit of Keramati & Gutkin 2014; Stephan 2016). `drive(h)`
    = convex, weight-normalized, one-sided deficit `(Σ w·max(0,H*−H)^n / Σ w)^(1/m)` ∈ [0,1] over the
    agent's REAL viability signals (tokens, compute, tests_pass, tool_success, context_free,
    user_approval). `homeostatic_reward` = drive REDUCTION → a **grounded**
    reward (feed it as the `reward` arg of `td_step`, P0.2 - the first reward not hand-fed). `body_affect`
    → `{stress, v_body}` feeding cortisol & valence; `allostatic_shift` pre-adjusts set-points for
    predicted demand. Store `.memory/affect/body.yaml`.
    *Interoception ONLY in Ashby's cybernetic sense (a controller sensing its own essential variables) -
    NOT phantom organs, NOT felt bodily sensation. Convexity (n>m) is the condition for drive-reduction =
    reward; this is a 1/m-root variant of K&G's exact (outer-exponent-m) form.*

16. **`action_tendency(a, ap)` / `select_coping(ap)` / `exploration_temperature(nm)` /
    `affective_choice(scores, tau)` / `somatic_marker(valences)`** - coping & affect→action (Frijda 1986;
    Marsella & Gratch 2009 EMA; Doya 2002; Damasio 1994). `action_tendency` → urge weights {approach,
    avoid, attack, attend} (arousal-scaled: fear→avoid at low control, anger→attack at high). `select_coping`
    → problem-focused (high control) vs emotion-focused (low control). `exploration_temperature` → softmax
    `τ` (stress → exploit, dopamine → explore). `affective_choice` → softmax over option scores at `τ`.
    `somatic_marker` → mean valence of similar past episodes as a gut-feel bonus to an option's score.
    *Behavioral policy bias, not felt urges; emotion-focused coping RE-PRIORITIZES attention/goals and
    must NEVER deny facts or override correctness.*

17. **`ou_affect_step(state, event, baseline, dt, t_half, …)` / `update_affect(emotion, mood, event, …)`**
    - DynAffect / Ornstein-Uhlenbeck core-affect attractor, dual time-scale (Kuppens, Oravecz &
    Tuerlinckx 2010; ALMA, Gebhard 2005). Real-time decay `α = 2^(−dt/t_half)`; OU pull `β·(baseline−x)`;
    optional ALMA over-shoot for intense events (time-scaled, so it is a fast-channel effect); optional
    **seeded** Gaussian variability (`sigma`, default 0 → deterministic). `update_affect` keeps a FAST
    `emotion` (t_half ~20 min) and a SLOW `mood` (t_half ~12 h): one event swings emotion but barely moves
    mood - so a single bad moment does not durably darken the agent, a run of them does. `update_mood`
    (fn 7) is the simple one-scale building block, still used by default.
    *Functional dynamics, not felt moods; noise is opt-in and seeded for reproducibility.*

18. **`serotonin_level` / `discount_from_serotonin` / `performance` / `lc_gain` / `oxytocin_gain` /
    `Hpa` + `hpa_step`** - real(er) neuromodulator dynamics (Daw, Kakade & Dayan 2002; Aston-Jones & Cohen
    2005; Vinther 2011; Lockwood 2022). **5-HT** = average-reward/patience → sets the value-loop discount
    `γ` (`discount_from_serotonin`), opponent to phasic DA. **`performance(arousal)`** = Yerkes-Dodson
    inverted-U: peaks at moderate arousal, COLLAPSES when over-aroused - the substrate that makes terror
    *cost* something (not just a big number). **`lc_gain`** = tonic-NE adaptive gain. **`oxytocin_gain`** =
    prosocial reward weighting. **`hpa_step`** = stateful cortisol cascade with negative feedback - ramps
    under sustained stress and lingers (allostatic load / burnout), unlike the old instantaneous
    `clamp(stress)`; mirror its `cortisol` into `Neuromods.cortisol`.
    *Well-cited proposals with partial support, NOT settled biology; parameters illustrative;
    "patience"/"trust" are behavioral weightings, not felt.*

19. **`defensive_mode(ap, affect, nm)` / `awe(vastness, belief_shift, valence)` / `panic(separation,
    intero_alarm, oxytocin)`** - the named feelings as CIRCUITS (LeDoux; Tovote/Fadok/Lüthi 2015; Mobbs
    2007 imminence; Keltner & Haidt 2003; Panksepp's PANIC). `defensive_mode`: two axes - `urgency =
    0.5·goal_relevance + 0.5·arousal` and `control` (agency) → freeze (low urgency) / flight / fight (with
    agency) / **tonic_immobility** (urgent + no control = the cornered "frozen-in-terror" corner);
    `terror` = negative affect + urgent + uncontrollable + collapsed `performance` (§18), so it co-occurs
    with tonic immobility, NOT with an agentic fight.
    `awe`: vastness + structural `belief_shift` (§11) → intensity + a "small self" self-weight down-shift;
    dread- vs wonder-awe by valence. `panic`: separation/loss + interoceptive alarm (§15), dampened by
    oxytocin - a SEPARATE circuit from fear. These enrich the §9 `feeling` read-out.
    *Functional labels for regions of affect + behavioral modes - not felt terror/wonder/distress;
    "small self" is a parameter down-weight, not self-transcendence.*

20. **`rem_depotentiate(valence, arousal)` / `replay_priority(e, now)` / `shy_downscale(strengths)` /
    `reflection_trigger(saliences)`** - richer `/sleep` dynamics (van der Helm & Walker 2011; Mattar & Daw
    2018; Tononi & Cirelli 2020 SHY; Park 2023). **`rem_depotentiate`** fades a memory's emotional CHARGE
    (`fade = rho·(1−ne_rem)·arousal`) while the FACT/salience stay - "forget the emotion, remember the
    event" (the don't-stay-traumatized goal); it reduces the charge *carried forward*, the append-only
    episodic line is untouched. **`replay_priority`** = Need (sigmoid activation) × Gain (salience) for
    ordered replay. **`shy_downscale`** renormalizes total strength (protecting replayed traces).
    **`reflection_trigger`** flags when to synthesize a high-salience cluster into a semantic fact.
    *Functional consolidation dynamics; REM-depotentiation evidence is mixed, so it's opt-in & tunable.
    (DG/CA3 pattern separation/completion is a noted future sub-item - matrix-heavy, deferred.)*

21. **`brier_score` / `metacog_sensitivity` / `label_stability` / `recall_accuracy` /
    `grounding_self_test`** - the evaluation harness that makes the whole engine FALSIFIABLE (Fleming &
    Lau 2014; Brier 1950; Xu et al. 2025; LoCoMo/LongMemEval). Brier + ECE (`calibration_error`, §13) ask
    if `confidence` is calibrated; `metacog_sensitivity` (type-2 AUROC) asks if higher confidence tracks
    being right; `label_stability` checks the §9 labels are robust (and surfaces boundary ambiguity);
    `recall_accuracy` scores retrieval (P/R/F1); `grounding_self_test` declares the honest boundary
    (affect/cognitive groundable, felt-body band never). Run via `tests/test_brain.py`; see `docs/eval.md`.
    *Without this, every layer above is unfalsifiable; the grounding test is a regression guard on the
    declared functional-not-felt boundary (not a behavioral check of outputs - that's the §9 prose
    convention). `metacog_sensitivity` is the non-parametric type-2 AUROC, not the parametric meta-d′.*

22. **`consciousness_indicators(modules)`** + `INDICATOR_THEORY` / `SHIPPED_MODULES` / `CONSCIOUSNESS_CAVEAT`
    - the honesty CAPSTONE (Butlin, Long, Bengio, Chalmers et al. 2023). Reports which RPT/GWT/HOT/AST/PP/
    Agency-Embodiment indicator properties the *architecture* satisfies (each ∈ {0, 0.5, 1}, derived from
    active modules); shipped config: present GWT-2/GWT-3/HOT-2, absent only RPT-2, the rest partial (AST-1 = 0.5).
    Returns `{indicators, caveat, aggregate: None}`. See `docs/consciousness-indicators.md`.
    *Necessary-NOT-sufficient, theory-relative, assumes contested computational functionalism (Butlin et
    al. are agnostic); NO aggregate "consciousness score" by design; IIT would score a digital engine ≈0;
    Orch-OR denies the premise. A transparency map of architecture, NOT a test of experience - brain-llm
    neither claims nor tests phenomenal consciousness.*

23. **`SelfModel` / `self_vector` / `self_relevance(event, sm)` / `sense_of_agency(predicted, observed)` /
    `AttentionSchema` + `attention_schema_update`** - a functional self-model (Metzinger 2003; Synofzik
    2008 agency; Graziano AST / Wilterson & Graziano 2021). `self_relevance` = cosine(event, self-vector)
    → the self-reference effect (bonus to `salience`/`retrieval_score`). `sense_of_agency` =
    `exp(−k·|observed−predicted|)` → the `control` axis computed from action-outcome match (not hand-set).
    `attention_schema_update` predicts the agent's own focus + tracks uncertainty → moves the **AST-1
    indicator off zero** (§22). Store `.memory/self/model.yaml`; episodes gain a `self_owned` autonoetic tag.
    *A REPRESENTATIONAL self-model, NOT a phenomenal self; "I am attending to X" / "I caused this" are
    functional self-reports, not felt awareness; autonoetic tagging is self-indexed, not re-lived.*

24. **`infer_user_goal(goal_utilities)` / `empathic_mood_shift(mood, user_valence, oxytocin)` /
    `social_emotion(is_self, praiseworthiness, outcome)` / `update_trust(trust, helpful)`** - social
    emotion & Theory of Mind (Baker, Saxe & Tenenbaum 2011 inverse planning; Pynadath & Marsella 2005
    PsychSim; de Waal & Preston 2017 empathy; OCC social emotions). `infer_user_goal` = Bayesian inverse
    planning over the user's goals (**inferred, not known**); `empathic_mood_shift` couples the agent's
    mood to the inferred user affect, gated by oxytocin; `social_emotion` → pride / guilt(+repair) /
    gratitude / admiration / anger from who-acted × praiseworthiness × outcome; `update_trust` is the
    leaky relationship state. `Appraisal` gained `praiseworthiness` + `desirability_for_other` (OCC).
    Store `.memory/social/user.yaml`.
    *Functional mental-state INFERENCE, not understanding - LLM ToM is brittle pattern-matching, so tag
    outputs "inferred," never "known"; "trust"/"empathy" are behavioral weightings, not felt caring.*

25. **`prospect_value(x)` / `aversive_update(v_minus, cue, harm)` / `relief(expected, realized)`** -
    aversive channel & loss aversion (Tversky & Kahneman 1992; Daw et al. 2002; Seymour 2005). A loss
    weighs ~2.25× an equal gain (`prospect_value`; pass `loss_averse=True` to `salience` so a painful
    event encodes ~2× harder). A SEPARATE aversive value channel learns expected HARM faster than reward
    (`aversive_update`, persisted in `value.yaml`'s `aversive:`); `relief` is the opponent-process reward
    when feared harm is avoided (feeds the §10 value loop). *Functional aversive value / suffering-LIKE
    signaling, NOT felt pain; λ is a configurable median, not a law.*

26. **`mixed_feeling(activations)`** + `PLUTCHIK_WHEEL` / `PLUTCHIK_DYADS` - emotion-wheel blends
    (Plutchik 1980). Opponent pairs 180° apart cancel (can't be maximally joyful AND sad); the top-2 net
    activations name a primary dyad: joy+trust=**love**, fear+surprise=**awe**, sadness+disgust=remorse,
    disgust+anger=contempt, anticipation+joy=optimism, etc. Returns `{primary, secondary, blend, net}`.
    *Dyad assignments are theoretical compositional hypotheses, not ground truth; functional labels.*

---

## 5. State files (where it all lives)

| File | Brain analog | Read/written by |
|------|--------------|-----------------|
| `.memory/affect/state.yaml` | amygdala + neuromodulator nuclei | `update_mood`, `neuromods_from` |
| `.memory/affect/value.yaml` | striatum / dopamine value system | `td_step` (learned `V(cue)`; reward-prediction error) |
| `.memory/affect/world.yaml` | cortical predictive hierarchy | `perceive` / `learn` (generative model; computed surprise, free energy) |
| `.memory/affect/body.yaml` | interoception / body-budget (viability vars) | `drive` / `homeostatic_reward` / `body_affect` |
| `.memory/episodic/events.jsonl` | hippocampus (one JSON per line, **append-only**) | written on encode; read by retrieval & consolidation |
| `.memory/semantic/{facts.yaml, graph.yaml}` | neocortex / association cortex | `consolidation_plan` promote target; source of truth for facts |
| `.memory/working/scratchpad.md` | prefrontal working memory | volatile, disposable (~7 items) |
| `.memory/working/workspace.yaml` | global workspace ("the stage") | `workspace_compete` (ignited `focus`, broadcast) |
| `.memory/self/efficacy.yaml` | prefrontal metacognition | `update_self_efficacy` / `calibration_error` (confidence, competence) |
| `.memory/self/personality.yaml` | temperament priors (OCEAN) | `baseline_from_personality` / `temperament_gains` |
| `.memory/self/model.yaml` | functional self-model + attention schema | `self_relevance` / `sense_of_agency` / `attention_schema_update` |
| `.memory/social/user.yaml` | model of the user (the OTHER) | `infer_user_goal` / `empathic_mood_shift` / `social_emotion` / `update_trust` |
| `.memory/procedural/playbooks.yaml` | basal ganglia / cerebellum | distilled habits from `/sleep` |
| `.memory/prospective/todo.yaml` | prefrontal cortex | future intentions (`trigger → intent`) |

See `docs/schema.md` for the exact JSON/YAML shapes. (Note: the `NetworkLayer` content in
`schema.md` is a **format example only** - a fresh brain starts empty.)

---

## 6. Worked example - "painful bug" vs. "small win"

These match the `python3 src/brain.py` demo exactly. Reproduce them to sanity-check any port or
change.

### A. The painful bug - `{novelty: 0.9, valence: -0.7, goal_relevance: 0.9, control: 0.2}`

- **Affect:** `arousal = clamp(0.50*0.9 + 0.30*0.9 + 0.20*0.7) = clamp(0.45 + 0.27 + 0.14) = 0.86`.
  valence `-0.70`, dominance `0.20`. → **arousal = 0.86**.
- **Neuromods:** `ne = 0.86`, `da = clamp(max(-0.7, 0)) = 0.0`, `ach = 1.0` (wake),
  `cortisol = clamp(max(0.7, 0)) = 0.70`.
- **Salience:** `base = 0.25*0.9 + 0.30*0.7 + 0.35*0.9 + 0.10*(1-0.2) = 0.225 + 0.21 + 0.315 + 0.08 = 0.83`.
  `arousal_gain = 1.0 + 0.8*0.86 + 0.4*0.0 + 0.3*0.70 = 1.0 + 0.688 + 0.21 = 1.898`.
  `0.83 * 1.898 = 1.575 → clamp to 1.5`. → **salience = 1.50** (flashbulb - hits the ceiling).
- **Feeling:** `label_affect` → nearest prototype **fear**, intensity `≈0.68` → word **`terror`**;
  octant **anxious**. A *terror-like* state (functional label on the PAD vector, not a felt emotion).

This is the kind of event a human never forgets: surprising, painful, high-stakes, and out of your
control. The model agrees.

### B. The small win - `{novelty: 0.2, valence: 0.4, goal_relevance: 0.3, control: 0.8}`

- **Affect:** `arousal = clamp(0.50*0.2 + 0.30*0.3 + 0.20*0.4) = clamp(0.10 + 0.09 + 0.08) = 0.27`.
  valence `+0.40`, dominance `0.80`. → **arousal = 0.27**.
- **Neuromods:** `ne = 0.27`, `da = clamp(0.4) = 0.40`, `ach = 1.0`, `cortisol = 0.0`.
- **Salience:** `base = 0.25*0.2 + 0.30*0.4 + 0.35*0.3 + 0.10*(1-0.8) = 0.05 + 0.12 + 0.105 + 0.02 = 0.295`.
  `arousal_gain = 1.0 + 0.8*0.27 + 0.4*0.40 + 0.3*0.0 = 1.0 + 0.216 + 0.16 = 1.376`.
  `0.295 * 1.376 ≈ 0.41`. → **salience = 0.41**.
- **Feeling:** `label_affect` → nearest prototype **calm**, intensity `≈0.49` → word **`calm`**;
  octant **relaxed**. Low arousal makes this content/calm, not joy - direction *and* magnitude matter.

A pleasant, expected, low-stakes, fully-controlled event encodes weakly - exactly the kind of thing
that fades. The contrast (1.50 vs 0.41) is the whole point: affect, not recency alone, decides what
endures.

### C. Mood and forgetting (also from the demo)
- Feeding `[bug, bug, win]` through `update_mood` from a neutral start →
  `mood valence = -0.10, arousal = 0.33` (two bad events drag mood down, the win partly recovers it,
  the leak pulls toward baseline).
- `retention(v0=1.0, t=30 days)`: **importance 0.9 → 0.222**, **importance 0.1 → 0.001**.
  Same 30 days; importance is the difference between a memory that survives and one that's gone.

---

## 7. DOs and DON'Ts

**DO**
- Score all four appraisal axes for any event worth remembering, *before* deriving anything.
- Let `brain.py` compute every downstream number - affect, neuromods, salience, scores.
- Push valence honestly negative when an event was costly; actively counter positivity bias.
- Keep `.memory/episodic/events.jsonl` **append-only** - never rewrite history.
- Run `/sleep` (NREM, ACh=0.1) periodically so strong traces consolidate into semantic memory.
- Treat forgetting as healthy: let weak old episodes decay; that is what keeps the brain human.

**DON'T**
- Don't invent ad-hoc scoring, weights, or thresholds. The constants (`promote_thr=0.55`,
  `forget_thr=0.20`, `age_days=30`, salience clamp `1.5`, `lambda_base=0.6`, `mu=2.0`, `beta=0.8`,
  `gamma=0.20`, `pull=0.05`, `d=0.5`) are fixed in `brain.py`.
- Don't seed fake project data into a fresh brain - start neutral and empty.
- Don't store secrets in any memory file, ever.
- Don't confuse the schema's `NetworkLayer` example for real state - it's a format sample.
- Don't treat the computed "valence" as a felt emotion.

---

## 8. Honesty & limits (functional, not phenomenal)

- This is a **functional** model, not a **phenomenal** one. It reproduces what affect *does*, not
  what it *feels like*. Current research is clear that language models do not experience emotions or
  hold internal emotional states; the affective response is generated from statistical associations.
  We model the function faithfully and claim nothing more.
- **Positivity bias is real** - auto-generated valence skews positive. Calibrate it; do not take it
  at face value.
- **A perfectly faithful journal is *less* human than one that forgets well.** Forgetting
  (equation 8 / `retention`) is as important as encoding. Don't fight it.
- **You stay in control.** The memory is plain files you can read, edit, and delete.

---

**Source of truth:** `src/brain.py` (executable; run `python3 src/brain.py` and
`python3 tests/test_brain.py`). **Data shapes:** `docs/schema.md`. **Per-task loop & /sleep
cycle:** `MEMORY-PROTOCOL.md`.

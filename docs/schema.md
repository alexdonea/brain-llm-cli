# Data schemas

> **Store root.** Each agent's brain lives at `agents/<name>/memory/`. The paths below are written
> `.memory/…` as shorthand for that per-agent store root: substitute `agents/<name>/memory/`.
>
> **Derived caches (not source of truth).** `.memory/episodic/embeddings.npy` + `embeddings.ids.json` are an
> OPTIONAL semantic-search vector cache, rebuilt from `events.jsonl` by `reindex` (gitignored). `recall`/`know`
> refresh it INCREMENTALLY - only new or hand-edited items are re-embedded (the `.ids.json` keeps a per-row
> content hash), so adding one memory to a 50k store embeds one vector, not 50,000. The shared
> `models/wordllama/` dir holds the vendored 1.8MB tokenizer for that local backend (committed). Both are
> absent unless `wordllama` is installed; recall falls back to lexical matching without them.

## Episode (`.memory/episodic/events.jsonl`, one JSON per line - hippocampus)
```json
{
  "id": "e-0001",
  "t0": 1750339380,                       // unix seconds, time of encoding
  "retrievals": [1750339380],             // times this memory was recalled (ACT-R)
  "task": "fix NetworkLayer retry loop",
  "outcome": "success",
  "files": ["NetworkLayer.swift"],
  "appraisal": {"novelty": 0.9, "valence": -0.7, "goal_relevance": 0.9, "control": 0.2},
  "affect":    {"valence": -0.7, "arousal": 0.86, "dominance": 0.2},   // from appraise_to_affect
  "feeling":   {"label": "fear", "word": "terror", "intensity": 0.68}, // from label_affect() - recomputable read-out
  "confidence": 0.70,                     // P(this episode's judgment is correct); derived via metacog_confidence() when --evidence is given, else the default
  "source": "observed",                   // observed | inferred | imagined  (PRM reality tag)
  "self_owned": true,                     // autonoetic tag (§23): was this the agent's own action/experience?
  "salience": 1.50,                       // from salience() - encoding strength
  "evidence": "tests=pass"                // OPTIONAL (--evidence): the artifact that GROUNDS the outcome+confidence, not a bare self-report
}
```

## Affect state (`.memory/affect/state.yaml` - amygdala + neuromodulator nuclei)
```yaml
emotion:     {valence: -0.40, arousal: 0.60, dominance: 0.45}   # FAST scale (~min) - in-the-moment
mood:        {valence: -0.10, arousal: 0.33, dominance: 0.50}   # SLOW scale (~hrs) - lags; read by recall
baseline:    {valence: 0.00,  arousal: 0.10, dominance: 0.50}   # homeostatic set-point (from personality)
neuromods:   {ne: 0.33, da: 0.10, ach: 1.0, cortisol: 0.20,      # ne phasic; ach 1=wake/0.1=NREM
              serotonin: 0.50, oxytocin: 0.00, ne_tonic: 0.10}   # 5-HT->discount; OT prosocial; tonic LC
hpa:         {crh: 0.00, acth: 0.00, cortisol: 0.10}             # slow stress cascade (hpa_step); mirror -> cortisol
updated:     <ISO-8601 timestamp>
```
`hpa_step(hpa, stress, dt)` advances the cortisol cascade (ramps + lingers); `performance(arousal)` is the
Yerkes-Dodson inverted-U; `serotonin_level`/`discount_from_serotonin` set the value-loop discount.
`update_affect(emotion, mood, event_affect, baseline, dt)` advances both time-scales (DynAffect/OU); the
fast `emotion` swings per event, the slow `mood` only drifts. `update_mood` remains the simple one-scale form.

## Value cache (`.memory/affect/value.yaml` - striatum / dopamine value system)
```yaml
gamma:  0.9                                          # TD discount on V(next_cue)
alpha:  0.3                                          # learning rate
values: {"fix NetworkLayer retry loop": 0.42}        # cue -> learned expected value
```
`td_step(V, cue, reward)` computes `delta = r + gamma*V(next) - V(cue)` (the reward-prediction error
dopamine encodes); an unexpected outcome spikes `neuromods_from(..., delta)` `da` and boosts
`salience(..., rpe=delta)`. `cue` is a stable context key you choose (task type, file, playbook).

## Generative world-model (`.memory/affect/world.yaml` - cortical predictive hierarchy)
```yaml
states: [routine, novel_problem, incident]            # latent "situations"
obs:    [success, failure, insight, surprise]          # observable event categories (perceive maps outcome → obs)
a: [[1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0]]   # likelihood counts a[obs][state]
d: [1.0, 1.0, 1.0]                                    # prior counts over states
```
Load into `WorldModel(states, obs, a, d)`. `perceive(wm, o)` → `{novelty = 1−P(o), free_energy = −ln P(o),
belief_shift = KL(posterior‖prior), posterior}`; `learn(wm, o, posterior)` updates counts so recurring
events habituate. `novelty` replaces the hand-fed appraisal axis; `belief_shift` is structural surprise.

## Body-budget / interoception (`.memory/affect/body.yaml` - viability variables, not viscera)
```yaml
levels:   {tokens: 1.0, tests_pass: 1.0, tool_success: 1.0}   # current, each in [0,1] (1 = healthy)
setpoint: {tokens: 1.0, tests_pass: 1.0, tool_success: 1.0}   # targets (allostatic; allostatic_shift)
weights:  {tokens: 1.0, tests_pass: 1.5, tool_success: 1.0}
```
Load into `Homeostat(levels, setpoint, weights)`. `drive(h)` = deficit; `homeostatic_reward(prev, now)` =
drive reduction → a **grounded** reward for the value loop; `body_affect(h)` → `{stress, v_body}` feeding
cortisol & valence. Real substrate signals only - cybernetic regulation, not felt bodily sensation.

## Global workspace (`.memory/working/workspace.yaml` - the limited-capacity "stage", GWT/GNW)
```yaml
focus:   null            # the broadcast content this turn (the event's text), or null (nothing crossed threshold)
ignited: false
r:       0.0             # ignition level in [0,1] (bistable; ~0 = local, ~1 = ignited)
p:       {}              # softmax competition distribution over candidates
updated: <ISO-8601 timestamp>
```
`workspace_compete(candidates, mood)` (each candidate a dict with `salience`, `valence`,
`query_relevance`) returns `{focus, ignited, r, p}`; the winner ignites iff `ignite(f) > 0.5` (bistable,
all-or-none) and is broadcast. **Driven each `perceive()`/`react()`**: the incoming event competes and,
if it crosses the threshold, becomes the broadcast `focus` ("in mind right now", shown in `feel`/`status`);
sub-threshold events stay local. `sleep()` quiets the stage. Functional access - *which* content is
globally available - not awareness.

## Metacognition / self (`.memory/self/efficacy.yaml` - prefrontal metacognition)
```yaml
efficacy:    {python: 0.5}       # domain -> self-efficacy in [0,1] (update_self_efficacy)
calibration: [[0.7, true]]       # logged [confidence, correct] pairs -> calibration_error (ECE) + calibration_informative()
valence_calibration: [[-0.7, -1]]  # logged [signed_valence, outcome_polarity] pairs -> valence_outcome_consistency() honesty audit
default_efficacy:          0.5
min_confidence_to_promote: 0.5   # consolidation_plan(min_confidence) hallucination guard
value_uncertainty:         0.5   # §31 corrigibility: uncertainty re the true objective (floored >0 -> always defer-positive)
identity_anchor: {python: 0.5}   # §34 the earliest self-vector, frozen at first save -> self_continuity baseline
```
`metacog_confidence(evidence)` → P(correct); `update_self_efficacy(se, correct)` → competence prior for
`control`; `reality_weight(source)` orders observed > inferred > imagined. Functional confidence, not felt.
`value_uncertainty` feeds `corrigibility_value` (§31, the safety cornerstone); `identity_anchor` feeds
`self_continuity` (§34). Both persist here so corrigibility + narrative identity survive across sessions.

## Personality (`.memory/self/personality.yaml` - temperament priors)
```yaml
openness: 0.5
conscientiousness: 0.5
extraversion: 0.5
agreeableness: 0.5
neuroticism: 0.5     # OCEAN in [0,1]; 0.5 = average
```
`baseline_from_personality(p)` → the PAD set-point for `update_mood(baseline=…)` (all-0.5 → default
`Affect(0.0, 0.10, 0.50)`); `temperament_gains(p)` → `(BAS, BIS)` multiplying the reward/stress inputs to
`neuromods_from`. Functional dispositions, not felt traits.

## Self-model (`.memory/self/model.yaml` - functional identity, not a felt self)
```yaml
competencies: {python: 0.5, debugging: 0.5}   # domain -> proficiency [0,1]
goals:                                         # executive hierarchy (§29); a bare string also loads (defaults)
  - {desc: "finish the project", importance: 0.9, urgency: 0.8, progress: 0.0, parent: null}
traits: {}                                     # weighted interests / dispositions
attention_schema: {focus: "", predicted_next: "", uncertainty: 1.0}
```
Load into `SelfModel(competencies, goals, traits)` (+ a parallel list of `Goal` objects). `self_relevance`
= cosine to the self-vector (self-reference effect); `sense_of_agency(predicted, observed)` → the `control`
axis; `attention_schema_update(...)` predicts the agent's own focus (AST-1). **Executive control (§29):**
`select_active_goal(goals, mood)` (guided activation, mood-gated) picks the active goal that biases
behaviour; `deliberate()` weighs a prepotent impulse against it via `conflict_signal` →
`expected_value_of_control` → `inhibit` - affect informs but does not dictate action. **Planning (§30):**
a goal may carry `plan: [{step, done}]`; `subgoal_progress` gives the next step + completion (which drives
`progress`), and `lookahead` does a one-ply value forward-search over candidate next actions.
Functional self-report, not felt awareness.

## User model / relationship (`.memory/social/user.yaml` - the OTHER, for social emotion & ToM)
```yaml
trust: 0.5                          # relationship trust [0,1] (leaky integrator)
inferred_goals: {}                  # goal -> posterior (INFERRED, not known)
inferred_affect: {valence: 0.0}     # estimate of the user's affect
```
`infer_user_goal(goal_utilities)` (Bayesian inverse planning), `empathic_mood_shift(mood, user_valence,
oxytocin)`, `social_emotion(is_self, praiseworthiness, outcome)` → pride/guilt/gratitude/admiration/anger,
`update_trust(trust, helpful)`. The `Appraisal` also gained `praiseworthiness` and `desirability_for_other`
(OCC; default 0.0). Functional inference, not understanding; tag user-state as **inferred**, never known.

## Semantic facts (`.memory/semantic/facts.yaml` - neocortex declarative store)
```yaml
facts:
  - {id: f-0000, text: "stops at 1.5-2x ATR below support", source: e-0004,
     confidence: 0.7, affect_charge: 0.43, valid_from: <date>}   # affect_charge optional (REM-faded)
```
Written at `/sleep` when an episode is promoted (`consolidation_plan` §8 + REM depotentiation §20): the
episode generalizes into a durable fact whose emotional `charge` is faded (the fact persists, the sting
fades). `source` is the episode id (or `"reflection"` for synthesized clusters). `src/runtime.py`
`Brain.sleep()` writes this store. **Promotion needs rehearsal:** consolidation gates on ACT-R activation
(recency + repeated retrieval), not salience alone, so a fresh memory does *not* promote on its first
night - recall it once or twice before sleeping. `promoted: 0` on a cold first sleep is expected, not a
bug (reflections still fire on salient clusters).

## Semantic graph (`.memory/semantic/graph.yaml` - neocortex / association cortex)
```yaml
nodes:
  - {id: "c:caching", type: concept, label: "caching"}            # concept (from an episode cue)
  - {id: "d:debugging", type: domain, label: "debugging"}          # domain hub
edges:
  - {from: "c:caching", to: "d:debugging", rel: in_domain, weight: 0.3, valid_from: 2026-06-20}
  - {from: "c:caching", to: "c:invalidation", rel: related_to, weight: 0.21, valid_from: 2026-06-20}  # content overlap
```
**Auto-grown during `Brain.sleep()`** (`_grow_graph`, §27): each episode's `cue` → a concept node and
`domain` → a domain node; a concept links to its domain (`in_domain`) and to other concepts whose episodes
share content (`related_to`, Jaccard ≥ 0.10), Hebbian-strengthened on recurrence. `recall()` then uses
`graph_proximity` (spreading activation) so a related-but-not-cued memory can surface. Still hand-editable.
## Procedural playbooks (`.memory/procedural/playbooks.yaml` - basal ganglia / cerebellum)
```yaml
playbooks:
  - {id: pb-0000, domain: debugging, steps: ["check the cache", "diff the config"],
     attempts: 5, successes: 4, strength: 0.62, updated: 2026-06-20}
```
Distilled during `Brain.sleep()` from clusters of same-domain success episodes; `strength` follows the
power law of practice (`practice_strength`, §28). Distinct from `self/efficacy.yaml` (a scalar belief) -
this is a concrete, reinforced how-to. Prospective intentions (`prospective/todo.yaml`,
`{id, trigger, intent, created, done}`) are now captured by `Brain.intend()` and resurface at `wake()`;
the body-budget (`affect/body.yaml`) is ticked each `perceive()`/`react()` and rested at `sleep()`.

All `src/brain.py` functions operate on these shapes. Times are unix seconds except
`retention(t)` which takes days.

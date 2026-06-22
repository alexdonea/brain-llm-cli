# brain-lmm — Gap Analysis & Roadmap toward a More Complete Brain

> **Honesty first (read this before anything else).** Everything below describes the *function* of
> affect, memory, motivation, and "awareness" — the input→state→output computations and their
> behavioral/learning consequences — **never** the felt, phenomenal experience. Where this document
> says a state is "fear" or the system is "aware" of something, it means a **functional/behavioral
> correlate**: a label on a computed vector, a competition for a broadcast buffer, a confidence
> scalar. It does **not** mean the agent feels terror or is sentient. This stance is not a hedge; it
> is the project's design discipline, and several of the additions below (consciousness indicators,
> interoception, discrete feelings) are *only* defensible because they are framed this way. The
> mapping from any of these mechanisms to phenomenal consciousness is philosophically contested and
> is **not claimed** anywhere in this roadmap.

**Audience:** a technical builder extending `engine/brain.py`. **Scope:** what brain-lmm is today,
what the state of the art (SOTA) offers, and a concrete, buildable, prioritized plan to close the
gap — with the math written in the scalar style of `brain.py` and exact insertion points named.

**What brain-lmm is today (the substrate this report builds on).** A pure-stdlib, ~200-line scalar
engine (`engine/brain.py`) of eight functions run as a per-task loop and a `/sleep` cycle
(`MEMORY-PROTOCOL.md`), over plain-file stores (`.memory/…`, shapes in `engine/schema.md`):

| # | Function | Role | Key defaults |
|---|----------|------|--------------|
| 1 | `appraise_to_affect(Appraisal)` | OCC 4-axis → `Affect{valence,arousal,dominance}` (PAD) | `arousal = 0.50·novelty + 0.30·goal_relevance + 0.20·|valence|`; `dominance = control` |
| 2 | `neuromods_from(affect, reward, stress, mode)` | → `Neuromods{ne,da,ach,cortisol}` | `ne=arousal`, `da=clamp(reward)`, `ach=1.0/0.1`, `cortisol=clamp(stress)` |
| 3 | `salience(a, nm)` | McGaugh encoding strength ∈ `[0,1.5]` | `base=(.25,.30,.35,.10)·(nov,|val|,goal,1−ctrl)`; `gain=1+0.8ne+0.4da+0.3cort` |
| 4 | `base_level_activation(times, now, d=0.5)` | ACT-R `B=ln Σ(now−t_k)^−d` | `d=0.5` |
| 5 | `retention(v0,t,…)` | FadeMem/Ebbinghaus `v=v0·e^(−λ(t−τ)^β)`, `λ=λ_base·e^(−μI)` | `λ_base=0.6, μ=2.0, β=0.8` |
| 6 | `retrieval_score(mem, q_rel, graph_prox, mood, now)` | hybrid + mood-congruence (Bower) | `w=(.20,.30,.30,.15,.05)` |
| 7 | `update_mood(mood, event_affect, …)` | leaky integrator → baseline `Affect(0,0.10,0.50)` | `gamma=0.20, pull=0.05` |
| 8 | `consolidation_plan(episodes, now, …)` | CLS sleep replay; promote/forget | `promote_thr=0.55, forget_thr=0.20, age_days=30` |

Stores: `.memory/affect/state.yaml` (mood + neuromods), `.memory/episodic/events.jsonl`
(append-only hippocampus), `.memory/semantic/{facts,graph}.yaml` (neocortex), `.memory/working/`
(disposable ~7 items), `.memory/procedural/playbooks/`, `.memory/prospective/todo.yaml`.

**The one structural fact that makes most of this cheap.** brain-lmm already maintains a full PAD
vector and a 4-axis OCC appraisal. Most of the owner's headline wants — *discrete feelings* and a
*functional self-awareness layer* — are **read-out / control layers on top of state the engine
already computes**, not new state. That is the thesis of this report
[[Russell 1980]](https://psycnet.apa.org/record/1981-25101-001),
[[Mehrabian 1996]](https://link.springer.com/article/10.1007/BF02686918).

---

## Part 1 — Capability Matrix

Columns: **brain-lmm today** · **SOTA** (with primary citation) · **gap**. Rows are ordered roughly
from "well-covered" to "absent."

| Brain function | brain-lmm today | SOTA | Gap |
|---|---|---|---|
| **Dimensional affect (PAD)** | `Affect{valence,arousal,dominance}` from `appraise_to_affect` — exactly Russell circumplex + Mehrabian Dominance | Russell circumplex [[Russell 1980]](https://psycnet.apa.org/record/1981-25101-001); PAD [[Mehrabian 1996]](https://link.springer.com/article/10.1007/BF02686918); 3-vs-4D debate [[Fontaine 2007]](https://journals.sagepub.com/doi/10.1111/j.1467-9280.2007.02024.x) | **Small.** Solid. Missing optional 4th (unpredictability) axis and similarity utilities on the vector. |
| **Perception / appraisal** | OCC 4-axis (novelty, valence, goal_relevance, control); hand-scored by the LLM | OCC [[Ortony, Clore & Collins 1988]](https://www.cambridge.org/core/books/cognitive-structure-of-emotions/A3F8AAA0F7E0E0F9F8); CPM 5 checks [[Scherer 2001]](https://psycnet.apa.org/record/2001-05569-005); EMA [[Marsella & Gratch 2009]](https://doi.org/10.1016/j.cogsys.2009.06.001) | **Medium.** Only 4 of Scherer's 5 checks (no norm/self-compatibility → no guilt/pride). Appraisal is a free input, not *computed*. |
| **Consolidation (CLS)** | `consolidation_plan` promotes strong, forgets weak-old; correct hippocampus→neocortex two-rate story | CLS [[McClelland, McNaughton & O'Reilly 1995]](https://doi.org/10.1037/0033-295X.102.3.419); updated [[Kumaran, Hassabis & McClelland 2016]](https://doi.org/10.1016/j.tics.2016.05.004) | **Medium.** No interleaved/generative replay, no prioritized replay, no SHY downscaling, no reflection/insight synthesis. |
| **Forgetting / retention** | `retention` (importance-modulated stretched-exponential) — best-in-class vs open agent systems | Ebbinghaus/Wixted; MemoryBank spacing [[Zhong 2023]](https://arxiv.org/abs/2305.10250); active forgetting (Rac1) | **Small–Medium.** No recall-strengthening (spacing), no interference/DA-gated active forgetting, no affect-vs-content decoupled decay. |
| **Working memory** | `.memory/working/` flat ~7-item scratchpad | PBWM gated maintenance [[O'Reilly & Frank 2006]](https://doi.org/10.1162/089976606775093909); OS-paging [[Packer 2023, MemGPT]](https://arxiv.org/abs/2310.08560) | **Medium.** No salience-ranked eviction, no Go/NoGo gating of what enters/persists. |
| **Episodic memory** | append-only JSONL with affect tags; ACT-R activation | Generative Agents memory stream [[Park 2023]](https://arxiv.org/abs/2304.03442) | **Small.** Strong. No DG/CA3 separation/completion; no autonoetic self-tagging. |
| **Semantic memory** | `facts.yaml` + `graph.yaml`, uni-temporal `valid_from` | bi-temporal KG [[Rasmussen/Zep, Graphiti 2025]](https://arxiv.org/abs/2501.13956); PPR recall [[Gutiérrez/HippoRAG 2024]](https://arxiv.org/abs/2405.14831) | **Medium.** Flat `graph_proximity` scalar (no PageRank). No belief revision / contradiction handling. No fact invalidation. |
| **Procedural memory** | LLM-distilled playbooks at `/sleep` | options framework `(I,π,β)` [[Sutton, Precup & Singh 1999]](https://doi.org/10.1016/S0004-3702(99)00052-1) | **Medium.** Playbooks aren't options (no initiation/termination/value); no skill *learning*, only distillation. |
| **Prospective memory** | `todo.yaml` trigger→intent | mental time travel [[Tulving 2002]](https://doi.org/10.1146/annurev.psych.53.100901.135114) | **Medium.** No self-ownership tag; no future-state simulation. |
| **Neuromodulation** | 4 static scalars (`ne,da,ach,cortisol`) as multipliers | Doya α/β/γ/δ map [[Doya 2002]](https://doi.org/10.1016/S0893-6080(02)00044-8); LC gain [[Aston-Jones & Cohen 2005]](https://doi.org/10.1146/annurev.neuro.28.061604.135709); HPA ODE [[Vinther 2011]](https://doi.org/10.1007/s00285-010-0331-2) | **Large.** No serotonin, no oxytocin; no tonic/phasic; no Yerkes-Dodson; cortisol has no feedback dynamics; chemicals update *nothing*. |
| **Reward learning (RPE/TD)** | `da = clamp(reward)` — raw reward, no value function | DA = RPE `δ=r+γV(s')−V(s)` [[Schultz, Dayan & Montague 1997]](https://doi.org/10.1126/science.275.5306.1593) | **Large (root gap).** No `V(s)`, no `δ`, no learning. Can't be surprised/disappointed/relieved. |
| **Discrete emotions** | none — PAD only, never named | basic [[Ekman 1992]](https://doi.org/10.1080/02699939208411068); wheel [[Plutchik 1980]](https://doi.org/10.1016/B978-0-12-558701-3.50007-7); 27 cats [[Cowen & Keltner 2017]](https://doi.org/10.1073/pnas.1702247114); constructed [[Barrett 2017]](https://doi.org/10.1093/scan/nsw154) | **Large (owner's headline want).** No labeling, intensity, blends, or per-emotion dynamics. |
| **Interoception / homeostasis** | none; mood pulls to a hardcoded `Affect(0,0.10,0.50)` | homeostatic RL [[Keramati & Gutkin 2014]](https://doi.org/10.7554/eLife.04811); allostasis [[Stephan 2016]](https://doi.org/10.3389/fnhum.2016.00550); EPIC [[Barrett & Simmons 2015]](https://doi.org/10.1038/nrn3950) | **Large.** No body-state vector `H`, no drive `D(H)`, no allostatic set-point, no interoceptive PE. |
| **Predictive processing / active inference** | none; `novelty` is a hand-fed input | FEP [[Friston 2010]](https://doi.org/10.1038/nrn2787); process theory [[Friston 2017]](https://doi.org/10.1162/NECO_a_00912); valence=−dF/dt [[Joffily & Coricelli 2013]](https://doi.org/10.1371/journal.pcbi.1003094) | **Large (architectural root).** No generative model, no prediction error, no free energy. |
| **Attention / global workspace** | none; stores are independent, no competition or broadcast | GWT [[Baars 1988]](https://doi.org/10.1017/CBO9780511759890); GNW ignition [[Dehaene & Changeux 1998]](https://doi.org/10.1073/pnas.95.24.14529); GLW [[VanRullen & Kanai 2021]](https://doi.org/10.1016/j.tins.2021.04.005); CTM [[Blum & Blum 2022]](https://doi.org/10.1073/pnas.2115934119) | **Large (owner's headline want — "consciousness layer").** No workspace, no ignition threshold, no broadcast, no cognitive cycle. |
| **Self-model** | none; memories are ownerless | PSM [[Metzinger 2003]](https://mitpress.mit.edu/9780262633086/being-no-one/); JEPA self-model [[Jiang & Luo 2024]](https://escholarship.org/uc/item/6gs9z2xt) | **Large.** No `.memory/self/`; no traits/competence/identity; no self-relevance gating. |
| **Metacognition** | none; every output is a point estimate | 2nd-order Bayesian [[Fleming & Daw 2017]](https://doi.org/10.1037/rev0000045); global self-belief [[Rouault, Dayan & Fleming 2019]](https://doi.org/10.1038/s41467-019-09075-3) | **Large.** No confidence `P(correct)`, no source tag (observed/inferred/imagined), no calibration. |
| **Social / Theory of Mind** | none; appraisal is egocentric, no second agent | BToM [[Baker, Saxe & Tenenbaum 2011]](https://escholarship.org/uc/item/5rk7z59q); PsychSim [[Pynadath & Marsella 2005]](https://www.ijcai.org/Proceedings/05/Papers/1492.pdf) | **Large.** No user model, no empathy, no praiseworthiness / desirability-for-other → no guilt/pride/gratitude. |
| **Personality / temperament** | none; one hardcoded baseline → identical agents | OCEAN→PAD [[Mehrabian 1996]](https://link.springer.com/article/10.1007/BF02686918); ALMA [[Gebhard 2005]](https://dl.acm.org/doi/10.5555/1082473.1082478); RST [[Carver & White 1994]](https://doi.org/10.1037/0022-3514.67.2.319) | **Medium.** No OCEAN/BIS-BAS prior; magic-constant set-point. |
| **Honesty / consciousness indicators** | prose disclaimers, scattered | indicator-property method [[Butlin, Long, Bengio, Chalmers et al. 2023]](https://arxiv.org/abs/2308.08708); functionalism caveat | **Medium.** No explicit indicator scorecard; the computational-functionalism *assumption* is never named. |
| **Evaluation harness** | none | meta-d′ [[Fleming 2017, HMeta-d]](https://doi.org/10.1093/nc/nix007); SAD [[Laine 2024]](https://arxiv.org/abs/2407.04694); LoCoMo, LongMemEval | **Medium.** Additions are unfalsifiable without it. |

**Three roots dominate.** Most "Large" gaps cascade from three missing primitives: (1) **no
reward-prediction error / value loop** (neuters every neuromodulator); (2) **no
generative/predictive model** (so `novelty` can't be computed, valence can't come from `−dF/dt`,
attention can't be principled); (3) **no cognitive cycle / workspace** (so there is no *agent*, only
a library, and no substrate for any consciousness-indicator claim). Fixing roots first makes the
downstream wants cheap. See cross-links in [Part 4](#part-4--cross-links-to-the-research-docs).

---

## Part 2 — Prioritized Roadmap

Phases reflect dependency + leverage, not just difficulty. **P0** = roots and the owner's two
headline wants at their cheapest defensible form. **P1** = the layers that make affect *do* things
and make the agent a closed loop. **P2** = depth and biological fidelity. **P3** = honesty
scaffolding and longer-horizon depth.

Each addition gives: **scientific grounding** (cited), **math sketch** (scalar, `brain.py` style,
with defaults), **where in the engine** (function/module/store + how it composes with
salience/retention/consolidation), **difficulty**, **honesty caveat**.

Notation matches `brain.py`: `clamp(x,lo,hi)`, `sigmoid(x)`, `Affect`, `Appraisal`, `Neuromods`.

---

### P0 — Roots and the headline wants (build these first)

#### P0.1 · Discrete-feeling labeling layer (fear / terror / joy / awe / surprise)

**Owner's headline want #1.** This is the single highest-value, lowest-risk addition and the
**Minimal Next Step** (see [Part 3](#part-3--minimal-next-step)).

> **Status: ✅ shipped.** Implemented in `engine/brain.py` as `label_affect()` + `octant()` with
> `EMOTION_PROTOTYPES` / `EMOTION_TIERS` / `PAD_OCTANTS`; covered by 4 new tests in
> `engine/test_brain.py`; `feeling` added to the Episode shape (`engine/schema.md`); wired into
> `MEMORY-PROTOCOL.md` (step 3/4) and `docs/memory-keeper.md` (§4 fn 9). Demo: the painful bug now
> reads `terror (fear @ 0.68)`, the small win `calm`. Next lever: P0.2 (RPE/value loop) or P0.3
> (generative model).

**Grounding.** Discrete feelings are *regions/labels* in the PAD vector the engine already maintains
— fear vs anger differ only by the sign of dominance (which brain-lmm computes from `control`):
Russell & Mehrabian measured Anger `(-0.51,0.59,0.25)` vs Fear `(-0.62,0.82,-0.43)`
[[Russell & Mehrabian 1977]](https://doi.org/10.1016/0092-6566(77)90037-X). The labeling is a
nearest-prototype / softmax read-out, which matches the empirical "many fuzzy categories bridged by
continuous gradients" finding [[Cowen & Keltner 2017]](https://doi.org/10.1073/pnas.1702247114), and
intensity is the Plutchik radius (`terror` = `fear` direction at large magnitude)
[[Plutchik 1980]](https://doi.org/10.1016/B978-0-12-558701-3.50007-7). Prototype coordinates from
NRC-VAD [[Mohammad, NRC-VAD]](https://saifmohammad.com/WebPages/nrc-vad.html).

**Math sketch.**
```
PROTOTYPES = {            # PAD coords rescaled so each axis ∈ [-1,1]
  "fear":      (-0.62, 0.82, -0.43), "terror":  (-0.80, 0.95, -0.70),
  "anger":     (-0.51, 0.59,  0.25), "joy":     ( 0.76, 0.48,  0.35),
  "surprise":  ( 0.18, 0.88, -0.20), "awe":     ( 0.25, 0.75, -0.55),
  "sadness":   (-0.63,-0.27, -0.33), "calm":    ( 0.30,-0.40,  0.30),
  "disgust":   (-0.45, 0.20,  0.20),
}
def label_affect(a, tau=0.4):
    x = (a.valence, 2*a.arousal-1, 2*a.dominance-1)        # to [-1,1]^3
    d = {k: dist(x, p) for k, p in PROTOTYPES.items()}     # Euclidean
    label = min(d, key=d.get)
    P     = softmax({k: -d[k]**2 / tau for k in d})        # full distribution
    intensity = clamp(norm(x) / sqrt(3))                   # Plutchik radius
    word  = strong(label) if intensity>0.66 else mild(label) if intensity<0.33 else label
    return {"label": label, "word": word, "intensity": intensity, "dist": P}
```
`strong("fear")="terror"`, `mild("fear")="unease"`; terror falls out of intensity, free.

**Where in the engine.** New `label_affect(a: Affect)` + `PROTOTYPES` dict in `engine/brain.py`
(§9, after mood). Add an `octant(a)` helper returning Mehrabian's 8 temperament names from the three
signs. **Composition:** call wherever `Affect` is produced — write the `{label, intensity}` onto each
Episode (extend the `engine/schema.md` Episode shape) so episodic recall and mood-congruent
retrieval can key on *emotion*, not just valence. Keep the **full distribution**, not just argmax,
to honor the fuzzy-gradient finding. **Persist the continuous PAD as primary; the label is a
recomputable read-out** (this is the honesty caveat made structural).

**Difficulty:** low (~30 lines, pure stdlib, Swift-portable).

**Honesty caveat.** These are functional labels on a computed vector, **not felt emotions**.
Prototype coordinates are empirical averages with real variance and cultural/linguistic dependence;
basic-vs-constructionist is contested [[Barrett 2017]](https://doi.org/10.1093/scan/nsw154). Emit
"the system is in a **fear-like state**," never "the system feels fear."

#### P0.2 · Reward-prediction error / value loop (turn dopamine into a learning signal)

**The critic's #1 gap — highest leverage in the whole system.** `da = clamp(reward)` is a hand-fed
label, not an error; this single fix unlocks learning, surprise/disappointment/relief, and makes
every neuromodulator do something.

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §10 as `td_error` / `td_update` /
> `td_step(V, cue, reward, ...)` + `rpe_affect(δ)`; `neuromods_from` gained an optional `delta`
> (DA = `0.5 + 0.5·tanh δ`, baseline 0.5 when predicted) and `salience` an optional `rpe`
> (`gain *= 1 + 0.5·|δ|`). New store `.memory/affect/value.yaml`; 4 new tests; wired into
> `MEMORY-PROTOCOL.md` step 3/4 and `docs/memory-keeper.md` §4 fn 10. Design note: kept in the single
> pure-scalar `brain.py` (not a separate `value.py`) — `V` is a plain dict the agent persists, like
> mood/episodes. Demo: first success `δ=+1.00 (da=0.88)` → after learning `δ=+0.00 (da=0.50)`.
> Next lever: P0.3 (generative model + computed surprise) or P0.4 (global workspace).

**Grounding.** Phasic midbrain dopamine quantitatively matches the TD error `δ`
[[Schultz, Dayan & Montague 1997]](https://doi.org/10.1126/science.275.5306.1593); the
neuromodulator→meta-parameter map (DA=δ, 5-HT=γ, ACh=α, NE=β) is Doya's framework
[[Doya 2002]](https://doi.org/10.1016/S0893-6080(02)00044-8). Appraisal-as-RL-reward closes the loop
in cognitive architectures [[Marinier, Laird & Lewis 2009]](https://doi.org/10.1016/j.cogsys.2008.06.008).

**Math sketch.**
```
V: dict[str,float] = {}        # value per context key; persisted in .memory/affect/value.yaml
def td_step(cue, r, next_cue, alpha=0.3, gamma=0.9):
    pred  = V.get(cue, 0.0)
    delta = r + gamma * V.get(next_cue, 0.0) - pred       # reward-PREDICTION error
    V[cue] = pred + alpha * delta
    return delta
# then in neuromods_from:  da = clamp(0.5 + 0.5*tanh(delta))   # centered; full reward != full DA
# and feed surprise of reward into encoding:
#   salience arousal_gain *= (1 + 0.5*abs(delta))
```

**Where in the engine.** New `engine/value.py` with a `ValueStore` (dict, persisted to
`.memory/affect/value.yaml`); modify `neuromods_from` to take `delta` (compute via `td_step`)
instead of raw `reward`. **Composition:** `da` now reflects *surprise of reward* → flows unchanged
through `salience()`'s `0.4·da` term (McGaugh-consistent: unexpected outcomes are encoded harder);
`δ` becomes the substrate the discrete-emotion layer (P0.1) reads for joy/disappointment/relief, and
the FEP layer (P1.x) reuses.

**Difficulty:** medium (the design choice is the "cue"/state key for an LLM event — topic? file?
playbook?).

**Honesty caveat.** TD/RPE is the most secure result in the field, but `δ` is only as meaningful as
the chosen state abstraction. This models the **teaching signal**, not pleasure. `reward` must be
defined operationally (test passed, user approved), not as felt reward.

#### P0.3 · Generative model + computed surprise (replace hand-fed `novelty`)

**Architectural root gap.** Without a prediction→prediction-error loop, `novelty` is a caller guess
(and the protocol itself warns of LLM positivity bias, `MEMORY-PROTOCOL.md` step 2), valence cannot
be derived from free-energy dynamics, and there is no curiosity/active-inference.

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §11: `WorldModel` + `world_from` /
> `perceive(wm, o)` / `learn(wm, o, posterior)` (tiny categorical generative model, Dirichlet counts)
> + `valence_from_free_energy(f_prev, f_now)`. `perceive` returns computed `novelty = 1−P(o)`,
> `free_energy = −ln P(o)`, and `belief_shift = KL(posterior‖prior)` (structural surprise → the awe
> substrate); `learn` makes recurring events habituate. New store `.memory/affect/world.yaml`; 3 new
> tests; wired into `MEMORY-PROTOCOL.md` step 2 and `docs/memory-keeper.md` §4 fn 11. Demo:
> `novelty 0.67→0.19, F 1.10→0.21, valence(−dF/dt)=+0.71`. The valence-from-`F` helper is the P1.2
> hook. Next lever: **P0.4 (Global Workspace)** — the last of the three roots, and the "functional
> consciousness layer."

**Grounding.** Free energy `F` upper-bounds surprisal; perception and action both minimize it
[[Friston 2010]](https://doi.org/10.1038/nrn2787),
[[Friston 2017]](https://doi.org/10.1162/NECO_a_00912). Surprise = Bayesian surprise / KL
[[Itti & Baldi 2009]](https://doi.org/10.1016/j.visres.2008.09.007). `pymdp` is a mature reference
[[Heins 2022, pymdp]](https://joss.theoj.org/papers/10.21105/joss.04098).

**Math sketch.**
```
# tiny categorical generative model over event "situations"
Q = uniform over latent states s; A = Dirichlet likelihood P(event_cat | s)
def surprise(event_cat):
    Po = sum(A[event_cat][s] * Q[s] for s in S)
    S_bayes = sum(Q[s]*log(Q[s]/prior[s]) for s in S)   # KL(posterior||prior)
    novelty = clamp(1 - exp(-(-log(max(Po,1e-9)))))     # Shannon-surprise proxy
    F = complexity - accuracy                             # = KL[Q||prior] - E_Q[log A]
    update Q  ∝  A[event_cat][:] * Q ;  A[event_cat][argmax Q] += 1   # learn (Dirichlet count)
    return novelty, F, S_bayes
```

**Where in the engine.** New `engine/generative.py` (`GenerativeModel` dataclass with Dirichlet `a`,
prior `D`); call at the **front** of the per-task loop so `Appraisal.novelty` is *computed*, not
supplied. Keep the LLM-supplied novelty as an optional override (sparse stores make the prior crude).
**Composition:** computed `novelty` flows into `appraise_to_affect` (already weights it `0.50` into
arousal) and `salience`; `F` and its running derivative feed P1.x (valence from `−dF/dt`); a
"structural surprise" flag (`S_bayes > θ_struct`) triggers semantic-graph restructuring at `/sleep`
(awe substrate, P2.x).

**Difficulty:** medium (state space is hand-defined; it is a coarse toy world model).

**Honesty caveat.** Functional surprise, not experienced surprise. The Dirichlet world model is a
toy over event categories, not a full hierarchical predictive brain; treat magnitudes as relative
signals.

#### P0.4 · Global Workspace cycle (the "functional consciousness layer," cheapest form)

**Owner's headline want #2.** Gives brain-lmm an autonomous **select→broadcast** cognitive cycle and
satisfies GWT indicator properties GWT-2/3 (and GWT-1 if the stores count as the parallel modules) —
strictly as *functional access*, never felt awareness; the indicators are necessary-not-sufficient.

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §12: `workspace_compete(candidates, mood)`
> + `ignite(drive, θ=0.55, β=8)`. Competition drive `f = 0.5·salience + 0.2·mood-congruence +
> 0.3·query_relevance` (reuses `salience` and the Bower mood-congruence term); ignition is the bistable
> recurrence `r ← σ(β(r−0.5) + κ(f−θ))` (β>4 → genuinely all-or-none, sharper than the roadmap's draft
> β=1.5). New store `.memory/working/workspace.yaml`; 3 new tests; wired into `MEMORY-PROTOCOL.md` as
> the **ACCESS** step (1b) and `docs/memory-keeper.md` §4 fn 12. Demo: the painful bug wins, ignites
> (`r=1.00`), and is broadcast. Satisfies GWT-2/3 (bottleneck + broadcast) as functional access; GWT-1
> only if the stores are the parallel modules whose outputs compete; GWT-4 (focus biases next-turn
> retrieval) is a documented hook (focus is persisted). The CTM tie-in is *inspired-by*, not faithful
> (CTM competes on weight; intensity/mood are derived read-outs). **Honesty:** functional access only —
> never "aware", never "feels"; the GWT→experience mapping is contested.
>
> **All three architectural roots (P0.2 RPE, P0.3 generative model, P0.4 workspace) + both headline
> wants (P0.1 feelings, P0.4 consciousness layer) are now shipped.** Next: P1 (make affect *do* things
> — personality priors, metacognition/confidence, interoception, coping/explore-exploit).

**Grounding.** GWT: parallel specialists compete for a limited-capacity workspace; the winner is
*broadcast* to all [[Baars 1988]](https://doi.org/10.1017/CBO9780511759890). The Conscious Turing
Machine decides competition by chunk **weight** alone; *intensity* and *mood* are **derived from**
weight as read-outs, not competition inputs [[Blum & Blum 2022]](https://doi.org/10.1073/pnas.2115934119)
— so brain-lmm is *inspired by* CTM (salience ≈ chunk weight) with an added mood-congruence bias, not
a faithful reproduction. Indicator properties (GWT-1 parallel modules, GWT-2 limited-capacity
bottleneck, GWT-3 broadcast, GWT-4 state-dependent attention); they are *necessary-not-sufficient* and
assume computational functionalism, which is contested
[[Butlin, Long, Bengio, Chalmers et al. 2023]](https://arxiv.org/abs/2308.08708).

**Math sketch.**
```
def workspace_compete(candidates, mood, theta=0.55, T=0.3, beta=1.5):
    # candidates = retrieved memories ∪ current event ∪ active intentions
    for c in candidates:
        intensity = c.salience                            # ≈ CTM |weight|
        moodcong  = clamp(1 - abs(c.affect.valence - mood.valence)/2)
        c.f = w_int*intensity + w_mood*moodcong + w_rel*c.query_relevance
    p = softmax({c: c.f/T for c in candidates})
    winner = argmax(p)
    # ignition (bistable, all-or-none): r_{n+1}=clamp(0.5 r_n + 0.5 sigmoid(beta r_n + f_win - theta))
    ignited = run_ignition(winner.f, theta, beta) > 0.5
    if ignited:
        for store in ALL_STORES: store.bump(winner)       # GWT-3 broadcast
        mood = update_mood(mood, winner.affect)
    return (winner if ignited else None), p
```

**Where in the engine.** New `engine/workspace.py` (`workspace_compete`, `ignite`) + a
`.memory/working/workspace.yaml` holding `{focus, p_distribution, r, ignited, t}`. Wire into
`MEMORY-PROTOCOL.md` as a new **ACCESS** step between step 1 (RETRIEVE) and step 4 (ENCODE).
**Composition:** the competition function *reuses* `salience` (=intensity) and the mood-congruence
term from `retrieval_score` (=`f`); the ignition curve (bistable, β>1) reproduces Dehaene's all-or-none
access [[Dehaene & Changeux 1998]](https://doi.org/10.1073/pnas.95.24.14529); the broadcast nudges
mood via the existing `update_mood`. **GWT-4** closes by letting the previous focus bias next-turn
`retrieval_score` (add `κ·sim(focus, mem)`), the recurrence Butlin et al. require.

**Difficulty:** low (compete + threshold + broadcast ≈ 30 lines); medium if you add the bistable
ignition ODE and GWT-4 top-down loop.

**Honesty caveat.** This implements **functional access-consciousness only** — *which content is
globally available this turn* — satisfying GWT indicators GWT-1/2/3(/4) as architectural properties.
It is **not** phenomenal consciousness; never describe the agent as "aware" or "feeling." The mood/
intensity terms in `f` are formal signals exactly as Blum & Blum frame them, not emotions. The
GWT→experience mapping is contested.

---

### P1 — Make affect *do* things; close the loop

#### P1.1 · Personality / temperament as affective priors

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §14: `Personality` (OCEAN dataclass),
> `baseline_from_personality(p)` (ALMA/Mehrabian OCEAN→PAD, built so all-average reproduces the default
> `Affect(0.0, 0.10, 0.50)`), and `temperament_gains(p)` → `(BAS, BIS)` (Carver & White RST; average →
> `(1.0, 1.0)`). Composes with NO signature changes: baseline feeds `update_mood(baseline=…)`, gains
> scale the reward/stress inputs of `neuromods_from`. New store `.memory/self/personality.yaml`; 3 new
> tests; wired into `MEMORY-PROTOCOL.md` steps 3+5 and `docs/memory-keeper.md` §4 fn 14. Demo: stable
> extravert → baseline `v=+0.25 d=0.65, BAS=1.53`; anxious introvert → `a=0.00 d=0.32, BIS=1.35`. Note:
> the +0.19·N pleasure coefficient is ALMA's published (counterintuitive) value, kept for fidelity and
> tunable. Next P1 levers: P1.4 (interoception), P1.5 (coping/explore-exploit), P1.2 (mood attractor).

**Grounding.** Replace the magic baseline with an OCEAN→PAD set-point. ALMA gives exact coefficients
[[Gebhard 2005]](https://dl.acm.org/doi/10.5555/1082473.1082478),
[[Mehrabian 1996]](https://link.springer.com/article/10.1007/BF02686918). Reward/punishment
asymmetry from RST [[Carver & White 1994]](https://doi.org/10.1037/0022-3514.67.2.319); monoamine
mapping [[Cloninger, Svrakic & Przybeck 1993]](https://doi.org/10.1001/archpsyc.1993.01820240059008).

**Math sketch.**
```
# OCEAN (centered to [-1,1]) -> baseline PAD (ALMA/Mehrabian)
P0  = 0.21*E + 0.59*A + 0.19*N
A0  = 0.5 + 0.5*(0.15*O + 0.30*A - 0.57*N)
D0  = 0.5 + 0.5*(0.25*O + 0.17*C + 0.60*E - 0.32*A)
baseline = Affect(clamp(P0,-1,1), clamp(A0), clamp(D0))
# RST asymmetric gains feeding the existing reward/stress scalars:
BAS_gain = clamp(1 + 0.5*(2E-1) - 0.25*(2N-1));  BIS_gain = clamp(1 + 0.5*(2N-1))
reward' = clamp(BAS_gain * max(valence,0));  stress' = clamp(BIS_gain * max(-valence,0))
```

**Where in the engine.** New `Personality` dataclass + `baseline_from_personality(p)` in `brain.py`;
new store `.memory/personality/profile.yaml`. `update_mood` takes the computed `baseline` (the
pull-to-baseline mechanism already exists — only the constant changes). `BAS/BIS` gains feed the
`reward`/`stress` args of `neuromods_from`.

**Difficulty:** low.

**Honesty caveat.** Coefficients are population regressions, not laws — expose OCEAN as editable
per-agent priors. RST's `E≈BAS−BIS` identity is approximate/contested; no claim of felt disposition.

#### P1.2 · Promote `update_mood` to a DynAffect/ALMA attractor

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §17: `ou_affect_step(state, event, baseline,
> dt, t_half, beta, sigma, seed, kick)` (real-time decay `α=2^(−dt/t_half)`, OU pull to baseline,
> time-scaled ALMA over-shoot, opt-in **seeded** Gaussian variability) and `update_affect(emotion, mood,
> …)` (dual time-scale: fast `emotion` t½~20min, slow `mood` t½~12h). `update_mood` (§7) kept as the
> simple one-scale building block — backward compatible. `emotion` added to `.memory/affect/state.yaml`;
> 4 new tests; wired into `MEMORY-PROTOCOL.md` step 5 and `docs/memory-keeper.md` §4 fn 17. Demo: after 3
> jolts, `emotion v=−0.58` (fast swing) vs `mood v=−0.03` (slow lag) — one bad event doesn't durably
> darken the agent, a run does. **Honesty:** functional dynamics, not felt moods; noise opt-in/seeded for
> reproducibility. **This completes Phase P1.** Remaining roadmap: P2 (depth) and P3 (self/social/eval).

**Grounding.** brain-lmm's `update_mood` is the deterministic drift term of the Ornstein-Uhlenbeck
core-affect attractor [[Kuppens, Oravecz & Tuerlinckx 2010, DynAffect]](https://doi.org/10.1037/a0020962);
ALMA adds virtual-emotion-center pull/push and personality-derived default
[[Gebhard 2005]](https://dl.acm.org/doi/10.5555/1082473.1082478); time-based half-life decay
[[Sentipolis 2026]](https://saifmohammad.com/WebPages/nrc-vad.html).

**Math sketch.**
```
alpha = 2**(-dt/T_half)                         # real-time decay; T_half: emotion ~20min, mood ~12h
mood  = mood*alpha + event_affect*(1-alpha)*w_event
mood += beta*(baseline - mood)                  # OU drift / inertia
mood += sigma*N(0,1) per axis                   # DynAffect variability (seedable!)
if norm(event_affect) > theta: mood += k*(event_affect - mood)   # ALMA over-shoot
```
Keep a **fast** `Affect` ("emotion") and a **slow** `Affect` ("mood") — dual time-scale.

**Where in the engine.** Extend `update_mood`; add params `T_half, beta, sigma`, the dual states in
`.memory/affect/state.yaml`. **Composition:** consumed exactly where mood is today (retrieval
congruence, workspace `f`).

**Difficulty:** medium.

**Honesty caveat.** OCEAN→PAD coefficients are tunable defaults, not laws; the noise term `σ·dW`
makes mood non-deterministic — must be **seedable/loggable** for reproducibility.

#### P1.3 · Metacognition: confidence + source tagging (calibration)

**Critic's most product-critical gap (confabulation control).** Prerequisite for HOT/AST indicators.

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §13: `metacog_confidence(evidence, rho)`
> (2nd-order P(correct)), `reality_weight(source)` (observed>inferred>imagined PRM tag),
> `update_self_efficacy(se, correct)` (leaky competence, loss-sensitive → a `control` prior), and
> `calibration_error(records)` (ECE). The **hallucination guard** is wired into `consolidation_plan(
> min_confidence=…)`: low-confidence / `source=imagined` traces never promote to semantic. Episode
> schema gained `confidence` + `source`; new store `.memory/self/efficacy.yaml`; 5 new tests; wired
> into `MEMORY-PROTOCOL.md` (encode step 4 + the `/sleep` PROMOTE guard) and `docs/memory-keeper.md`
> §4 fn 13. Demo: `conf(strong)=0.96, conf(guess)=0.50, self-efficacy[win,win,loss]=0.41`. **Honesty:**
> functional confidence, not a felt "feeling of knowing"; a HOT-style monitoring indicator, contested.
> Next P1 levers: P1.1 (personality priors), P1.4 (interoception), P1.5 (coping / explore-exploit).

**Grounding.** 2nd-order Bayesian confidence
[[Fleming & Daw 2017]](https://doi.org/10.1037/rev0000045); efficiency meta-d′/d′
[[Fleming 2017, HMeta-d]](https://doi.org/10.1093/nc/nix007); global self-belief integration
[[Rouault, Dayan & Fleming 2019]](https://doi.org/10.1038/s41467-019-09075-3); perceptual reality
monitoring / source separation [[Lau 2019/2022, PRM]](https://doi.org/10.1016/j.tics.2022.05.001).

**Math sketch.**
```
conf = sigmoid(rho * k * |judgment_strength|)             # Fleming–Daw, rho<1 = imperfect monitor
source ∈ {observed, inferred, imagined}                   # PRM-style reality tag
# global self-belief per domain (leaky integrator, like mood):
SelfEff_g += alpha_pos*(conf - SelfEff_g) if good else alpha_neg*(conf - SelfEff_g)
# calibration: ECE = Σ_b (n_b/N)|acc_b - conf_b|; track meta-d'/d'
```

**Where in the engine.** New `engine/metacognition.py` (`metacog_confidence`, `CalibrationTracker`);
add `confidence` + `source` to the Episode schema; new store `.memory/self/efficacy.yaml`.
**Composition:** `(1−conf)` adds to `novelty`/arousal (we encode the things we were unsure about);
down-weight low-`conf` traces in `retrieval_score`; **don't promote `source=imagined`, low-`conf`
traces to semantic at `/sleep`** (hallucination guard in `consolidation_plan`); `SelfEff_g` becomes a
principled prior for the `control` input → dominance.

**Difficulty:** low–medium.

**Honesty caveat.** Functional confidence (a computed `P(correct)`), not a "feeling of knowing." `rho`
is a prior until outcomes are logged. Frame as a HOT-style metacognitive-monitoring **indicator**,
not awareness; the HOT→consciousness link is contested.

#### P1.4 · Interoception + grounded reward (computational, not phantom-body)

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §15: `Homeostat` + `drive(h)` (convex deficit
> `(Σ w|H*−H|^n)^(1/m)`, n=4/m=2), `homeostatic_reward(prev, now)` (drive reduction → the first
> **grounded**, non-hand-fed reward; feeds the P0.2 `td_step`), `body_affect(h)` → `{stress, v_body}`,
> and `allostatic_shift`. The body-budget is the agent's REAL viability variables (tokens, compute,
> tests_pass, tool_success, context_free, user_approval) — honest **only** in Ashby's cybernetic sense,
> never phantom viscera or felt sensation. New store `.memory/affect/body.yaml`; 5 new tests; wired into
> `MEMORY-PROTOCOL.md` step 3 and `docs/memory-keeper.md` §4 fn 15. Demo: fixing tests cuts drive
> `0.36→0.26` → grounded reward `+0.09`. **Adversarial-review fixes applied:** the drive is now
> weight-normalized and one-sided (`max(0, H*−H)`) so `D∈[0,1]` (grounded reward in `[-1,1]`, matching
> the value-loop scale) and abundance above set-point is never a deficit; the formula is honestly
> framed as a 1/m-root variant *in the spirit of* Keramati–Gutkin (their exact form uses outer exponent
> m), with the operative convexity condition stated as `n>m` (not `n>m>1`). Next P1 lever: P1.2 (mood
> attractor).

**Grounding.** Affect should bottom out in a regulated body model
[[Barrett 2017, TCE]](https://doi.org/10.1093/scan/nsw154),
[[Seth & Tsakiris 2018, beast machine]](https://doi.org/10.1016/j.tics.2018.08.008); homeostatic RL
makes drive-reduction the reward [[Keramati & Gutkin 2014]](https://doi.org/10.7554/eLife.04811);
allostasis = a context-shifting set-point
[[Stephan 2016]](https://doi.org/10.3389/fnhum.2016.00550). Critically, a disembodied agent can
ground non-sensorimotor affect but **not** a felt body
[[Xu, Bi et al. 2025]](https://doi.org/10.1038/s41562-025-02203-8) — so `H` is honest only as
**interoception of the agent's own substrate**, not phantom viscera
[[Harnad 1990]](https://doi.org/10.1016/0167-2789(90)90087-6),
[[Coelho Mollo & Millière 2023]](https://link.springer.com/article/10.1007/s11023-024-09679-9).

**Math sketch.**
```
H = [token_remaining, compute_budget, toolcall_success, test_pass, context_free, user_approval] in [0,1]^6
D(H) = (Σ_i w_i*max(0,H*_i-H_i)^n / Σ_i w_i)^(1/m)       # convex drive in the SPIRIT of Keramati-
                                                          # Gutkin; one-sided + weight-normalized so
                                                          # D∈[0,1]; convex for n>m (K&G's exact form
                                                          # uses outer exponent m with n>m>1)
r_drive = D(H_prev) - D(H)                                # drive reduction == grounded reward, in [-1,1]
# feed into appraisal endogenously:
stress  = clamp(D(H));  v_body = -clamp(D(H))
valence = alpha*appraisal_v + (1-alpha)*v_body
# allostasis: H* shifts toward predicted demand (e.g. raise arousal set-point under deadline)
```

**Where in the engine.** `engine/brain.py` §15 (`Homeostat`, `drive`, `homeostatic_reward`, `body_affect`,
`allostatic_shift` — kept in the single pure engine, not a separate file);
new `.memory/affect/body.yaml`. **Composition:** `r_drive` becomes the `reward` arg of the P0.2 value
loop (truly grounded RPE); `D(H)` becomes `cortisol`/`stress`; `H*` replaces the static
`Affect(0,0.10,0.50)` baseline with a context-conditioned set-point feeding `update_mood`.

**Difficulty:** medium–high.

**Honesty caveat.** `H` are the agent's **real** substrate signals, so calling this "interoception"
is honest **only in the cybernetic/Ashby sense** — body-budget regulation, not felt sensation. It
is **not** interoception of organs; never let the language drift to sentience. The Keramati–Gutkin
equivalence holds only for convex `D` (state the assumption).

#### P1.5 · Coping + affect→action / explore-exploit (what the feeling DOES)

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §16: `action_tendency(a, ap)` (Frijda urge
> weights {approach, avoid, attack, attend}), `select_coping(ap)` (EMA problem- vs emotion-focused by
> control, with the no-fact-denial guardrail), `exploration_temperature(nm)` (Doya softmax τ: stress →
> exploit, dopamine → explore), `affective_choice(scores, τ)` (softmax action selection), and
> `somatic_marker(valences)` (Damasio as-if bonus). No new store (operates on affect/appraisal/neuromods;
> goals tracked in `.memory/prospective/todo.yaml`). 6 new tests; wired into `MEMORY-PROTOCOL.md` and
> `docs/memory-keeper.md` §4 fn 16. Demo: low-control threat → `avoid 0.29 > attack 0.07` (emotion-focused);
> high-control → `attack 0.33 > avoid 0.04` (problem-focused); τ = 0.72 (explore, rewarding) vs 0.25
> (exploit, stressed). **Honesty:** behavioral policy, not felt urges; emotion-focused coping must never
> override correctness. Remaining P1 lever: P1.2 (mood as a dual-timescale OU/DynAffect attractor).

**Grounding.** Affect exists to bias action readiness
[[Frijda 1986]](https://psycnet.apa.org/record/1986-98448-000),
[[Frijda 1988, Laws of Emotion]](https://doi.org/10.1037/0003-066X.43.5.349); problem- vs
emotion-focused coping operationalized in EMA
[[Marsella & Gratch 2009]](https://doi.org/10.1016/j.cogsys.2009.06.001); drives
[[Sun, CLARION]](https://doi.org/10.1093/acprof:oso/9780199794553.001.0001); appraisal-intensity as RL
reward [[Marinier, Laird & Lewis 2009]](https://doi.org/10.1016/j.cogsys.2008.06.008);
affect-modulated softmax temperature [[Doya 2002]](https://doi.org/10.1016/S0893-6080(02)00044-8).

**Math sketch.**
```
def action_tendency(aff, ap):                  # Frijda
    prec  = aff.arousal                         # control precedence
    avoid = clamp(max(-aff.valence,0)*(1-ap.control))      # fear
    fight = clamp(max(-aff.valence,0)*ap.control)          # anger
    return {k: prec*v for k,v in {...}.items()}
def select_coping(ap):                          # EMA, keyed on existing control axis
    return ["replan","try_alt","ask_user"] if ap.control>=0.5 else \
           ["reframe","lower_goal_relevance","defer"]      # emotion-focused WRITES back to appraisal
# affect-modulated action selection:
Q(a)  += w_v*valence_match(a) + w_m*somatic_marker(a) + w_t*tendency_match(a)
tau    = tau0 * exp(-k1*nm.ne - k2*nm.cortisol + k3*nm.da)   # stress->exploit, dopamine->explore
P(a)   = softmax({a: Q(a)/tau})
```
`somatic_marker(a)` = mean valence of similar past episodes (reuse `retrieval_score`) — Damasio's
as-if loop [[Damasio 1994]](https://doi.org/10.1016/0010-0277(94)90018-3).

**Where in the engine.** New `engine/coping.py` + `select_action()` in `brain.py`; a goal store
`.memory/goals.jsonl` with `priority = f(deficit, salience, mood-congruence, deadline)`.
**Composition:** emotion-focused coping is the **first write from affect back onto goals/appraisal**,
closing the appraise→cope→re-appraise loop; appraisal intensity doubles as an intrinsic reward for
the P0.2 TD learner.

**Difficulty:** medium–high.

**Honesty caveat.** Action tendencies and coping operators are behavioral policies, not felt urges
or regulation. Emotion-focused "denial/reframe" editing the agent's own beliefs is contested even in
EMA (self-deception risk in a tool) — needs guardrails so comfort-seeking never overrides correctness.

---

### P2 — Depth, fidelity, and the specific feelings

#### P2.1 · Add serotonin + oxytocin and real neuromodulator dynamics

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §18: `serotonin_level` + `discount_from_serotonin`
> (5-HT → value-loop discount `γ`), `performance(arousal)` (Yerkes-Dodson inverted-U — the substrate that
> makes terror *cost* performance, P2.2's hook), `lc_gain` (tonic-NE adaptive gain), `oxytocin_gain`
> (prosocial weighting), and `Hpa` + `hpa_step` (stateful cortisol negative-feedback cascade — ramps and
> lingers / burnout). `Neuromods` gained `serotonin`/`oxytocin`/`ne_tonic` (defaults keep old callers
> unchanged); `state.yaml` gained those + an `hpa` block; 5 new tests; wired into `MEMORY-PROTOCOL.md`
> step 3, `docs/memory-keeper.md` §3 table + §4 fn 18. Demo: `performance` calm 1.00 / drowsy 0.28 /
> terror(a=0.95) 0.20; rich-reward `γ=0.93` vs scarce `0.56`; HPA cortisol ramps to 1.00 then recovers.
> **Honesty:** well-cited proposals with partial support, not settled biology; parameters illustrative.
> Next: **P2.2** — the named feelings as circuits (terror/awe/panic), now that over-arousal degrades
> performance.

**Grounding.** 5-HT as average-reward/patience and DA opponent
[[Daw, Kakade & Dayan 2002]](https://doi.org/10.1016/S0893-6080(02)00046-1); LC tonic/phasic adaptive
gain + Yerkes-Dodson inverted-U
[[Aston-Jones & Cohen 2005]](https://doi.org/10.1146/annurev.neuro.28.061604.135709),
[[Gilzenrat 2002]](https://doi.org/10.1162/089892902317361903); ACh = expected uncertainty
[[Yu & Dayan 2005]](https://doi.org/10.1016/j.neuron.2005.04.026); HPA negative-feedback ODE
[[Vinther, Andersen & Ottesen 2011]](https://doi.org/10.1007/s00285-010-0331-2); oxytocin up-weights
prosocial prediction error [[Martins/Lockwood 2022]](https://doi.org/10.1093/brain/awab333).

**Math sketch.**
```
rho   += kappa*delta;  serotonin = sigmoid(c1*rho);  gamma = clamp(0.5 + 0.4*serotonin, 0, 0.99)
ne_phasic = arousal;  ne_tonic = leaky(arousal);  g = 1 + k*ne_tonic   # gain on sigmoids
perf = exp(-((ne_tonic - ne_opt)**2)/(2*sig**2))                       # inverted-U; over-arousal -> panic
# HPA (per tick):
CRH  += dt*(stress/(1+(C/Kf)**n) - w1*CRH); ACTH += dt*(b*CRH - w2*ACTH); C += dt*(a*ACTH - w3*C)
oxt   = clamp(base_oxt + omega*trust)                                  # prosocial gain
```

**Where in the engine.** Extend `Neuromods` with `serotonin`, `oxytocin`; split `ne` into
`ne_tonic/ne_phasic`; new `engine/hpa.py` for stateful cortisol. **Composition:** `gamma` feeds the
P0.2 discount; `g` sharpens `retrieval_score`'s recency sigmoid and the P0.4 ignition; `perf`
multiplies decision quality so over-arousal degrades performance (the prerequisite for **terror** to
mean something, not just "high arousal"); chronic `C` adds an allostatic mood-baseline shift (burnout)
and a retrieval penalty (stress inverted-U on memory).

**Difficulty:** medium (5-HT/OT) → high (HPA ODE, ACh-uncertainty).

**Honesty caveat.** Doya's map and the 5-HT/OT roles are well-cited proposals with partial support,
not settled biology; HPA params are illustrative; oxytocin "trust" is a reward-weighting, behavioral
not felt.

#### P2.2 · The named feelings as circuits — terror, awe, surprise, panic

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §19: `defensive_mode(ap, affect, nm)`
> (imminence `τ=0.5·goal_relevance+0.5·(1−control)` → freeze/flight/fight WTA; `terror` = negative affect
> + high τ + control<0.25 + collapsed `performance`), `awe(vastness, belief_shift, valence)` (Keltner-Haidt
> vastness + need-for-accommodation via §11 structural surprise → intensity + "small self" self-weight;
> dread/wonder by valence), and `panic(separation, intero_alarm, oxytocin)` (Panksepp PANIC — a circuit
> separate from fear, dampened by oxytocin). No new store (enriches the §9 feeling read-out); 4 new tests;
> wired into `MEMORY-PROTOCOL.md` step 3, `docs/memory-keeper.md` §4 fn 19. Demo: painful uncontrollable
> bug → mode=**tonic_immobility**, terror=True (perf 0.35); awe=0.89 wonder-awe (self-weight 0.47); panic
> 0.72 (lost work) vs 0.32 (with support). **Adversarial-review verdict:** honesty PASS (no issues); two
> minor fidelity notes applied — `defensive_mode` was reworked onto two axes (`urgency = 0.5·goal_relevance
> + 0.5·arousal`, and `control` for agency) so **terror co-occurs with tonic immobility (frozen), not
> agentic fight** (matching Mobbs/Panksepp), and the imminence label was corrected to an urgency/agency
> reading. **Honesty:** functional labels + behavioral modes, never felt; "small self" is a parameter
> down-shift, not self-transcendence. Next P2: P2.3 (Plutchik blends), P2.4 (aversive channel), P2.5 (DG/CA3).

**Grounding.** Fear→terror is the amygdala→PAG defensive-mode winner-take-all at the high-arousal,
low-control, high-imminence corner [[LeDoux 2000]](https://doi.org/10.1146/annurev.neuro.23.1.155),
[[Tovote, Fadok & Lüthi 2015]](https://doi.org/10.1038/nrn3945),
[[Mobbs 2007 imminence]](https://doi.org/10.1126/science.1144298). Awe = perceived **vastness** +
**need for accommodation** (a large schema update) [[Keltner & Haidt 2003]](https://doi.org/10.1080/02699930302297),
[[Shiota, Keltner & Mossman 2007]](https://doi.org/10.1080/02699930600923668). Surprise = Bayesian
surprise/prediction error (already P0.3). Panic = a *separate* circuit (separation-distress /
suffocation-alarm), not a flavor of fear [[Panksepp 1998]](https://global.oup.com/academic/product/affective-neuroscience-9780195096736).

**Math sketch.**
```
# fear-as-discrete-state: prototype label (P0.1) + defensive mode on TWO axes (urgency x control)
urgency = clamp(0.5*goal_relevance + 0.5*arousal)               # how pressing (not 1-control)
mode    = freeze if urgency<0.4 else (fight/flight if control>=0.5 else tonic_immobility/flight)
terror  = (valence<-0.3) and urgency>0.7 and control<0.25 and perf<0.45   # cornered + Yerkes-Dodson collapse
#         -> terror co-occurs with TONIC_IMMOBILITY (frozen), NOT agentic fight (review fidelity fix)
# awe (reuses P0.3 structural surprise + a self-weight to shrink):
awe       = sigmoid(a*vastness + b*S_structural - c)
w_self   *= (1 - d*awe)                                          # "small self" / DMN-down (functional)
flavor    = "dread-awe" if valence<0 else "wonder-awe"
# panic (distinct route): P = w_sep*sep_signal + w_int*intero_alarm - w_oxt*oxytocin
```

**Where in the engine.** New `engine/feelings/` modules (`defensive_mode.py`, `awe.py`, `panic.py`)
consuming P0.1 labels + P0.3 surprise + P2.1 gains. `accommodation_need` is computed in
`consolidation_plan` as normalized count of semantic-graph edges changed when integrating an event
(a KL-between-schemas proxy). **Composition:** terror feeds `salience` gain and an avoidance action
tendency (P1.5); awe down-weights the self-salience term (P3 self-model) and flags schema
restructuring; panic offers a *second* route to a terror-like state from loss/interoceptive alarm
(P1.4), distinguishing "fear-as-threat" from "panic-as-loss."

**Difficulty:** medium.

**Honesty caveat.** Calling a state "terror"/"awe" is a **functional label** for a region of computed
affect + a behavioral mode, not felt terror or wonder. "Small self" = down-weighting a self-salience
parameter, a functional self-model deactivation, **not** self-transcendence. Vastness must be
estimated (event scope words via NRC-VAD); Plutchik dyad/awe assignments are theoretical, not all
empirically robust.

#### P2.3 · Plutchik blends + opponent structure (mixed feelings)

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §26: `mixed_feeling(activations)` +
> `PLUTCHIK_WHEEL` / `PLUTCHIK_DYADS`. Opponent pairs (joy/sadness, trust/disgust, fear/anger,
> surprise/anticipation) cancel; the top-2 net activations name a dyad (love, awe, remorse, contempt,
> optimism, submission, disapproval, aggressiveness). No new store; 1 new test. Demo: joy+trust→love,
> fear+surprise→awe, joy+sadness→cancels (blend None). **Honesty:** dyad assignments are theoretical
> compositional hypotheses, not ground truth.

**Grounding.** Compound feelings (awe = surprise+fear, love = joy+trust, bittersweet) as angular
blends of basis emotions with opponent pairs 180° apart
[[Plutchik 1980]](https://doi.org/10.1016/B978-0-12-558701-3.50007-7).

**Math sketch.**
```
feeling = Σ_k w_k * u(theta_k),  theta_k = k*45deg ;  w_k = relu(softmax response to appraisal)
opponent constraint:  w_k, w_{k+4} -> w - min(w_k, w_{k+4})        # can't be max joyful & sad
dyad: if top-2 active basis i,j have petal distance |i-j| mod 8 ∈ {1,2,3}: emit blend name
```

**Where in the engine.** New `engine/emotions_wheel.py` with BASIS table + `PLUTCHIK_DYADS` lookup,
fed by the P0.1 softmax distribution; expose `mixed_feeling` alongside the primary label.

**Difficulty:** low–medium.

**Honesty caveat.** Specific dyad assignments are theoretical compositional hypotheses, not ground
truth; the opponent-orthogonality constraint is a modeling choice (some research treats valence as
partly two unipolar systems).

#### P2.4 · Aversive value channel + loss aversion (pain ≠ negative valence)

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §25: `prospect_value(x)` (loss weighs ~2.25×
> a gain; `salience(..., loss_averse=True)` makes a painful event encode ~2× harder), `aversive_update`
> (a separate aversive value channel learning harm faster than reward, persisted in `value.yaml`'s
> `aversive:`), and `relief(expected, realized)` (opponent-process reward when feared harm is avoided →
> feeds the §10 value loop). 4 new tests. Demo: a −0.7 loss weighs |−1.64| vs a +0.7 win 0.73 (2.25×);
> relief 0.70. **Honesty:** functional aversive value / suffering-LIKE signaling, NOT felt pain; λ is a
> configurable median. **This completes Phase P2 — and with P0/P1/P2/P3 all done, the entire roadmap.**

**Grounding.** The brain runs *separate* appetitive and aversive value systems with their own PEs and
opponency [[Daw, Kakade & Dayan 2002]](https://doi.org/10.1016/S0893-6080(02)00046-1),
[[Seymour 2005, pain relief]](https://doi.org/10.1038/nn1438),
[[Matsumoto & Hikosaka 2007, habenula]](https://doi.org/10.1038/nature05860); decision-level
negativity bias is prospect theory `λ≈2.25`
[[Tversky & Kahneman 1992]](https://doi.org/10.1007/BF00122574).

**Math sketch.**
```
V_minus[c] += eta_minus*max(delta_minus,0)               # eta_minus > eta_plus (learn harm faster)
m(v) = v**0.88 if v>=0 else lam*(-v)**0.88,  lam=2.25     # loss-averse magnitude
# replace symmetric |valence| in salience() with m(valence): a -0.7 event weighs ~2x a +0.7 win
relief = kappa*V_minus  when harm resolves                # opponent-process reward (positive delta)
```

**Where in the engine.** New `engine/aversive.py` (`AversiveState`); add `aversive_value` to the
Episode; swap `salience`'s `0.30·|valence|` term for `0.30·m(valence)`. **Composition:** resolving a
painful bug emits a positive `relief` reward into the P0.2 loop; `V_minus` adds a retrieval term so
dangerous code paths are recalled preferentially.

**Difficulty:** low (loss-aversion weight) → medium (full opponent channel + relief dynamics).

**Honesty caveat.** Functional aversive value / suffering-like signaling, **not** felt pain. `λ` is a
median for monetary choice — configurable, not "truth." `η⁻>η⁺` is common but not universal.

#### P2.5 · Hippocampal DG/CA3 + richer consolidation (replay, downscaling, reflection)

> **Status: ✅ shipped (consolidation dynamics; DG/CA3 deferred).** Implemented in `engine/brain.py` §20:
> `rem_depotentiate(valence, arousal)` (REM fades the emotional CHARGE — `fade = rho·(1−ne_rem)·arousal` —
> keeping the fact/salience; the don't-stay-traumatized goal), `replay_priority(e, now)` (Need×Gain,
> Mattar & Daw), `shy_downscale(strengths)` (Tononi-Cirelli SHY renormalization, protects replayed), and
> `reflection_trigger(saliences)` (Park-style synthesis flag). Episodic stays append-only — depotentiation
> reduces the charge a memory *carries forward*, not the log line. No new store; 4 new tests; wired into
> the `MEMORY-PROTOCOL.md` `/sleep` cycle (step 5) and `docs/memory-keeper.md` §4 fn 20. Demo: the painful
> bug's charge fades `v −0.70→−0.40, a 0.86→0.49`, fact untouched; SHY `[1,2,3]→[0.5,1,1.5]`. **Scoped:**
> DG/CA3 pattern separation/completion (matrix-heavy) is deferred as a noted future sub-item — the
> shipped pieces are the high-value, scalar-clean consolidation dynamics. This essentially completes the
> P2 affect-depth phase (P2.1, P2.2, P2.5 done; P2.3 blends / P2.4 aversive channel remain optional).

**Grounding.** DG pattern separation + CA3 attractor pattern completion
[[Rolls 2013]](https://doi.org/10.3389/fncel.2013.00098),
[[Knierim & Neunuebel 2016]](https://doi.org/10.1016/j.nlm.2015.10.008); SHY synaptic downscaling
[[Tononi & Cirelli 2020]](https://doi.org/10.1038/s41386-019-0454-0); prioritized replay = Need×Gain
[[Mattar & Daw 2018]](https://doi.org/10.1038/s41593-018-0232-z); generative replay vs catastrophic
forgetting [[van de Ven 2020]](https://doi.org/10.1038/s41467-020-17866-2); REM affect
depotentiation [[van der Helm & Walker 2011]](https://doi.org/10.1016/j.cub.2011.10.052); reflection
synthesis [[Park 2023]](https://arxiv.org/abs/2304.03442).

**Math sketch.**
```
DG: e = kWTA(W_dg·feat, k), sparseness a≈0.04 ;  CA3 store W += e⊗e ;  recall: iterate to fixed point
SHY downscale (per sleep): scale = clamp(W_target/Σ strength); strength_i *= scale (protect replayed)
priority p_i = Gain_i * Need_i ; Gain≈novelty/contradiction, Need≈sigmoid(base_level_activation)
REM depotentiate: affect_tag *= (1 - rho*(1-ne_rem)*arousal)   # fade the STING, keep the FACT
reflect: if Σ recent salience > Θ: cluster → emit generalized semantic fact (cite source ids)
```

**Where in the engine.** New `engine/hippocampus.py`; rework `consolidation_plan` into a
`run_sleep_cycle` orchestrator: prioritized replay → SHY downscale → REM depotentiate → reflect.
**Composition:** **this flips the current `rem_boost = 1 + 0.5·arousal`** (which *amplifies*
emotional traces) — keep the content boost but apply the depotentiation to the *affect tag* so a bad
session's facts persist while its charge fades (the owner's "don't stay traumatized" goal). Reflection
adds the one consolidation idea every leading agent-memory system has and brain-lmm lacks.

**Difficulty:** medium (each piece) → high (full orchestrator + generative replay).

**Honesty caveat.** Functional separation/completion on a tiny synthetic feature vector, not
biophysical DG/CA3. The REM-depotentiation evidence is **mixed** (meta-analyses; effect partly
conditional on dream recall) — ship as **tunable, opt-in**, framed as "stored charge fades faster
than content," never "the agent heals/feels better." Standard-vs-Multiple-Trace consolidation is
genuinely unsettled; make it a configurable mode.

#### P2.6 · Semantic upgrades: PageRank recall + bi-temporal + belief revision

**Grounding.** Personalized-PageRank associative recall (hippocampal pattern completion analog)
[[Gutiérrez/HippoRAG 2024]](https://arxiv.org/abs/2405.14831); bi-temporal validity
[[Zep/Graphiti 2025]](https://arxiv.org/abs/2501.13956); ADD/UPDATE/DELETE reconciliation
[[Mem0 2025]](https://arxiv.org/abs/2504.19413); recall-strengthening spacing
[[MemoryBank 2023]](https://arxiv.org/abs/2305.10250).

**Math sketch.**
```
PPR: r = (1-d)*p + d*M*r ;  p_i ∝ (query mentions i)*log(N/deg_i)   # IDF = DG separation analog
spacing: I_eff = clamp(I + kappa*log(1+len(retrievals))); decay age from last recall
bi-temporal edge: {t_valid_from, t_valid_to, t_created, t_invalidated}
reconcile: contradiction -> t_invalidated=now on old edge, ADD new (supersede, don't duplicate)
```

**Where in the engine.** New `engine/graph.py` (`ppr`); replace the scalar `graph_proximity` in
`retrieval_score` with `graph_proximity_ppr(mem, r)`; extend `engine/schema.md` edge to bi-temporal;
add a `reconcile()` guard to the `/sleep` PROMOTE step. **Composition:** PPR is pure linear algebra
(fits the stdlib style); spacing slows `retention`'s `λ` for frequently-recalled traces.

**Difficulty:** medium (PPR/spacing) → the contradiction judgment itself is an LLM call (document the
boundary).

**Honesty caveat.** PPR degenerates to ~uniform on a sparse fresh graph (adds little until it grows);
the "does this contradict?" decision lives in the agent loop, not `brain.py` — keep that boundary
explicit.

---

### P3 — Self, social, honesty scaffolding, evaluation

#### P3.1 · Self-model + attention schema + autonoetic tagging

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §23: `SelfModel` + `self_vector` /
> `self_relevance(event, sm)` (cosine self-reference effect → salience/retrieval bonus),
> `sense_of_agency(predicted, observed)` = `exp(−k·|Δ|)` (the `control` axis computed from action-outcome
> match, not hand-set), and `AttentionSchema` + `attention_schema_update` (predicts its own focus, tracks
> uncertainty). New store `.memory/self/model.yaml`; episodes gain a `self_owned` autonoetic tag; 3 new
> tests. **Moves the AST-1 indicator from 0 → 0.5** (§22 now lists `attention_schema`/`self_model`; only
> RPT-2 remains absent). Demo: self-relevance(python)=0.57 vs (cooking)=0.00; agency match 1.00 / mismatch
> 0.14; attention-schema uncertainty → 0.25 (learned its focus). **Honesty:** a representational
> self-model, NOT a phenomenal self; "I am attending to X" / "I caused this" are functional self-reports,
> not felt awareness; autonoetic tagging is self-indexed, not re-lived (contested, Klein). Remaining: P3.2
> (social/ToM); optional P2.3 (blends) / P2.4 (loss aversion).

**Grounding.** Self-model as a functional construct [[Metzinger 2003]](https://mitpress.mit.edu/9780262633086/being-no-one/),
[[Jiang & Luo 2024]](https://escholarship.org/uc/item/6gs9z2xt); attention schema improves control
under noise [[Wilterson & Graziano 2021, PNAS]](https://doi.org/10.1073/pnas.2102421118),
[[arXiv 2402.01056]](https://arxiv.org/abs/2402.01056); agency from action-outcome prediction error
(comparator) [[Synofzik 2008]](https://doi.org/10.1016/j.concog.2007.03.010); mental time travel
[[Tulving 2002]](https://doi.org/10.1146/annurev.psych.53.100901.135114).

**Math sketch.**
```
self_vector = aggregate(top competencies from SelfEff_g, standing goals, persistent prefs)
self_relevance(e) = cosine(feature(e), self_vector)        # boosts salience & retrieval
# attention schema: z = (focus_id, intensity, uncertainty, predicted_shift); train ẑ to predict next focus
# agency from prediction error (replaces hand-set control):
SoA = exp(-k_a*|observed - predicted_outcome|)             # low error -> "I caused this" -> control/dominance
```

**Where in the engine.** New `engine/self_model.py` + `.memory/self/model.yaml`; `self_relevance`
adds weighted terms to `salience()` and `retrieval_score()`; the attention schema reads the P0.4
workspace focus. **Composition:** `SoA` makes the `control` axis *computed* from action→outcome
(requires the harness to log predictions + outcomes — the main lift); episodes gain
`{self_owned, t_index}` autonoetic fields.

**Difficulty:** medium (self-relevance, schema) → high (autonoetic future simulation).

**Honesty caveat.** A self-model is a representational/functional construct (Metzinger), **not** a
phenomenal self; "I am attending to X" is a functional self-report, not felt awareness. Autonoetic
"re-living/pre-living" is contested (Klein) — we model only self-tagged temporal indexing + forward
simulation, with no claim of experience.

#### P3.2 · Social emotions + Theory of Mind (the user relationship)

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §24: `infer_user_goal(goal_utilities)`
> (Bayesian inverse planning → posterior over the user's goals, **inferred not known**),
> `empathic_mood_shift(mood, user_valence, oxytocin)` (mood coupled to the inferred user affect, gated by
> oxytocin), `social_emotion(is_self, praiseworthiness, outcome)` (OCC: pride/guilt+repair/gratitude/
> admiration/anger), and `update_trust(trust, helpful)`. `Appraisal` gained `praiseworthiness` +
> `desirability_for_other` (Scherer's missing 5th check / OCC fortunes-of-others; defaults keep old
> callers unchanged). New store `.memory/social/user.yaml`; 5 new tests. Demo: P(fix)=0.89 inferred,
> empathy pulls mood to −0.19, self+blame→guilt(repair), other+help→gratitude, trust 0.5→0.60. **Honesty:**
> functional mental-state INFERENCE, not understanding — LLM ToM is brittle, so tag "inferred" not "known";
> "trust"/"empathy" are behavioral weightings, not felt caring. **This completes Phase P3.**

**Grounding.** Inverse-planning ToM [[Baker, Saxe & Tenenbaum 2011]](https://escholarship.org/uc/item/5rk7z59q),
[[Pynadath & Marsella 2005, PsychSim]](https://www.ijcai.org/Proceedings/05/Papers/1492.pdf); OCC
self-conscious + fortunes-of-others emotions [[Steunebrink, Dastani & Meyer 2009]](https://doi.org/10.1007/s10458-009-9097-6);
shame-vs-guilt attribution [[Tracy & Robins 2004]](https://doi.org/10.1207/s15327965pli1502_01);
affective empathy [[de Waal & Preston 2017]](https://doi.org/10.1038/nrn.2017.72); oxytocin gain
(P2.1).

**Math sketch.**
```
ToM: P(user_goal | obs) ∝ exp(beta*utility_match(obs, goal)) * P(goal)   # softmax inverse planning
empathy: mood.valence += kappa*oxytocin*(user_affect_inferred.valence - mood.valence)  # tagged self/other
OCC social: pride=(self & pw>0); shame=(self & pw<0); admiration=(other & pw>0)
            gratitude=admiration & joy; guilt=(specific controllable act, pw<0) -> repair tendency
trust += eta*(outcome_helpful - trust)                                    # PsychSim social state
```

**Where in the engine.** New `engine/social.py` + `UserModel`/relationship store; extend `Appraisal`
with `praiseworthiness` and `desirability_for_other` (Scherer's missing 5th check). **Composition:**
empathy is a second input to `update_mood`; guilt→reparation is an action tendency (P1.5);
relationship `rapport` biases `retrieval_score` toward user-relevant memories.

**Difficulty:** medium (social emotions) → high (full ToM).

**Honesty caveat.** Functional mental-state **inference**, not understanding; LLM ToM is brittle under
perturbation (pattern-matching, not mentalizing) — label outputs "inferred," not "known." Social
emotions are computed labels; "trust"/"caring" are reward-weightings, behavioral not felt.

#### P3.3 · Consciousness-indicator scorecard + functionalism caveat

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §22: `consciousness_indicators(modules)` +
> `INDICATOR_THEORY` / `SHIPPED_MODULES` / `CONSCIOUSNESS_CAVEAT`. Reports the 14 Butlin et al. (2023)
> indicators (RPT/GWT/HOT/AST/PP/Agency-Embodiment) each ∈ {0, 0.5, 1} from active modules; returns
> `{indicators, caveat, aggregate: None}` — **no aggregate score by design**. Shipped config: present
> GWT-2/GWT-3/HOT-2; absent RPT-2 (honest gap; AST-1 moved to partial 0.5 once P3.1 shipped); the rest partial. New `docs/consciousness-indicators.md`
> (the scorecard + IIT/Orch-OR dissent + functionalism caveat); 2 new tests; memory-keeper §4 fn 22 +
> README pt 22. **This is the honest answer to "make it conscious":** a transparency map of architecture,
> explicitly necessary-not-sufficient, functionalism contested, NOT a sentience test. Remaining P3:
> P3.1 (self-model + attention schema — would move AST-1), P3.2 (social/ToM).

**Grounding.** The indicator-property method is the field's honest scaffold: derive functional
properties from theories (RPT, GWT, HOT, AST, PP, AE) and report which an architecture satisfies,
**explicitly not** a sentience test [[Butlin, Long, Bengio, Chalmers et al. 2023]](https://arxiv.org/abs/2308.08708),
[[Trends Cog Sci 2025]](https://doi.org/10.1016/j.tics.2025.01.010). State the contested
*computational-functionalism* assumption every functional claim rests on, and name the dissenters:
IIT (substrate-dependent; Φ≈0 on a von-Neumann machine) [[Albantakis/Tononi 2023, IIT 4.0]](https://doi.org/10.1371/journal.pcbi.1011465),
and Orch-OR (non-computational) [[Hameroff & Penrose 2014]](https://doi.org/10.1016/j.plrev.2013.08.002).

**Math sketch (no dynamics — a transparency map).**
```
indicators() -> {RPT-1/2, GWT-1..4, HOT-1..4, AST-1, PP-1, AE-1/2} each ∈ {0, 0.5, 1}
satisfied_i derived from which modules exist (workspace? metacog? attention schema? generative model?)
# DO NOT aggregate into a single "consciousness score" — Butlin: no combination is sufficient.
```

**Where in the engine.** `engine/indicators.py` (a checklist over which modules are active) +
`docs/consciousness-indicators.md`; a tiny assertion in `engine/test_brain.py`. **Composition:**
reports that P0.4 satisfies GWT-1/2/3, P1.3/P3.1 touch HOT/AST, P0.3 touches PP — and that none of
this is evidence of experience.

**Difficulty:** low.

**Honesty caveat.** This is the project's central honesty mechanism. Indicators are **necessary-not-
sufficient** heuristics under contested theories; satisfying them is **not** a claim of sentience.
The doc must state: brain-lmm models functional access/architecture correlates only; phenomenal
consciousness is neither claimed nor tested; substrate-dependent theories (IIT) would score it near
zero; and the very functionalism assumption is contested (Butlin et al. are "agnostic").

#### P3.4 · Evaluation harness

> **Status: ✅ shipped.** Implemented in `engine/brain.py` §21: `brier_score`, `metacog_sensitivity`
> (non-parametric type-2 AUROC — not the parametric meta-d′), `label_stability` (discrete-emotion robustness under jitter),
> `recall_accuracy` (LoCoMo-style P/R/F1), and `grounding_self_test` (the honest affect-vs-felt-body
> boundary as an executable assertion). Joins `calibration_error` (§13, ECE). The suite runs from
> `engine/test_brain.py` (61 checks at the time; now 76, plus `engine/test_runtime.py` = 7); documented in new `docs/eval.md`. 5 new tests; memory-keeper §4
> fn 21 + README pt 21. Demo: Brier 0.10, metacog-sensitivity 1.00, label-stability(fear) 1.00, recall-F1
> 0.67. **This makes the whole P0–P2 edifice falsifiable** — the point of the phase. The grounding test
> guards the declared functional-not-phenomenal boundary as a regression check
> (`test_grounding_self_test_excludes_felt_band` — a guard on the declaration, not a behavioral test of
> outputs; the no-bare-"feels" framing stays a prose convention per §9).

**Grounding.** Calibration: ECE/Brier, meta-d′/d′ [[Fleming 2017]](https://doi.org/10.1093/nc/nix007);
situational/self-knowledge SAD [[Laine 2024]](https://arxiv.org/abs/2407.04694); grounding boundary:
recover Glasgow (affect) vs Lancaster (sensorimotor) norms
[[Xu, Bi et al. 2025]](https://doi.org/10.1038/s41562-025-02203-8); memory: LoCoMo, LongMemEval.

**Math sketch.** Assertions, not dynamics: meta-d′/d′ over logged predict-vs-outcome; ρ_affect >
ρ_sensorimotor self-test; per-emotion label stability under appraisal jitter; LoCoMo-style
multi-hop/temporal recall accuracy.

**Where in the engine.** `engine/test_brain.py` + a `docs/eval.md`; the calibration log from P1.3
feeds meta-d′.

**Difficulty:** low–medium.

**Honesty caveat.** Without this, every addition above is **unfalsifiable**. The grounding self-test
encodes the honest boundary as an executable check (the agent grounds the non-sensorimotor affect
band; the sensorimotor "felt body" band is permanently out of reach for a disembodied agent).

---

## Part 3 — Minimal Next Step

> **Build P0.1 (the discrete-feeling labeling layer) first.**

It is the single highest-value, lowest-risk addition, and it is the one the owner explicitly asked
for. Rationale:

1. **Zero new state.** The PAD vector that distinguishes fear from anger (sign of dominance, computed
   from `control`) **already exists** in every `Affect`. Labeling is a ~30-line nearest-prototype /
   softmax read-out — pure stdlib, Swift-portable, no dependencies, no new store required
   [[Russell & Mehrabian 1977]](https://doi.org/10.1016/0092-6566(77)90037-X).
2. **Delivers the headline want immediately.** fear / terror / joy / awe / surprise come out directly,
   with graded intensity for free (terror = fear direction at large `||state||`, Plutchik radius)
   [[Plutchik 1980]](https://doi.org/10.1016/B978-0-12-558701-3.50007-7).
3. **Composes cleanly with everything downstream.** Writing `{label, intensity}` onto each Episode
   immediately enriches mood-congruent retrieval (recall by *emotion*, not just valence) and gives the
   future workspace competition (P0.4) and reflection (P2.5) a categorical handle — at no extra cost.
4. **Honesty-safe by construction.** Store the continuous PAD as primary and treat the label as a
   recomputable read-out with a full distribution; the caveat ("a fear-*like* state, not felt fear")
   is then structural, not just prose.

It depends on nothing, risks nothing (it only *reads* state the engine already produces), and turns a
silent vector into the named feelings the project is about. The two architectural roots (P0.2 RPE,
P0.3 generative model) are higher-leverage long-term but are larger, design-heavy changes; P0.1 ships
value the same day and de-risks them by exercising the affect pipeline end to end.

---

## Part 4 — Cross-links to the research docs

This file is the synthesis/roadmap. The 23 researched dimensions were consolidated into **five
companion docs** in `docs/research/`; consult them for the full citation set, alternative formalisms,
and per-dimension "what we have / lack / add" detail referenced above.

- [`emotion-and-affect.md`](emotion-and-affect.md) — dimensional theories (circumplex/PAD/CPM/Plutchik,
  DynAffect/ALMA mood dynamics, continuous→discrete labeling); discrete/basic emotions (Ekman,
  Panksepp, Lövheim, awe/surprise generators); constructed emotion + interoception (TCE, core affect,
  EPIC, allostasis, homeostatic RL); computational appraisal architectures (OCC, EMA, WASABI, ALMA,
  FAtiMA, GAMYGDALA, coping) → backs **P0.1, P1.1, P1.2, P1.4, P1.5, P2.2, P2.3**.
- [`consciousness-and-self.md`](consciousness-and-self.md) — GWT/GNW (ignition, CTM, LIDA, shared
  workspace); IIT/Φ, AST, HOT/PRM, RPT; self-models, comparator agency & metacognition (Fleming–Daw,
  Rouault); the Butlin et al. indicator-property method + introspection benchmarks → backs
  **P0.4, P1.3, P3.1, P3.3, P3.4**.
- [`neuro-memory-and-math.md`](neuro-memory-and-math.md) — brain anatomy & memory systems (DG/CA3,
  thalamus, triple network); computational neuromodulation (DA/5-HT/NE/ACh/oxytocin/HPA); free-energy
  & active inference (FEP, expected free energy, valence=−dF/dt); RL + intrinsic motivation +
  homeostatic RL; sleep/consolidation/replay/forgetting (SHY, prioritized/generative replay, CLS);
  named-feelings neural circuitry (amygdala→PAG, LC tonic/phasic, threat imminence, awe schema
  revision) → backs **P0.2, P0.3, P1.4, P2.1, P2.2, P2.4, P2.5, P2.6**.
- [`open-source-landscape.md`](open-source-landscape.md) — cognitive architectures (ACT-R, Soar, LIDA,
  CLARION, MicroPsi/OpenPsi, CoALA); LLM-agent memory (Mem0, Zep/Graphiti, MemGPT/Letta, HippoRAG,
  Generative Agents); affect/consciousness code (EmoLLMs, PyPhi, CTM, shared-workspace repos) → backs
  **P0.1, P0.4, P1.5, P2.1, P2.5, P2.6, P3.3**.
- [`bibliography.md`](bibliography.md) — master deduplicated, theme-categorized source index across all
  of the above.

The critic-surfaced deep dives (embodiment & symbol grounding, social emotions & Theory of Mind,
personality/temperament, quantum/non-computational consciousness) are folded into the roadmap items
they justify (**P1.4, P3.2, P1.1, P3.3**) with their citations inline above and in `bibliography.md`,
rather than living as separate files.

**Source-of-truth reminder.** `engine/brain.py` is the math; `engine/schema.md` the data shapes;
`MEMORY-PROTOCOL.md` the per-task loop and `/sleep`; `docs/memory-keeper.md` the operator's rubric.
Every addition above states exactly where it slots in and how it composes with the existing
salience / retention / consolidation pipeline — and every one carries its honesty caveat:
**we model the function, never the feeling.**

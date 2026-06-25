# Neuroanatomy, Neuromodulation, Active Inference & Memory Dynamics - State of the Art

> **Audience:** a technical builder extending `brain-llm`.
> **Honesty stance (carried throughout):** everything below is **functional**, not **phenomenal**. Where this report writes "valence," "feeling," "fear," "surprise," or "subjective fitness," it means a *computed signal that behaves like* the named quantity in how it modulates memory, learning, and behaviour. None of it is a claim about felt experience or sentience. This mirrors the stance already baked into `src/brain.py`: *"This reproduces the function of the brain's memory + affect machinery, not the subjective experience."* The neuroscience itself is also genuinely unsettled in places (e.g. Standard-vs-Multiple-Trace consolidation, whether the Free-Energy Principle is falsifiable, the heterogeneity of serotonin) - those debates are flagged where they arise.

This document surveys the relevant state of the art across five linked areas, gives the math/formalism inline for every model, and ends each major section with a **brain-llm mapping** (what we have / what we lack / what to add). The final section consolidates concrete build proposals with equations.

---

## 0. Where `brain-llm` stands today (baseline)

`src/brain.py` is a single global pipeline of eight scalar functions:

1. `appraise_to_affect` - OCC-style appraisal → PAD affect (valence/arousal/dominance).
2. `neuromods_from` - four static gains: `ne=arousal`, `da=clamp(reward)`, `ach=1.0|0.1` (wake/NREM flag), `cortisol=clamp(stress)`.
3. `salience` - McGaugh arousal-gain on a four-axis base value.
4. `base_level_activation` - ACT-R `B_i = ln(Σ_k (now − t_k)^(−d))`.
5. `retention` - FadeMem/Ebbinghaus `v(t) = v0·exp(−λ(t−τ)^β)`, `λ = λ_base·exp(−μI)`.
6. `retrieval_score` - linear blend of recency + salience + relevance + graph proximity + mood congruence.
7. `update_mood` - leaky integrator toward baseline `Affect(0, 0.1, 0.5)`.
8. `consolidation_plan` - CLS promote/forget pass with a REM arousal boost.

This is a **competent functional model of a unitary memory + mood loop**, with classic, citable equations. Its central limitation is twofold: (a) anatomically it names ~14 structures (in `docs/memory-keeper.md`) but implements region-specific computation for almost none of them; and (b) every neuromodulator and affect axis is a **static scalar read off the current event**, with no prediction, no learning loop, and no dynamics. The rest of this report is organised around closing those two gaps.

---

## 1. Brain anatomy & memory systems

### 1.1 The memory taxonomy

The canonical division [Squire 2004](http://whoville.ucsd.edu/PDFs/384_Squire_%20NeurobiolLearnMem2004.pdf) splits long-term memory into:

- **Declarative (explicit):** episodic + semantic, medial-temporal-lobe / hippocampus dependent.
- **Non-declarative (implicit):** procedural skills (striatum/cerebellum), priming, classical/Pavlovian conditioning, perceptual learning.

Retrieval is itself **dual-process**: *recollection* (hippocampus) vs *familiarity* (perirhinal cortex), often modelled with ROC/signal-detection analysis: `P(old) = R + F`, where `R` is recollection probability and `F` is familiarity as a signal-detection `d′`.

### 1.2 Hippocampal microcircuit - pattern separation & completion

The most consequential omission in any flat-vector memory model. The canonical computational theory [Knierim & Neunuebel 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792674/); [Rolls 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3691555/) assigns:

- **Dentate gyrus = pattern separation.** Sparse, orthogonalizing recoding of entorhinal input. Sparseness `a` = fraction active ≈ 0.01–0.05; similar experiences get non-overlapping codes, reducing interference.
- **CA3 = pattern completion.** An autoassociative (Hopfield-style) attractor network whose recurrent collaterals reconstruct a complete memory from a partial cue:

```
recurrent weights:  w_ij = Σ_p η_i^p η_j^p        (Hebbian over patterns p)
recall dynamics:    x_i(t+1) = f( Σ_j w_ij x_j(t) )   → energy-descent to nearest attractor
capacity:           p_max ≈ C / (a · ln(1/a))     (C = recurrent synapses per cell)
```

CA3 toggles between separation and completion depending on input strength vs internal attractor strength.

### 1.3 Complementary Learning Systems (CLS)

[McClelland, McNaughton & O'Reilly 1995](https://doi.org/10.1037/0033-295X.102.3.419); updated [Kumaran, Hassabis & McClelland 2016](https://doi.org/10.1016/j.tics.2016.05.004): a **fast, sparse, one-shot hippocampus** (`η_H`) and a **slow, overlapping, statistical neocortex** (`η_C ≪ η_H`). Interleaved hippocampal replay during sleep trains cortex gradually, avoiding catastrophic interference. The 2016 update adds that replay should be *weighted/prioritized*, and schema-consistent material can consolidate rapidly.

### 1.4 Thalamus, insula, hypothalamus - the missing dynamical substrates

- **Thalamus** as relay/gate: nearly all sensory input passes through it; tonic vs burst firing gates information, and thalamocortical loops (with the inhibitory thalamic reticular nucleus, TRN) are a leading substrate for arousal and conscious access. Closed-loop model: cortical `x_c` and thalamic `x_t` with reciprocal weights, TRN lateral inhibition, gating `g ∈ {tonic, burst}`.
- **Insula / interoceptive inference** [Seth 2013; Barrett & Simmons 2015](https://www.nature.com/articles/nrn3950): emotions as predictions about bodily state that minimize interoceptive prediction error. `ε = s_body − ĝ(μ)`; update `μ̇ = −∂F/∂μ`; precision `π` weights `ε`. **Feeling ≈ precision-weighted interoceptive error.**
- **Hypothalamus / homeostatic RL** [Keramati & Gutkin 2014](https://elifesciences.org/articles/04811): tracks setpoints `H*`, generates drives (see §4.3).

### 1.5 Amygdala - discrete defensive modes

[Fadok et al. 2018](https://www.nature.com/articles/nrn.2018.22): lateral amygdala learns CS–US threat associations; the central nucleus (CeA) uses mutually-inhibitory winner-take-all microcircuits to select among **discrete** defensive states (freeze/flight/fight):

```
WTA dynamics:        ẋ_k = −x_k + f( I_k − Σ_{j≠k} w·x_j )
CS-US association:   ΔV = αβ(λ − ΣV)            (Rescorla–Wagner)
```

This is precisely the discrete-emotion machinery a flat valence scalar cannot express.

### 1.6 Large-scale networks - the triple-network model

[Menon & Uddin 2010](https://doi.org/10.1007/s00429-010-0262-0): a **Salience Network** (anterior insula + dACC, with fast von Economo neurons) detects what matters and **switches** between the **Default Mode Network** (self-referential, autobiographical, mind-wandering) and the **Central Executive / frontoparietal Network** (goal-directed control). Modelled as a dynamic-causal/switching system where the SN is the causal driver toggling DMN↔CEN.

### 1.7 Cerebellum & PFC–BG working-memory gating

- **Cerebellum** forward/internal models (Marr–Albus–Ito; Kawato): supervised learning via climbing-fiber error at parallel-fiber→Purkinje synapses, `Δw_pf→Pkj = −η·(cf_error)·pf_activity`.
- **PBWM** [O'Reilly & Frank 2006](https://pubmed.ncbi.nlm.nih.gov/16378516/): PFC working-memory maintained behind a basal-ganglia/thalamus striatal Go/NoGo gate, trained by dopamine `δ`; thalamic disinhibition opens the gate.

### 1.8 brain-llm mapping - anatomy

**What we have**
- A documentation-level mapping (`docs/memory-keeper.md`) of episodic (JSONL) ↔ hippocampus, semantic+graph ↔ neocortex, playbooks ↔ basal-ganglia/cerebellum, scratchpad ↔ prefrontal, affect/state ↔ amygdala+nuclei.
- A **correct CLS consolidation** in `consolidation_plan` (episodic→semantic promotion of strong traces). This is the one region-pair implemented faithfully.
- The **declarative branch** (episodic + semantic) covered; a gesture at non-declarative via procedural playbooks.

**What we lack**
- **No DG pattern separation** → near-duplicate episodes interfere instead of orthogonalizing.
- **No CA3 attractor completion** → cannot reconstruct a full episode from a partial cue; `retrieval_score` is a linear blend, not energy-descent.
- **No entorhinal gateway**, no grid/place-style relational/contextual coordinate.
- **No thalamus** → no relay/gating bottleneck, no thalamocortical recurrence, no TRN attentional selection (the missing "consciousness gateway").
- **No insula/interoception** → "feelings" are hand-scored valence numbers with no embodied prediction-error grounding.
- **No hypothalamus/homeostatic drives** → reward supplied externally rather than derived from drive reduction.
- **Amygdala reduced to a scalar** (`arousal_gain` in `salience`) → no threat learning, no discrete-mode WTA selection.
- **No large-scale networks** (DMN/Salience/CEN).
- **Cerebellum/PFC-BG gating absent**: playbooks are LLM-distilled text, not error-driven learning; working memory is a flat ~7-item list with no Go/NoGo gating.
- **Non-declarative branch mostly missing**: no priming, conditioning, perceptual learning, or recollection-vs-familiarity dual process.

**What to add** (detailed in §6): a DG/CA3 separation+completion layer; an interoceptive/homeostatic core; a thalamus/salience gating bottleneck feeding a global-workspace broadcast - the last being the architectural precondition (not a sufficient condition) for any defensible *functional* consciousness-indicator claim.

> **Consciousness framing.** Any indicator claim should be scored against the theory-derived **indicator properties** in [Butlin, Long, Bengio et al. 2023](https://arxiv.org/abs/2308.08708) (global workspace, recurrence, higher-order/metacognition, attention schema, agency/embodiment, predictive processing). This is an **access/architecture** rubric, scored 0..1 - never a claim of felt qualia, which is explicitly out of scope.

---

## 2. Computational neuromodulation

The keystone view [Doya 2002](https://doi.org/10.1016/S0893-6080(02)00044-8) is that the four ascending neuromodulators are not generic "gains" but the **meta-parameters of an RL agent**:

| Neuromodulator | RL meta-parameter | Role |
|---|---|---|
| Dopamine (DA) | error `δ` | the learning signal (RPE) |
| Serotonin (5-HT) | discount `γ` | time horizon / patience |
| Acetylcholine (ACh) | learning rate `α` | how fast to update beliefs |
| Noradrenaline (NE) | inverse temperature `β` | exploration vs exploitation |

RL core that they parameterize:

```
Q(s,a) ← Q(s,a) + α·δ            δ = TD error,   α from ACh
discount by γ ∈ [0,1]            γ from serotonin
P(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))     β from NE
```

This is the single highest-leverage upgrade: `brain-llm` already *has* `ne/da/ach/cortisol`; re-interpreting them as `α,β,γ,δ` makes the chemicals *causally drive* a learning/decision loop instead of only scaling salience.

### 2.1 Dopamine = TD reward-prediction error

The most secure result in the field [Montague, Dayan & Sejnowski 1996](https://www.jneurosci.org/content/16/5/1936); [Schultz, Dayan & Montague 1997](https://www.science.org/doi/10.1126/science.275.5306.1593); review [Glimcher 2011](https://doi.org/10.1073/pnas.1014269108); modern high-res confirmation [Amo et al. 2022](https://www.nature.com/articles/s41593-022-01109-2):

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
V(s_t) ← V(s_t) + α·δ_t
TD(λ):  e_t = γλ·e_{t-1} + ∇V(s_t),   V ← V + α·δ_t·e_t
actor:  θ ← θ + α·δ_t·∇log π(a|s)
```

Dopamine fires to unpredicted reward, is silent for predicted reward, dips for omitted reward, and the burst migrates from reward-time to cue-time over learning - exactly TD. `brain-llm`'s `da = clamp(reward)` is the raw immediate reward, so it can never be surprised, disappointed, or relieved.

### 2.2 Serotonin–dopamine opponency & average-reward TD

[Daw, Kakade & Dayan 2002](http://www.princeton.edu/~ndaw/dkd02.pdf): tonic serotonin reports the long-run **average reward rate** `ρ`, which is the subtractive term in average-reward (R-learning) TD; tonic dopamine reports average punishment; phasic 5-HT may carry a punishment-prediction error.

```
δ_t = (r_t − ρ) + V(s_{t+1}) − V(s_t)
ρ ← ρ + κ·δ_t            (κ ≪ α, slow)
high 5-HT  ⇒  high γ (waits for delayed reward), lower impulsivity
```

This grounds harm aversion, patience, and **mood as the slow average of reward**. A low-mood agent becomes both pessimistic (lower `V`) and impatient (lower `γ`) - matching depression phenomenology.

### 2.3 Noradrenaline - adaptive gain & neural gain (Yerkes–Dodson)

Two complementary stories:

1. **Adaptive gain** [Aston-Jones & Cohen 2005](https://www.annualreviews.org/doi/10.1146/annurev.neuro.28.061604.135709): phasic LC-NE = exploitation (sharpened gain); high tonic LC = disengagement/exploration. NE sets the exploration temperature (lower `β`).
2. **Neural gain** [Servan-Schreiber, Printz & Cohen 1990](https://www.science.org/doi/10.1126/science.2392679): NE multiplies the slope `g` of neuronal sigmoid transfer functions, raising signal-to-noise:

```
f(x) = 1/(1 + e^{−g·x})       g ↑ with phasic NE
perf(arousal) ∝ inverted-U    (Yerkes–Dodson: peak at moderate g)
```

Because performance is best at intermediate gain and worse at both extremes, this directly yields the inverted-U - the basis for "panic/terror degrades performance" dynamics. `brain-llm`'s `ne = arousal` only multiplies salience: no sharpening, no exploration knob, no inverted-U.

### 2.4 Acetylcholine = expected uncertainty; NE = unexpected uncertainty

[Yu & Dayan 2005](https://www.cell.com/neuron/fulltext/S0896-6273(05)00362-4): ACh tracks known, within-context unreliability (**expected uncertainty**); NE tracks surprising context switches / model breaks (**unexpected uncertainty**). Together they set how much to trust priors vs evidence - and the effective learning rate. With volatility-driven learning [Behrens et al. 2007](https://www.nature.com/articles/nn1954):

```
Pearce–Hall:  α_t = ζ·|δ_{t-1}|              (associability ∝ recent surprise)
Kalman:       α_t = σ²_prior / (σ²_prior + σ²_obs)
```

High ACh and NE spikes both raise `α`. `brain-llm`'s ACh is a binary 1.0-wake / 0.1-sleep flag with no uncertainty content.

### 2.5 Oxytocin - prior/precision on social value, gate on social RPE

Oxytocin up-weights others' outcomes in the value function and *discounts* contradictory social prediction errors (making early trust/distrust sticky):

```
r' = r_self + ω·r_other                       (ω ↑ with oxytocin)
V_social ← V_social + α·(1 − κ_OT)·δ_social   (κ_OT raises with oxytocin)
```

`brain-llm` has no social/affiliative axis at all - relevant for a "relationship with the user" goal.

### 2.6 HPA axis / cortisol dynamics (allostasis)

The slow stress loop: stress→CRH→ACTH→cortisol, with cortisol negatively feeding back via Hill inhibition. Minimal 3-ODE model [Vinther, Andersen & Ottesen 2011](https://doi.org/10.1007/s00285-010-0384-2); ultradian pulsatility [Walker, Terry & Lightman 2010](https://royalsocietypublishing.org/doi/10.1098/rspb.2009.2148):

```
dCRH/dt  = f_stress · 1/(1+(C/Kf)^n)  − w1·CRH
dACTH/dt = b·CRH · 1/(1+(C/K2)^m)     − w2·ACTH
dC/dt    = a·ACTH                     − w3·C
```

Produces circadian + ultradian rhythms; under chronic load the set-point `Kf` drifts (allostatic load → burnout). Acute moderate cortisol aids consolidation; chronic high impairs retrieval (an inverted-U). `brain-llm`'s `cortisol = clamp(stress)` is instantaneous with no feedback, recovery, or allostatic memory.

### 2.7 brain-llm mapping - neuromodulation

**What we have:** the right four chemicals (`ne/da/ach/cortisol`), each computed once from the current event; McGaugh arousal-gain on salience (a defensible NE/amygdala consolidation effect).

**What we lack:** no RPE (`da` = raw reward, no `V`, no `γ`); no learning loop (no chemical updates any value/weight/policy); no serotonin / average-reward `ρ`; no exploration `β` or phasic/tonic split; no neural-gain / inverted-U; ACh carries no uncertainty; cortisol has no dynamics; no oxytocin; no opponency or interaction between modulators.

**What to add:** the Doya `α/β/γ/δ` mapping (§6.2); a TD-error dopamine core (§6.1); serotonin as `ρ`+`γ` (§6.3); NE neural-gain + inverted-U; an HPA ODE; an oxytocin-like trust variable. See §6.

---

## 3. Free-Energy Principle & Active Inference

### 3.1 Variational free energy

The FEP [Friston 2010](https://www.nature.com/articles/nrn2787) holds that perception, learning, and action all minimize a single quantity - variational free energy `F`, an upper bound on surprisal (negative log model-evidence):

```
F = D_KL[ Q(s) || P(s,o) ] = E_Q[ ln Q(s) − ln P(s,o) ]
  = D_KL[ Q(s)||P(s) ] − E_Q[ ln P(o|s) ]
  = complexity − accuracy   (= energy − entropy)
surprisal = −ln P(o);   F ≥ −ln P(o)
```

Minimizing `F` = maximizing model evidence ("self-evidencing"). The object all of this is computed against is the **generative model** `P(o,s)=P(o|s)P(s)`, in discrete form a tuple `A` (likelihood `P(o|s)`), `B` (transitions), `C` (`ln P(o)` preferences), `D` (initial prior), learned by Dirichlet count updates.

### 3.2 Expected free energy & action

[Friston, FitzGerald, Rigoli, Schwartenbeck & Pezzulo 2017](https://activeinference.github.io/papers/process_theory.pdf); epistemic value [Friston et al. 2015](https://pubmed.ncbi.nlm.nih.gov/25689102/). Action minimizes **expected free energy** `G(π)`:

```
G(π,τ) = −E_Q[ D_KL[ Q(s_τ|o_τ,π) || Q(s_τ|π) ] ]   (epistemic: −info gain = curiosity)
         − E_Q[ ln P(o_τ) ]                            (pragmatic: preference = reward)
equivalently:  G = D_KL[ Q(o_τ|π)||P(o_τ) ] + E_Q[ H[P(o_τ|s_τ)] ]   (risk + ambiguity)
policy prior:  π = σ(−γ·G)                              (softmax, γ = precision = 1/β)
precision update:  β̇ = γ²·ε_γ,   ε_γ = (β − β_prior) + (π − π_0)·G
```

This subsumes exploration, exploitation, reward-seeking, curiosity, and Bayesian surprise in one decomposition. In Friston's process theory **dopamine ≈ policy precision `γ`**, sensory precision ≈ attention/ACh - i.e. neuromodulators are precisions (this unifies §2 and §3).

### 3.3 Affective inference I - Joffily–Coricelli: `valence = −dF/dt`

[Joffily & Coricelli 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003094) define emotional valence as the **negative rate of change of free energy**, and read **discrete emotions** off the velocity + acceleration:

```
valence(t) = −dF/dt
emotion from the quadrant of ( dF/dt , d²F/dt² ):
  happiness     ( dF/dt < 0 , d²F/dt² > 0 )
  hope          ( dF/dt < 0 , d²F/dt² < 0 )    getting better, faster
  fear          ( dF/dt > 0 , d²F/dt² > 0 )    getting worse, faster
  unhappiness   ( dF/dt > 0 , d²F/dt² < 0 )
  relief        sign flip  dF/dt: + → −
  disappointment sign flip dF/dt: − → +
```

Plus a meta-learning rule coupling valence to learning rate (posterior variance):

```
σ²_emotion = σ²·exp(α·valence + β·mood)     (good news ⇒ trust model ⇒ learn slowly)
```

This is the **single most implementable affective upgrade**: it yields the discrete feelings (fear, hope, relief, happiness, disappointment) from the dynamics of *one* scalar, and `brain-llm`'s mood leaky-integrator is already the slow "mood" prior `β`. (Caveat: the quadrant→emotion labels are a theoretical proposal validated in toy simulations, not measured brain facts.)

### 3.4 Affective inference II - Hesp et al.: valence = dynamics of expected precision

[Hesp, Smith, Parr, Allen, Friston & Ramstead 2021](https://direct.mit.edu/neco/article/33/2/398/95642) ground valence one level deeper - in the dynamics of **expected precision** over the action model ("subjective fitness," an internal estimate of how well I am doing). The update term is **affective charge**:

```
AC = (π̄ − π) · G_π        (policy-belief update · expected free energy)
```

`AC` is exactly the precision-prediction-error term that updates `γ` (= dopamine) in §3.2 - so affect and neuromodulation share one equation. `AC` lends a **sign** to otherwise-unsigned divergences: rising confidence/fitness = positive affect, falling = negative. This maps cleanly onto `brain-llm`'s existing **control/dominance** axis (control == confidence in one's action model).

Related: discrete-state emotion *inference and concept learning* [Smith, Parr & Friston 2019](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.02844/full) - a worked template where an agent infers which emotion it is in and learns emotion concepts via Dirichlet updates on `A` and `D`. Interoceptive/allostatic version [Seth & Friston 2016](https://royalsocietypublishing.org/doi/10.1098/rstb.2016.0007): the same FEP machinery on the interoceptive channel, set-points as prior preferences `C`, allostasis as anticipatory model-based regulation.

### 3.5 brain-llm mapping - FEP / active inference

**What we have (in ad-hoc form):** the OCC `novelty` axis *is* Bayesian surprise / epistemic value; `ne/da/ach/cortisol` *are* precisions in disguise; `salience` is precision-weighted prediction error (McGaugh); the mood leaky-integrator *is* the slow mood prior `β`; the `control` axis is a hook for Hesp affective charge.

**What we lack:** the **generative model** and the **prediction-error loop**. `brain-llm` appraises events but never *predicts* them, so it cannot compute surprise, `F`, or `dF/dt` - meaning its "novelty" must be supplied by the caller. Consequently: no `F`, no `dF/dt` → no derived valence/discrete emotions; no expected free energy → no principled curiosity; no precision formalism with update rules; no interoception/allostasis; no valence→precision→learning-rate coupling; no active-inference action loop.

**What to add:** a minimal discrete generative model + running `F` (so novelty is *computed*, §6.4); valence + discrete emotions from `(dF/dt, d²F/dt²)` (§6.4); affective charge `AC` tied to the control axis; an epistemic-value curiosity signal; a minimal interoceptive/allostasis layer.

> **FEP honesty caveat.** `valence = −dF/dt` and `AC = (π̄−π)·G` are computational correlates of how affect modulates cognition; they are not claims about felt experience. The FEP is also philosophically contested (the "dark room problem," the reality of preferences-as-priors, falsifiability). Present it as a powerful *organizing formalism*, not settled science.

---

## 4. Reinforcement Learning + Emotion + Homeostasis

### 4.1 Emotion as a function of RL signals

The standard taxonomy [Moerland, Broekens & Jonker 2018](https://doi.org/10.1007/s10994-017-5666-0) maps emotions to RL quantities; concrete equations in [Broekens, Jacobs & Jonker 2015](https://doi.org/10.1080/09540091.2015.1031081):

```
joy / distress  ∝  δ_t           (signed RPE: + → joy, − → distress)
surprise        ∝  |δ_t|  or |s_actual − ŝ_predicted|
hope            ∝  max(0,  V_anticipated  − V_now)
fear            ∝  max(0,  V_now − V_anticipated_worst)
```

This derives the discrete "feelings" the project wants as *computed* quantities composing with the existing OCC/PAD front-end - and it is the same `δ` and `V` machinery as §2.1.

### 4.2 Mood as momentum

[Eldar, Rutledge, Dolan & Niv 2016](https://doi.org/10.1016/j.tics.2015.07.010): mood is an exponentially-weighted average of recent RPEs that **biases perceived reward**, producing self-amplifying streaks:

```
m_{t+1} = m_t + η'·( (r_t − V(s_t)) − m_t )      (mood = EW-average of RPEs)
V_{t+1} = V_t + η·( f·m_t + r_t − V_t )           (mood biases the value update)
happiness_t = w0 + Σ_j γ^{t−j} ( w1·CR_j + w2·EV_j + w3·RPE_j )
```

Strictly more powerful than `brain-llm`'s `update_mood`, which integrates *appraised affect* (not RPEs) and never feeds back onto valuation.

### 4.3 Homeostatic RL - reward = drive reduction

[Keramati & Gutkin 2011](https://proceedings.neurips.cc/paper/2011/file/9778d5d219c5080b9a6a17bef029331c-Paper.pdf), [2014](https://doi.org/10.7554/eLife.04811) prove reward maximization ≡ physiological-deviation minimization when reward is **drive reduction**:

```
D(H_t) = ( Σ_i |H*_i − H_{t,i}|^n )^(1/m),   n > m > 1
r_t = D(H_t) − D(H_{t+1})                    (drive reduction from outcome K_t)
H_{t+1} = H_t + K_t                           (toward setpoint vector H*)
```

This naturally generates motivation, satiety, and context-dependent value (the same outcome is more rewarding when the corresponding need is high). Extensions: continuous-time/space CTCS-HRRL ([Laurençon et al. 2024](https://arxiv.org/abs/2401.08999)); modular competing drives ([Dulberg et al. 2022](https://arxiv.org/pdf/2204.06608)).

### 4.4 Intrinsic motivation / curiosity

`brain-llm`'s `novelty` only feeds arousal/salience; it never *drives* behaviour. The literature offers principled drives:

```
compression progress (Schmidhuber):  I(t) = C_{t-1} − C_t   (learning progress; noise-robust)
pseudo-count bonus (Bellemare 2016):  r⁺ = β / √N̂(s)
RND (Burda 2018):                      r_i = ‖ f̂_θ(s) − f(s) ‖²   (f fixed random net)
empowerment (Klyubin 2005):            E = max_{p(a)} I(A; S')    (channel capacity action→future state)
```

[Schmidhuber 2010](https://doi.org/10.1109/TAMD.2010.2056368) - reward the *first derivative of compressibility*, not raw novelty (avoids the "noisy-TV" trap); [Bellemare et al. 2016](https://papers.nips.cc/paper/6383-unifying-count-based-exploration-and-intrinsic-motivation); [Burda et al. 2018](https://arxiv.org/abs/1810.12894); [Klyubin, Polani & Nehaniv 2005](https://doi.org/10.1007/11553090_75). Empowerment is the computable analog of `brain-llm`'s hand-appraised `control`/dominance axis.

### 4.5 Affective biasing of explore/exploit

DA/5-HT opponency [Daw, Kakade & Dayan 2002](https://doi.org/10.1016/S0893-6080(02)00052-7) (also §2.2) plus mood/optimism set the softmax temperature:

```
avg-reward TD:  δ_t = r_t − ρ + V(s_{t+1}) − V(s_t)
softmax:        π(a) = exp(Q(a)/τ) / Σ exp(Q/τ),   τ = g(arousal, mood, 5-HT-analog)
```

`brain-llm` has no action-selection layer, but it can *advise* the host agent of a temperature `τ` as a function of mood/arousal/serotonin.

### 4.6 brain-llm mapping - RL + emotion + homeostasis

**What we have:** an OCC→PAD continuous affect pipeline; a mood leaky-integrator; `da` *intended* as dopamine; a `novelty` axis described as a Bayesian-surprise proxy.

**What we lack:** no value function / RPE (`da = clamp(reward)`, `reward = max(valence,0)`); no discrete emotions derived from RL signals; no interoception/drives/setpoints (only a single mood baseline; `goal_relevance` hand-appraised, not computed as drive reduction); mood does not feed back into valuation or track RPEs; no intrinsic-motivation drive; no serotonin/average-reward; no explore/exploit control.

**What to add:** a value/RPE core (§6.1); RPE-derived discrete emotions (§6.5); homeostatic drives with drive-reduction reward (§6.6); mood-as-momentum (§6.7); an intrinsic-reward term (learning-progress + pseudo-count); an advisory exploration temperature + serotonin-analog.

---

## 5. Advanced sleep, consolidation, replay & forgetting

`brain-llm` already encodes the *spirit* of CLS in `consolidation_plan` (`strength = salience · sigmoid(activation)`, REM arousal boost, promote/forget thresholds) and `retention` (importance-modulated decay). The modern science adds three layers it flattens into scalars.

### 5.1 SHY - synaptic homeostasis / down-selection

[Tononi & Cirelli 2020](https://doi.org/10.1111/ejn.14335): waking nets a global increase in synaptic strength; NREM performs a **proportional renormalization** that preserves relative ranking while improving SNR and freeing capacity.

```
multiplicative renormalization:  w_i ← w_i · s,   s = W_target / Σ_i w_i  (s < 1)
selective (down-SELECTION):      w_i ← max(0, w_i − c·(1 − r_i))
                                 r_i ∈ [0,1] = fraction of replay/reactivation received
```

`brain-llm` has *no* homeostatic renormalization - forgetting is purely per-item, so total "memory mass" is unbounded.

### 5.2 Active systems consolidation - SO–spindle–ripple nesting

NREM transfer is timed by cortical slow oscillations (~0.5–1 Hz) gating thalamocortical spindles (~12–15 Hz) gating hippocampal sharp-wave ripples (~80–200 Hz). Operationally a gating schedule: `transfer_rate(t) ∝ 1[up-state] · spindle_envelope(t) · ripple_event(t)`; REM preferentially restabilizes emotional/procedural traces. `brain-llm` captures only the coarse wake/NREM ACh switch and a flat REM arousal bonus.

### 5.3 Standard vs Multiple-Trace consolidation

- **Standard Consolidation Theory:** the trace migrates hippocampus→neocortex and becomes hippocampus-independent.
- **Multiple-Trace / Trace-Transformation Theory:** the hippocampus *retains* a detailed episodic trace for vivid recall while a generalized **gist** is independently laid down in cortex. Recent 7T-fMRI evidence [Steel et al. 2022](https://doi.org/10.1073/pnas.2123426119) favours MTT for remote episodic detail.

```
two-trace dynamics:
  episodic:  E(t) = E0·exp(−λ_E·t)     (λ_E reduced by reactivation)
  semantic:  G(t) = G0·(1 − exp(−κt))  (gist accumulates from replay of E)
  recall = max( act_E ,  act_G·gist_match )
  SCT = special case: delete E once G > threshold
```

`brain-llm`'s promote step implicitly follows SCT (moves a gist and can drop the episode) - it cannot represent "I both remember the specific incident AND learned the general lesson." (The debate is genuinely unresolved; expose this as a configurable mode, not a fact.)

### 5.4 Prioritized replay - EVB and PER

Biological replay is **prioritized and reconstructive**, not a binary promote. Waking SWRs *tag* a subset of experiences and sleep SWRs preferentially replay the tagged ones [Yang et al. 2024](https://doi.org/10.1126/science.adk8261). The normative rule [Mattar & Daw 2018](https://doi.org/10.1038/s41593-018-0232-z):

```
EVB(s,a) = Gain(s,a) · Need(s)
Gain(s,a) = Σ_a [ π_new(a|s) − π_old(a|s) ] · Q(s,a)   (policy improvement; large for surprising/inconsistent)
Need(s)   = Σ_t γ^t P(S_t = s | S_0 = current)          (successor-representation occupancy = row of (I−γT)^{-1})
```

The RL analog, **Prioritized Experience Replay** [Schaul et al. 2016](https://arxiv.org/abs/1511.05952):

```
priority   p_i = |δ_i| + ε
sampling   P(i) = p_i^α / Σ_k p_k^α          (α ≈ 0.6)
IS weight  w_i = ( 1 / (N·P(i)) )^β           (β annealed 0.4 → 1)
```

`brain-llm` replays *nothing*: `consolidation_plan` is a single relabel pass with no budget, ordering, or reconstruction.

### 5.5 Catastrophic forgetting & continual learning

The neocortex-analog (semantic store / any fine-tuned model) needs protection. The field [Wang et al. 2024](https://doi.org/10.1109/TPAMI.2024.3367329) is a stability-plasticity trade-off across regularization / replay / architecture / optimization:

```
EWC (Kirkpatrick 2017):  L(θ) = L_B(θ) + Σ_i (λ/2)·F_i·(θ_i − θ*_i)²    F_i = diag Fisher
SI  (Zenke 2017):        ω_k = −∫ (∂L/∂θ_k)(dθ_k/dt) dt;  Ω_k = Σ_μ ω_k^μ / ((Δθ_k^μ)² + ξ)
```

[Kirkpatrick et al. 2017](https://doi.org/10.1073/pnas.1611835114); [Zenke, Poole & Ganguli 2017](https://arxiv.org/abs/1703.04200). The biologically faithful family is **generative replay**: Deep Generative Replay [Shin et al. 2017](https://arxiv.org/abs/1705.08690) and **Brain-Inspired Replay** [van de Ven, Siegelmann & Tolias 2020](https://doi.org/10.1038/s41467-020-17866-2), which replays *internal/hidden* representations via the network's own feedback (a VAE head, conditional Gaussian-mixture latents, distillation), nearly eliminating forgetting without storing raw data:

```
generator prior:  p(z) = Σ_c p(c)·N(μ_c, σ_c²I)
combined loss:    L = (1/t)·L_current + (1 − 1/t)·L_replay
distillation:     L_D = −T²·Σ_c ỹ_c·log p^T(c|x),   T = 2
```

Sleep-phase studies [Tadros et al. 2022](https://doi.org/10.1038/s41467-022-34938-7) and [Robinson et al. 2022](https://arxiv.org/abs/2209.05245) show that combining **NREM veridical replay + REM generative replay + synaptic downscaling** beats any single mechanism, with downscaling strength as a direct stability-plasticity knob.

### 5.6 Active forgetting

[Davis & Zhong 2017](https://doi.org/10.1016/j.neuron.2017.05.039): forgetting is a constitutive, dopamine-gated process (DA→Rac1→cofilin actin remodeling) competing with consolidation, accelerated by interference.

```
state-dependent decay:  λ_i = λ_base·exp(−μ·I_i)·(1 + ρ·DA)·(1 + σ·interference_i)
active erosion:          w_i ← w_i − κ·DA·overlap(i, newly_consolidated)
```

`brain-llm`'s `retention` has importance-modulated `λ` but no DA gating and no interference term.

### 5.7 brain-llm mapping - sleep/consolidation/forgetting

**What we have:** the right ontology - CLS dual store, a sleep mode (ACh wake/NREM + REM boost), importance-modulated decay (`retention`), and a working promote/forget consolidation pass.

**What we lack:** no SHY downscaling (unbounded memory mass); no actual replay (no buffer/budget/ordering); no replay-priority math (no EVB, no PER); conflates consolidate with move/delete (assumes SCT, no parallel episodic trace); no EWC/SI protection of the semantic store; no multi-cycle NREM/REM scheduling; no active/interference forgetting; no generative/reconstructive replay.

**What to add:** a per-cycle homeostatic downscale; a budgeted prioritized-replay queue scored by EVB; a Multiple-Trace split; an EWC/SI-style importance lock; active interference-based forgetting; a multi-cycle `run_sleep_cycle` orchestrator. See §6.

---

## 6. Concrete proposals for brain-llm (with equations)

All proposals are scalar/linear-algebra operations consistent with `brain.py`'s pure-stdlib style. Each carries an honesty caveat: these reproduce the *function* of the named mechanism, not phenomenal experience, and several rest on hand-defined representations (the "state," the "context," the "body") that are engineered proxies, not biology.

### 6.1 A TD reward-prediction-error dopamine core *(difficulty: medium)*

Replace `da = clamp(reward)` with a learned error so encoding is driven by **surprise**.

```python
# src/value_rl.py
V = {}                                  # cue -> value, default 0
def rpe(key, r, next_key, alpha=0.3, gamma=0.9):
    delta = r + gamma*V.get(next_key,0.0) - V.get(key,0.0)
    V[key] = V.get(key,0.0) + alpha*delta
    return delta
# in neuromods_from:  da = clamp(0.5 + 0.5*math.tanh(delta))   # phasic DA ∝ RPE
# in salience:        arousal_gain *= (1 + 0.5*abs(delta))     # surprise boosts encoding
```

*Caveat:* the cue/state featurization for an LLM agent is a modeling choice; `δ` is only as meaningful as the chosen features. Reward must be defined operationally (test pass, user approval). Persist `V` alongside the semantic store.

### 6.2 Doya neuromodulator↔meta-parameter mapping *(medium)*

Make the four chemicals causally set the RL knobs.

```python
alpha = clamp(alpha0 * (0.5 + ach))           # ACh -> learning rate
beta  = beta0 * (1.0 - 0.7*ne_tonic)          # high tonic NE -> explore (low beta)
gamma = clamp(0.5 + 0.4*serotonin, 0, 0.99)   # serotonin -> patience/horizon
P_i   = exp(beta*score_i) / Σ_j exp(beta*score_j)   # softmax retrieval/action ranking
```

Replace the linear `retrieval_score` ranking with a softmax at temperature `1/β`. *Caveat:* the 1-to-1 mapping is Doya's proposal with partial empirical support; gains are heuristic.

### 6.3 Serotonin = average reward `ρ` + mood set-point *(medium)*

```python
rho += kappa*delta                       # slow average reward, kappa << alpha
delta = (r - rho) + V[next] - V[cur]     # average-reward TD error
serotonin = sigmoid(c1*rho)
baseline_mood.valence = math.tanh(rho)   # mood set-point follows avg reward (replaces fixed 0.0)
rho_pun += kappa*max(-delta, 0.0)        # punishment opponent (tonic DA analog)
```

*Caveat:* serotonin-as-average-reward is the Daw–Kakade–Dayan hypothesis; serotonin's real functions are heterogeneous. A useful functional abstraction, not settled biology.

### 6.4 A free-energy / surprise backbone + discrete emotions *(generative model: medium; emotions: low)*

The recommended unifying upgrade. Add a minimal discrete generative model so the engine **computes** novelty/surprise, then derive valence and discrete emotions from `F`'s dynamics.

```python
# src/generative.py - categorical belief Q(s) over a small set of "situation" states,
# Dirichlet-counted likelihood A = P(event_category | s), prior D.
P_o   = sum(A[o][s]*Q[s] for s in states)         # predicted prob of observed category
surprise = -math.log(max(P_o, 1e-9))
Q     = normalize([A[o][s]*Q[s] for s in states]) # posterior belief update
F     = sum(Q[s]*(math.log(Q[s]+1e-9) - math.log(A[o][s]+1e-9) - math.log(D[s]+1e-9))
            for s in states)                       # complexity - accuracy
appraisal.novelty = 1 - math.exp(-surprise)        # computed, not caller-supplied
A[o][s] += Q[s]                                    # Dirichlet learning
```

Discrete emotions (Joffily–Coricelli, §3.3) from a running history of `F`:

```python
dF  = F_t - F_tm1
d2F = (F_t - F_tm1) - (F_tm1 - F_tm2)
valence = -dF                                      # optionally EMA-smoothed
emotion = ( "happiness" if dF<0 and d2F>0 else
            "hope"      if dF<0 and d2F<0 else
            "fear"      if dF>0 and d2F>0 else
            "unhappiness" )                         # + sign-flip -> relief / disappointment
sigma2 = sigma2_0 * math.exp(a_v*valence + b_m*mood.valence)   # valence -> learning rate
```

*Caveat:* the state space is hand-defined (a coarse toy generative model); "surprise" and the emotion labels name *functional regimes*, not felt states. Needs ≥3 time points, so it only works on a running event stream.

### 6.5 RPE-derived discrete emotions (RL route) *(medium)*

A complementary, RL-grounded route to the same discrete feelings (§4.1), composing with §6.1:

```python
joy      = max(0,  delta)
distress = max(0, -delta)
surprise = clamp(abs(delta) / s_norm)
hope     = max(0, V_anticipated - V_now)
fear     = max(0, V_now - V_worstcase_anticipated)
```

Store the dominant emotion label on each episode for emotion-congruent retrieval. *Caveat:* hope/fear need a forward model `brain-llm` lacks; initially coarse (one-step lookahead).

### 6.6 Homeostatic drives - endogenous motivation *(high)*

Give the agent genuine needs so `goal_relevance` and reward become *computed* (§4.3, Keramati–Gutkin).

```python
# src/homeostasis.py - H: internal vars (cognitive_load, task_debt, context_budget,
# competence, curiosity_satiety) with setpoints H*.
def drive(H, Hstar, n=2.0, m=2.0):
    return (sum(abs(Hstar[i]-H[i])**n for i in H) ) ** (1.0/m)
r_drive = drive(H_prev, Hstar) - drive(H_now, Hstar)   # reward = drive reduction
# interoceptive feeling (insula, §1.4):
eps = pi * (H_now[i] - H_pred[i])      # precision-weighted prediction error
arousal = clamp(abs(eps)); valence = -math.copysign(1, deviation)
```

Feed `r_drive` as the `r` in §6.1, and the dominant unmet drive into `appraise_to_affect`. *Caveat:* the physiological variables are metaphorical (an LLM agent has no glucose); this models the *function* of homeostatic motivation and must be labeled as such.

### 6.7 Mood as momentum + HPA dynamics *(mood: low; HPA: high)*

Upgrade `update_mood` from an affect integrator to an RPE-momentum signal that biases valuation (§4.2):

```python
m += eta_m * (delta - m)              # mood = EW-average of RPEs (eta_m ~ 0.1, slow)
r_perceived = r + f*m                 # mood biases perceived reward (f ~ 0.3)
delta = r_perceived + gamma*V[next] - V[cur]
```

*Caveat:* the feedback loop can run away (mania/depression); clamp and keep `eta_m` slow. Replace instantaneous cortisol with the 3-ODE HPA loop (§2.6) for chronic-stress/burnout and a stress inverted-U on consolidation:

```python
CRH  += dt*( stress*1/(1+(C/Kf)**n) - w1*CRH )
ACTH += dt*( b*CRH - w2*ACTH )
C    += dt*( a*ACTH - w3*C )
Kf   += dt*eps*(C - C_target)         # allostatic set-point drift (modeling extension)
consol_gain = C*math.exp(-C/Copt)     # acute aids, chronic impairs (inverted-U)
```

### 6.8 Richer consolidation - downscaling, prioritized replay, MTT, importance-lock, active forgetting *(low–medium each)*

Wrap into a multi-cycle `run_sleep_cycle` orchestrator (§5).

```python
# (a) SHY homeostatic downscale (after promote/forget):
W = sum(e["strength"] for e in retained)
s = clamp(W_target / W, 0, 1)
for e in retained:
    e["strength"] = e["strength"]*s + protect*replayed[e]* (1-s)*e["strength"]

# (b) prioritized replay (EVB ≈ Gain × Need, sampled PER-style):
gain_i = wg*novelty_i + wc*contradiction_i + (1 - goal_match_i)
need_i = sigmoid(base_level_activation(retrievals_i, now))      # reuse ACT-R
p_i    = gain_i*need_i + eps
# draw top-B by P(i) = p_i^alpha / Σ p_k^alpha ; each replay: strength *= (1+eta), transfer += k*strength

# (c) Multiple-Trace split: keep decaying episodic E alongside growing semantic gist G;
#     recall = max(act_E, act_G * gist_match)   (configurable SCT vs MTT mode)

# (d) EWC/SI importance-lock on semantic facts:
f_new = f_old + delta_proposed / (1 + lam*Omega_f)              # high-importance facts move little

# (e) active interference forgetting:
lam_i = lambda_base*exp(-mu*I_i)*(1 + rho*da)*(1 + sigma*interference_i)
```

*Caveats:* `W_target`, `B`, and the allostatic drift are free parameters with no biological ground truth (tune empirically). EVB's true `Need` is successor-representation occupancy; ACT-R activation is a defensible proxy. EWC/SI were derived for differentiable parameters; for `brain-llm`'s symbolic facts, "importance"/"gradient" reinterpret as usage/support counts and contradiction magnitude - a faithful analogy, not literal Fisher information. Over-locking risks freezing genuinely outdated facts; interference erasure needs a high-salience floor.

### 6.9 Architectural precondition for a *functional* consciousness indicator *(high)*

A thalamus/salience gating bottleneck feeding a global-workspace broadcast (§1.4, §1.6):

```python
s_i = w · [novelty_i, abs(valence_i), goal_relevance_i, drive_i]   # salience switch
g   = softmax(beta * s)                  # fast WTA admits top-1 (or softmax) item
broadcast(top_item)                      # written to a shared workspace readable by all modules for one cycle
```

Score the architecture against the [Butlin et al. 2023](https://arxiv.org/abs/2308.08708) indicator properties (global broadcast? recurrence? attention schema? metacognitive higher-order monitor?) as a 0..1 checklist. **This yields access/architecture indicators only - never felt experience or sentience; the qualia question is philosophically contested and explicitly out of scope.**

---

## 7. Reference open-source projects

- **pymdp** - discrete active inference (variational + expected free energy). https://github.com/infer-actively/pymdp - best source for §6.4.
- **Emergent / Leabra** - O'Reilly CCN models incl. DG/CA3 hippocampus and PBWM gating. https://github.com/emer/leabra - for §1.2, §1.7.
- **Stable-Baselines3** - TD/actor-critic, value functions, softmax temperature. https://github.com/DLR-RM/stable-baselines3 - for §6.1–§6.3, §5.4 (PER).
- **Avalanche** - continual learning: EWC, SI, replay. https://github.com/ContinualAI/avalanche - for §6.8(d).
- **continual-learning** / **brain-inspired-replay** (van de Ven) - exact EWC/SI/DGR and generative-replay code. https://github.com/GMvandeVen/continual-learning · https://github.com/GMvandeVen/brain-inspired-replay
- **pathint** (Zenke) - original Synaptic Intelligence. https://github.com/ganguli-lab/pathint
- **PrioritizedReplay** (Mattar & Daw) - EVB = Gain × Need. https://github.com/marcelomattar/PrioritizedReplay
- **RLeXplore** - intrinsic-reward zoo (ICM, RND, pseudo-counts). https://github.com/RLE-Foundation/RLeXplore - for §6.6 curiosity.
- **pyhgf** - Hierarchical Gaussian Filter, lightweight precision-weighted PE. https://github.com/ComputationalPsychiatry/pyhgf
- **The Virtual Brain** - region/thalamocortical/large-scale dynamics (conceptual reference). https://github.com/the-virtual-brain/tvb-root
- **mammoth** - large continual-learning framework incl. DER++. https://github.com/aimagelab/mammoth

---

## 8. Recommended build order

1. **Free-energy / surprise backbone + discrete emotions (§6.4)** - highest leverage; makes affect self-computing and yields the wanted discrete feelings from one scalar's dynamics. Pairs with the existing mood integrator as the slow prior.
2. **TD-error dopamine core + Doya mapping (§6.1, §6.2)** - turns the four chemicals into a real learning/decision loop; minimal code, maximal functional gain.
3. **Homeostatic drives (§6.6)** - gives endogenous motivation; lets `goal_relevance`/reward be computed, not supplied.
4. **Richer consolidation (§6.8)** - downscaling + prioritized replay + MTT + importance-lock + active forgetting, wrapped in a multi-cycle sleep orchestrator.
5. **DG/CA3 separation+completion (§1.2)** and **thalamus/workspace gating (§6.9)** - the architectural layer; the latter is the precondition for any defensible *functional* consciousness-indicator score (Butlin et al.), never a sentience claim.

> **Closing honesty note.** Every equation here describes computation that *behaves like* memory, emotion, motivation, and attention in its effect on the system - what gets kept, transformed, learned, dropped, and prioritized. None of it licenses a claim that `brain-llm` feels anything. Keep that line explicit in code comments, docs, and any indicator rubric.

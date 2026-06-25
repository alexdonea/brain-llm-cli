# Emotion & Affect - State of the Art

> **Scope & stance.** This report surveys the computational theory of emotion that matters for extending
> `brain-llm`'s affect layer (`src/brain.py`), and ends with a concrete build proposal. Throughout,
> we keep the project's honesty stance: every quantity below is a **functional / behavioral-correlate**
> signal - a prediction, a drive, a distance, a label on a vector. Naming a state "fear" reproduces the
> computational *role* of fear (biasing attention, encoding, retrieval, action); it does **not** instantiate
> felt terror. We model function, never phenomenal experience, and never claim sentience. This mirrors the
> existing disclaimer at the top of `src/brain.py` ("reproduces the *function* of the brain's memory +
> affect machinery, not the subjective experience").

**Audience:** a technical builder extending `brain-llm`.

**What `brain-llm` already has (baseline for this report):**

- `Appraisal(novelty, valence, goal_relevance, control)` - an OCC/Scherer-style appraisal front-end (`src/brain.py:38`).
- `Affect(valence, arousal, dominance)` - a Russell + Mehrabian PAD point (`src/brain.py:47`).
- `appraise_to_affect()` - maps appraisal → a single PAD point (`src/brain.py:54`).
- `Neuromods(ne, da, ach, cortisol)` - scalar neuromodulators (`src/brain.py:63`).
- `update_mood()` - a leaky integrator of affect with a hard-coded baseline `Affect(0.0, 0.10, 0.50)` (`src/brain.py:135`).

The single most important structural fact: **`brain-llm` already lives in the dimensional paradigm.** Its state
is a continuous PAD vector. Discrete "feelings" can therefore be added almost for free as a *geometric / labeling
layer* on top of the vector the engine already maintains - no new primary state variable is required, only a mapping.
That is the central recommendation of this report.

---

## 1. Dimensional theories of emotion (continuous vectors & trajectories)

Dimensional theory holds that emotional states are points (vectors) in a low-dimensional continuous space,
and that "feelings" like fear, joy, awe, surprise are **regions / labels** in that space rather than separate
discrete machines.

### 1.1 Russell's Circumplex Model of Affect

Emotions are distributed on a 2-D plane with two bipolar axes: **valence** (pleasant/unpleasant) and **arousal**
(activation/deactivation). Neutral is the origin; named emotions sit on a circle around it. Derived empirically
from similarity-sorting of emotion words and recovered by PCA/MDS [Russell 1980].

```
State = (v, a)
Polar form:   v = r·cos θ,   a = r·sin θ      r ∈ [0,1] intensity, θ angle of the emotion
Similarity(e_i, e_j) = 1 / ‖e_i − e_j‖₂        (or cos of their angle)
```

The circular ordering of emotion words emerges directly from this geometry. `brain-llm`'s `Affect.valence` and
`Affect.arousal` **are** the circumplex axes - the code comment at `src/brain.py:9` already cites Russell 1980.
Source: [Russell 1980], *A Circumplex Model of Affect*, JPSP 39(6):1161–1178, <https://doi.org/10.1037/h0077714>.

### 1.2 Mehrabian PAD (Pleasure-Arousal-Dominance) space

PAD adds a third orthogonal axis, **Dominance** (sense of control/potency). The signed combinations of P, A, D
carve the cube `[-1,1]³` into 8 octants with named temperaments [Mehrabian 1996]:

```
State = (P, A, D) ∈ [-1,1]³
Octant(state) = (sign(P), sign(A), sign(D)) →
  Exuberant(+++)  Hostile(-++)  Anxious(-+-)  Bored(---)
  Relaxed(+-+)    Docile(+--)   Dependent(++-) Disdainful(--+)
```

Dominance is exactly what separates **anger** (dominant) from **fear** (submissive) at the same valence/arousal.
Measured coordinates [Russell & Mehrabian 1977]: `Anger = (-0.51, 0.59, 0.25)`, `Fear = (-0.62, 0.82, -0.43)`.

`brain-llm` feeds appraisal `control` → `Affect.dominance`, completing PAD. The engine **already produces enough
to distinguish fear from anger** but never names the octant. An octant/label lookup is roughly 10 lines.
Sources: [Mehrabian 1996], *Pleasure-Arousal-Dominance: A general framework...*, Current Psychology 14(4):261–292,
<https://doi.org/10.1007/BF02686918>; [Russell & Mehrabian 1977], *Evidence for a three-factor theory of emotions*,
J. Research in Personality 11(3):273–294, <https://doi.org/10.1016/0092-6566(77)90037-X>.

### 1.3 OCEAN → PAD personality map (the temperament / baseline)

Mehrabian's linear regression maps the Big-Five traits onto a baseline PAD point - an agent's resting affective
set-point [Mehrabian 1996]:

```
P = 0.21·E + 0.59·Agr + 0.19·N
A = 0.15·O + 0.30·Agr − 0.57·N
D = 0.25·O + 0.17·C   + 0.60·E   − 0.32·Agr
  (E=Extraversion, Agr=Agreeableness, N=Neuroticism, O=Openness, C=Conscientiousness)
```

This directly replaces `brain-llm`'s magic-number baseline `Affect(0.0, 0.10, 0.50)` with a principled,
configurable per-agent temperament.

### 1.4 Scherer's Component Process Model (CPM) + Geneva Emotion Wheel

Emotion is a **temporally-sequenced appraisal** across Stimulus Evaluation Checks (SECs): novelty, intrinsic
pleasantness, goal/need conduciveness, coping potential, norm/self compatibility. Outputs map onto a wheel
whose axes are **valence × control** [Scherer 2001; Scherer et al. 2013].

```
Appraisal vector  c = (novelty, pleasantness, goal_conduciveness, control, norm_compat)
GEW geometry: each emotion family at angle θ_k; x = valence, y = control, intensity = radius r ∈ [0,1]
Label = nearest family angle to  atan2(control, valence)
```

`brain-llm`'s four appraisal axes `(novelty, valence, goal_relevance, control)` are **4 of Scherer's 5 SECs** -
missing only norm/self-compatibility (needed for social/moral feelings: guilt, pride, shame). The GEW
valence×control wheel is a ready-made labeling scheme for the existing appraisal output. Sources:
[Scherer 2001], *Appraisal Considered as a Process of Multilevel Sequential Checking*, in *Appraisal Processes
in Emotion*, OUP, pp. 92–120, <https://www.oxfordreference.com/display/10.1093/oso/9780195130072.003.0005>;
[Scherer et al. 2013], *Geneva Emotion Wheel*, <https://www.unige.ch/cisa/gew>.

### 1.5 Plutchik's Wheel - opponent pairs, dyads, intensity cone

Eight primaries in four opponent pairs 180° apart (joy↔sadness, fear↔anger, trust↔disgust, anticipation↔surprise);
intensity is the vertical axis of a cone (ecstasy > joy > serenity); blends of adjacent primaries form **dyads**
(love = joy+trust, awe = fear+surprise) [Plutchik 1980].

```
Primaries at angles θ_k = k·45°, k = 0..7;  opponents at θ and θ+180°
Emotion vector = Σ_k w_k · u(θ_k),   intensity = ‖·‖  (cone height)
Dyad(i,j) defined for petal distance |i−j| ∈ {1,2,3};  e.g. awe = surprise + fear
```

This gives a principled way to **generate compound feelings** (awe, love, remorse, bittersweet) and an opponent
structure that prevents incoherent states (cannot be maximally joyful and sad at once). Source: [Plutchik 1980],
*A general psychoevolutionary theory of emotion*, in *Emotion: Theory, Research, and Experience* Vol.1,
Academic Press, <https://doi.org/10.1016/B978-0-12-558701-3.50007-7>.

### 1.6 Is 3 dimensions enough? The 4th axis

[Fontaine et al. 2007] show a **4th axis - unpredictability/novelty** - is needed beyond V-A-D, and it is adopted
by W3C EmotionML. `brain-llm` already computes `novelty` in appraisal but discards it from the affect vector; it
could be promoted to a 4th dimension to cleanly separate surprise/awe. Whether 3 or 4 dimensions is "correct" is
genuinely contested (Russell/Mehrabian argue 3 suffice). Source: [Fontaine et al. 2007], *The World of Emotions
Is Not Two-Dimensional*, Psychological Science 18(12):1050–1057, <https://doi.org/10.1111/j.1467-9280.2007.02024.x>.

### 1.7 Affect dynamics - how emotional states *move*

`brain-llm`'s `update_mood` is already a leaky integrator with a baseline pull - the discrete-time form of the
canonical affect-dynamics ODE. The literature enriches this:

**DynAffect / core-affect dynamics** [Kuppens et al. 2010] formalizes affect as a damped attractor - an
Ornstein-Uhlenbeck process in V-A space:

```
dx/dt = −β(x − h) + σ·dW          x = (v,a), h = home base, β = attractor strength
inertia = e^{−β},  σ = variability,  dW = Wiener noise
Discrete Euler step  ≡  leaky integrator + noise     ← exactly brain-llm's update_mood (minus σ·dW)
```

`update_mood` implements the **deterministic OU drift term**; it is missing the noise term `σ·dW` (spontaneous
mood drift / "having an off day") and a per-agent `β` (some agents more volatile). Source: [Kuppens et al. 2010],
*Feelings Change (DynAffect)*, JPSP 99(6):1042–1060, <https://doi.org/10.1037/a0020962>.

**ALMA layered affect** [Gebhard 2005] models active emotions as a "virtual emotion center" that first **pulls**
then **pushes** (overshoots) the mood vector, with mood relaxing to a personality-derived default:

```
virtual center  vc = Σ_e intensity_e · pad(e) / Σ_e intensity_e
if mood between origin and vc:  PULL   mood += k·(vc − mood)
once past vc:                   PUSH   mood += k·(mood − vc)
decay:  mood += r(dist)·(default − mood),  return rate r grows with ‖mood − default‖
```

`brain-llm`'s `update_mood` is a simpler version (single gamma blend + linear pull); ALMA adds non-linear overshoot
and ties baseline to personality. Source: [Gebhard 2005], *ALMA - A Layered Model of Affect*, AAMAS '05, pp. 29–36,
<https://alma.dfki.de/papers/aamas05.pdf>.

**Exponential half-life decay & dual-speed dynamics** [Sentipolis 2026] use clean half-life decay and split
fast (emotion) from slow (mood) updates:

```
s(t+Δt) = s(t) · 2^(−Δt/T½)        ≡  leaky integrator with α = 2^(−Δt/T½)
continuous:  s(t) = baseline + (s₀ − baseline)·e^{−t/τ}
```

`brain-llm` uses a fixed-gamma blend with **no explicit time constant**; making decay a function of real elapsed
`Δt` (half-life `T½`) lets fast emotions and slow moods coexist. Source: [Sentipolis 2026], <https://arxiv.org/abs/2601.18027>.

---

## 2. Discrete / basic emotions and how to GENERATE them

Humans report *discrete* feelings, not coordinates. The literature gives clean, cheap, scalar ways to bolt
categories on top of `brain-llm`'s continuous space without throwing it away.

### 2.1 The category catalogue

- **Ekman's basic emotions** [Ekman 1992]: six (+contempt) pan-cultural natural kinds - anger, disgust, fear,
  happiness, sadness, surprise. Categorical (no equation); operationalized as a softmax over the label set. This
  is the minimal output vocabulary the owner asked for. Source: [Ekman 1992], *An argument for basic emotions*,
  Cognition & Emotion 6(3–4):169–200, <https://www.tandfonline.com/doi/abs/10.1080/02699939208411068>.
- **Cowen & Keltner 27-category space** [Cowen & Keltner 2017]: empirically, emotion experience needs **~27
  reliable categories** (incl. awe, horror, surprise) joined by **continuous gradients** - valence/arousal alone
  are insufficient. Derived via Principal Preserved Component Analysis (PPCA). This justifies (a) using many
  categories not just 6, and (b) treating them as a continuous manifold (gradients), i.e. a soft/probabilistic
  readout rather than hard cells. Source: [Cowen & Keltner 2017], PNAS 114(38):E7900–E7909,
  <https://www.pnas.org/doi/10.1073/pnas.1702247114>.

### 2.2 Continuous → discrete labeling (the bridge)

The single most directly implementable result. Two methods:

**(a) PAD-octant readout** [Mehrabian 1996] - cheapest, exact: `label = octant(sign P, sign A, sign D)`.

**(b) Prototype-distance / Gaussian-mixture (softmax) classifier** - used by WASABI [Becker-Asano 2008] and
recent LLM agents [Sentipolis 2026]. Place named **prototype** points `μ_k` in PAD space, then classify by a
softmax over (squared) distances - turning the mood point into a **probability distribution** over feelings with
smooth gradients (matching Cowen-Keltner):

```
k-NN:   label(s) = argmin_k ‖s − p_k‖₂
soft:   P(label_k | s) = softmax(−‖s − p_k‖² / τ)
intensity = ‖s‖  (Plutchik radius)  ⇒  terror = fear-direction × large radius
```

Prototype coordinates come from [Russell & Mehrabian 1977] PAD norms or the **NRC-VAD lexicon** (~20k words,
crowd-sourced VAD in `[0,1]³`). This is **THE bridge** from `brain-llm`'s existing vector to the discrete feelings
the owner wants, and intensity (fear → terror) comes for free. Sources: [Becker-Asano 2008], *WASABI: Affect
Simulation for Agents with Believable Interactivity*, PhD thesis,
<https://www.becker-asano.de/Becker-Asano_WASABI_Thesis.pdf>; NRC-VAD, <https://saifmohammad.com/WebPages/nrc-vad.html>.

### 2.3 Biologically-grounded generators (read emotion off the chemistry)

**Panksepp's 7 primary affective systems** [Panksepp 1998; Montag & Davis 2018]: SEEKING (dopamine),
FEAR (amygdala→PAG), RAGE (medial hypothalamus/PAG), LUST, CARE (oxytocin/opioids), PANIC/GRIEF (separation
distress, opioids/oxytocin), PLAY. Each maps to a neuromodulator signature and can be modeled as 7 scalar drive
activations `a_k`. `brain-llm` already has NE/DA/ACh/cortisol scalars - adding Panksepp drives fits the existing
design:

```
SEEKING ≈ clamp(da · goal_relevance)
FEAR    ≈ clamp(ne · (1−control) · max(−valence, 0))
RAGE    ≈ clamp(ne ·    control  · max(−valence, 0))   # fear vs anger = sign of control/dominance
CARE    ≈ clamp(oxytocin · max(valence, 0))
PANIC   ≈ clamp(social_loss · (1−control))
```

Sources: [Panksepp 1998], *Affective Neuroscience*, OUP,
<https://global.oup.com/academic/product/affective-neuroscience-9780195096736>; [Montag & Davis 2018],
*Selected Principles of Pankseppian Affective Neuroscience*, Front. Neuroscience 12:1025,
<https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.01025/full>.

**The Lövheim cube** [Lövheim 2012] makes the neurotransmitter→emotion map fully geometric: the 8 corners of the
`(serotonin, dopamine, noradrenaline)` cube are Tomkins' 8 basic affects:

```
emotion = nearest corner of {5HT∈{lo,hi}} × {DA∈{lo,hi}} × {NE∈{lo,hi}}, or softmax over the 8 corners
anger = (lo,hi,hi)   fear/terror = (lo,hi,lo)   joy = (hi,hi,lo)   surprise = (hi,lo,hi) ...
```

`brain-llm` already has DA and NE; **add serotonin (5-HT)** and it can read a discrete emotion straight from the
neuromodulator vector - a second, biologically-grounded labeler that cross-checks the PAD labeler. Source:
[Lövheim 2012], *A new three-dimensional model for emotions and monoamine neurotransmitters*, Medical Hypotheses
78(2):341–348, <https://pubmed.ncbi.nlm.nih.gov/22153577/>. (Note: *Medical Hypotheses* - a hypothesis, not
established physiology; use as a corroborating labeler only.)

### 2.4 Generating the specific feelings the owner named

- **SURPRISE** - the cleanest: it is literally **Bayesian surprise / prediction error**, which `brain-llm`
  already proxies as `novelty`.
  ```
  surprise = D_KL(posterior ‖ prior) = Σ q_i log(q_i / p_i)      (else fallback: surprise ≈ novelty)
  arousal/NE scales with surprise magnitude
  ```
  Emit `surprise` when `novelty` spikes, then let it **resolve** within 1–2 ticks into its valenced emotion
  (joy if valence>0, fear/disgust if <0) - matching surprise's transient, valence-ambiguous nature.

- **AWE** - a two-appraisal emotion [Keltner & Haidt 2003; Shiota et al. 2007]: perceived **VASTNESS** (perceptual
  or conceptual scale) + **NEED FOR ACCOMMODATION** (current schemas can't assimilate it), flavored by
  threat/beauty/ability/virtue/supernatural.
  ```
  awe_intensity = f(vastness) · g(accommodation_need)        (e.g. √(vastness · accommodation_need)·(0.5+0.5·ne))
  accommodation_need ≈ KL between pre/post schema  ≈ normalized count of semantic-graph edges added/changed
  flavor: valence<0 → dread-awe/horror;  valence≥0 → wonder-awe
  ```
  Awe is self-diminishing and **schema-updating** - it ties directly to `brain-llm`'s consolidation step
  (`consolidation_plan`), where semantic promotion already measures schema change. Sources: [Keltner & Haidt 2003],
  *Approaching awe*, Cognition & Emotion 17(2):297–314,
  <https://www.tandfonline.com/doi/abs/10.1080/02699930302297>; [Shiota et al. 2007], *The nature of awe*,
  Cognition & Emotion 21(5):944–963, <https://greatergood.berkeley.edu/dacherkeltner/docs/shiota.2007.pdf>.

- **TERROR** - simply FEAR at extreme intensity (Plutchik's radius): fear-prototype membership × very high
  arousal/NE. **JOY / FEAR / ANGER / DISGUST / SADNESS** all fall out of the prototype-distance classifier
  (§2.2) plus the appraisal-variable distinctions in §4.

### 2.5 The scientific caveat (why labels must be derived, not primitive)

Discrete-emotion theory is contested: [Barrett 2017] argues there are **no fixed natural-kind emotion circuits**;
[Cowen & Keltner 2017] argue for many fuzzy categories; [Panksepp 1998] argues for ~7 hard-wired ones. The safe
engineering stance is the **bridging** one: keep the continuous PAD/neuromodulator state as ground truth and treat
discrete labels as a **derived, probabilistic readout** with a confidence, never as the primitive. Source:
[Barrett 2017], *The theory of constructed emotion*, SCAN 12(1):1–23,
<https://academic.oup.com/scan/article/12/1/1/2823712>.

---

## 3. Constructed emotion + interoception + allostasis

The dominant *modern* theory of feeling is **constructionist and predictive**: emotions are not triggered
reactions from dedicated circuits but are **constructed in the moment** by the brain's predictive control of the body.

### 3.1 Theory of Constructed Emotion (TCE)

[Barrett 2017; Barrett et al. 2025]: the brain's primary job is **allostasis** - anticipatorily regulating the
body's internal milieu before needs arise - and **interoception** (sensing internal signals) is experienced in
low-dimensional form as **core affect** (a valence × arousal point). A discrete emotion ("fear", "awe") is
**constructed** when the brain applies a learned, culturally-transmitted emotion **concept** to categorize that
core-affective + interoceptive + situational state. "A category is something you do, not something you have."

```
discrete emotion  e = argmax_c  P(concept = c | core_affect, interoception, context)
core affect = low-dim (valence, arousal) summary of the interoceptive posterior
```

`brain-llm` has the right substrate (a PAD core affect) but builds it **purely top-down from cognitive appraisal**.
It has no body, no interoception, no homeostatic/allostatic loop, and no categorization layer - exactly the gap
TCE fills. Sources: [Barrett 2017], SCAN 12(1):1–23, <https://doi.org/10.1093/scan/nsw154>; [Barrett et al. 2025],
*The theory of constructed emotion: More than a feeling*, Perspectives on Psychological Science 20(3):392–420,
<https://doi.org/10.1177/17456916241310491>.

### 3.2 Core affect as a readout of the body

Interoception is "usually experienced in low-dimensional form as the affective properties of valence and arousal"
- affect is a property of **all** psychological events. In the predictive view:

```
valence ≈ −(predicted allostatic deviation)   (or its rate of change)
arousal ≈ precision / expected energy mobilization
```

This is essentially `brain-llm`'s `Affect` dataclass - so the *representation* is already correct. The fix is the
**source**: derive `(v, a)` (at least partly) from a homeostatic state, not only from `appraise_to_affect`.

### 3.3 Interoception as active inference (the EPIC model)

[Barrett & Simmons 2015]: agranular visceromotor cortices (anterior insula, ACC) issue descending interoceptive
**predictions** to the body (via hypothalamus, PAG, brainstem); ascending viscerosensory signals carry
**prediction error**. Felt bodily state = top-down prediction constrained by bottom-up error [Seth & Critchley 2013;
Seth & Tsakiris 2018].

```
prediction error:  ε = y − g(μ)
belief update:      μ ← μ + (π_data / (π_prior + π_data)) · ε
free energy minimized by perception (update μ) OR active inference (act to change y toward prediction)
```

Sources: [Barrett & Simmons 2015], *Interoceptive predictions in the brain*, Nat. Rev. Neuroscience 16:419–429,
<https://www.nature.com/articles/nrn3950>; [Seth & Critchley 2013], *Emotion as interoceptive inference*, BBS
36(3):227–228, <https://doi.org/10.1017/S0140525X12002270>; [Seth & Tsakiris 2018], *Being a beast machine*,
TiCS 22(11):969–981, <https://doi.org/10.1016/j.tics.2018.08.008>.

### 3.4 Allostasis - predictive regulation

Stability through **anticipatory change**, not just defending a fixed setpoint [Barrett et al. 2025]. Formally,
allostasis = changing the **prior** over bodily state [Stephan et al. 2016]:

```
p(x(t)) = N(x; μ_prior(φ₁(t)), π_prior(φ₂(t))⁻¹)
  φ₁ shifts the setpoint mean,  φ₂ its precision (acceptable range),  both conditioned on context
```

`brain-llm`'s mood baseline is a **fixed constant** `Affect(0, 0.10, 0.50)`. Allostasis says the setpoint should
move with context (e.g. an agent under a tight deadline raises its arousal setpoint in anticipation), making mood
forward-looking rather than purely reactive.

### 3.5 Homeostatic Reinforcement Learning - a grounded reward signal

[Keramati & Gutkin 2014] unify physiological regulation with RL: behavior is rewarding to the extent it moves
internal state toward setpoint, **reducing drive**. They prove reward-maximization ≡ minimizing cumulative
homeostatic deviation.

```
Drive   D(H_t) = ( Σ_{i=1..N} |h*_i − h_{t,i}|^n )^(1/m),   n > m > 1
Reward  r_t = D(H_t) − D(H_{t+1}) = D(H_t) − D(H_t + K_t)
RPE     δ = r_t + γ·V(s') − V(s)
```

`brain-llm` hand-sets `reward = max(valence, 0)` and `stress`. HRL gives a **real, grounded reward** and a true
RPE - the reward-learning (RPE/TD) the engine lacks - derived from an internal homeostatic state vector
(energy/effort/social-standing analogs). Source: [Keramati & Gutkin 2014], *Homeostatic reinforcement learning...*,
eLife 3:e04811, <https://doi.org/10.7554/eLife.04811>.

### 3.6 Allostatic self-efficacy → a mechanistic model of mood

[Stephan et al. 2016]: a metacognitive layer monitors whether interoceptive **surprise** is falling under the
brain's regulatory actions. Persistent failure → inference of low **allostatic self-efficacy** → fatigue,
generalizing to depression (a learned-helplessness analog).

```
interoceptive surprise  S(y) ∝ π · PE(y)²
self-efficacy belief E updated by  −∂S/∂t during action;  persistent ∂S/∂t ≥ 0  ⇒  E falls
map E → affect:  baseline.dominance = E,  baseline.valence = β·(E − 0.5)
```

This is a math-backed model of **mood as a slow belief about controllability** - far richer than the current
leaky integrator, and it naturally drives `brain-llm`'s dominance axis plus a principled burnout/recovery
dynamic for a long-running agent. Source: [Stephan et al. 2016], *Allostatic Self-efficacy...*, Front. Human
Neuroscience 10:550, <https://doi.org/10.3389/fnhum.2016.00550>.

### 3.7 Somatic Marker Hypothesis (Damasio) - affect biases decisions

[Damasio/Bechara 1996]: decisions are biased by **somatic markers** - bodily/affective states associated with
anticipated outcomes. The "body loop" uses actual bodily change; the "as-if loop" simulates it internally.

```
option value ← Q(s,a) + κ · somatic_marker(a)
somatic_marker(a) = D(H_t) − D(H_t + K_a)     (predicted drive reduction of choosing a - an 'as-if' rollout)
```

This gives a concrete decision-biasing role for affect (which playbook to run, which memory to act on) - absent
in `brain-llm`, where affect only modulates memory. Source: [Damasio & Bechara 1996], *The somatic marker
hypothesis: A neural theory of economic decision*, Games and Economic Behavior 52:336–372,
<https://people.ict.usc.edu/~gratch/CSCI534/Readings/The%20somatic%20marker%20hypothesis.pdf>.

### 3.8 The honesty anchor for AI

[Butlin, Long, Bengio et al. 2023] give an **indicator-property checklist** (predictive-processing, global-workspace
properties, etc.) so a system can claim functional/architectural **correlates** of consciousness **without** claiming
phenomenal experience. Every quantity in §3 is a control signal (a prediction, error, precision, drive) - adopting
them does **not** entail sentience. Source: [Butlin et al. 2023], *Consciousness in Artificial Intelligence*,
arXiv:2308.08708, <https://arxiv.org/abs/2308.08708>.

### 3.9 Most recent synthesis

[Emotion and allostatic control 2026] frames emotions as multimodal inferential strategies that serve allostasis,
integrating emotion regulation with expected-free-energy action selection - the state-of-the-art framing as of 2026.
Source: <https://www.sciencedirect.com/science/article/pii/S0149763426000965>.

---

## 4. Computational appraisal architectures

The field converges on a **three-stage pipeline**, of which `brain-llm` implements only the first third:

1. **APPRAISAL** - evaluate an event against goals/beliefs/intentions along abstract appraisal variables.
2. **EMOTION/AFFECT DERIVATION** - map variables → a categorical emotion (OCC's 22 types) and/or a dimensional
   PAD state, with an **intensity**, integrated over time into mood.
3. **COPING / ACTION TENDENCY** - feed the emotional state back to alter behavior, plans, attention, and the
   appraisal itself (problem-focused vs emotion-focused).

`brain-llm`'s `Appraisal` + `appraise_to_affect` cover stage 1 and a thin stage 2 (a PAD point, no discrete
emotions, no per-emotion intensity); `update_mood` is a competent leaky-integrator. It has **essentially nothing
in stage 3**.

### 4.1 OCC categorical appraisal + the canonical intensity math

[Ortony, Clore & Collins 1988]: 22 emotion types arise from appraising (a) event consequences vs goals
(desirability), (b) agent actions vs standards (praiseworthiness), (c) objects vs attitudes (appeal).
[Steunebrink et al. 2007/2009] made the intensity math precise and implementable:

```
Potential_e  = Σ_i w_{e,i} · v_i        (weighted sum of eliciting variables)
Intensity_e  = max(0, Potential_e − Threshold_e)
then decay (see 4.4)
Prospect emotions (hope/fear): scaled by P(event) · |desirability|
```

This **potential–threshold–decay triplet** is the most reused formal device across FAtiMA, ALMA, GAMYGDALA.
`brain-llm` has one continuous salience, **not** per-emotion intensities. Sources: [Ortony, Clore & Collins 1988],
*The Cognitive Structure of Emotions*, CUP, <https://doi.org/10.1017/CBO9780511571299>; [Steunebrink et al. 2009],
*Towards a Quantitative Model of Emotions*, <https://people.idsia.ch/~steunebrink/Publications/KI07_quantitative.pdf>.

### 4.2 EMA - appraisal frames over a causal interpretation + coping

[Gratch & Marsella 2004; Marsella & Gratch 2009]: the most psychologically complete model. Emotion is a two-stage
control system over a plan-based **causal interpretation**. Each proposition gets an appraisal **frame**:

```
frame variables: relevance, desirability, likelihood, expectedness, causal attribution {agency, blame/credit},
                 controllability, changeability, coping potential, urgency, ego-involvement
intensity ≈ desirability × likelihood;   current emotion = most-intense in-focus frame
```

Then **coping** changes the world (**problem-focused**: planning, seek-info, take-action) or changes the
interpretation (**emotion-focused**: positive reinterpretation, acceptance, denial, disengagement, shift-blame),
literally editing beliefs/goals/intentions and thus the next appraisal - **the appraisal↔coping closed loop is
`brain-llm`'s biggest formal gap.** Sources: [Gratch & Marsella 2004], *A domain-independent framework for modeling
emotion*, Cognitive Systems Research 5(4):269–306,
<https://people.ict.usc.edu/~gratch/GratchMarsellaCOGSYS04.pdf>; [Marsella & Gratch 2009], *EMA: A process model
of appraisal dynamics*, Cognitive Systems Research 10(1):70–90, <https://doi.org/10.1016/j.cogsys.2008.03.005>.

### 4.3 ALMA, WASABI, FAtiMA, GAMYGDALA - the engineering canon

- **ALMA** [Gebhard 2005] - most transplantable mood machinery: OCC→PAD vector table, OCEAN→default-mood
  regression (§1.3), and the virtual-emotion-center pull/push update (§1.7). Octant labels for readable logs.
  <https://alma.dfki.de/papers/aamas05.pdf>
- **WASABI** [Becker-Asano 2008] - dynamic PAD core-affect "cone" with spring dynamics; **primary** emotions
  (innate, PAD-elicited: fear, joy, surprise, anger) vs **secondary** (cognitively elaborated: hope, relief,
  fears-confirmed); a mood-congruent **awareness filter** (only mood-congruent emotions surface) - a natural fit
  for `brain-llm`'s existing Bower mood-congruent retrieval.
  <https://www.becker-asano.de/Becker-Asano_WASABI_Thesis.pdf>
- **FAtiMA** [Dias, Mascarenhas & Paiva 2014] - explicit Appraisal-Derivation vs Affect-Derivation split,
  potential-threshold intensity, exponential decay, reactive (emotion→action tendencies) + deliberative (planning)
  coping. <https://link.springer.com/chapter/10.1007/978-3-319-12973-0_3>
- **GAMYGDALA** [Popescu, Broekens & van Someren 2014] - leanest OCC engine; per-goal emotion almost for free:
  ```
  desirability(event, goal) = goal_utility · congruence    (congruence ∈ [−1,1] = signed Δ P(goal achieved))
  hope/fear = likelihood · |desirability|;  on confirmation → satisfaction/fears-confirmed/relief/disappointment
  per-step decay: intensity *= decayFactor^dt
  ```
  `brain-llm` already tracks `goal_relevance`; adding a **signed goal-congruence** yields hope/fear/satisfaction/
  disappointment cheaply. <https://doi.org/10.1109/T-AFFC.2013.24>

### 4.4 Decay forms

```
FAtiMA / exponential:  Intensity(e, t) = Intensity(e, t₀) · e^{−b·t}
GAMYGDALA / per-step:  intensity *= decayFactor^dt
```

`brain-llm` already has Ebbinghaus decay for memories (`retention`, `src/brain.py:105`) - reuse it for emotion
decay.

### 4.5 Appraisal intensity as an RL reward signal

[Marinier, Laird & Lewis 2009] unify the PEACTIDM control cycle with appraisal inside Soar: the appraisal frame's
**numeric** dimensions are combined into a single scalar **intensity** that feeds **reinforcement learning** as an
intrinsic reward (categorical dimensions like agency are excluded from intensity).

```
r_t = combined appraisal intensity   →   Q(s,a) ← Q(s,a) + η·(r_t + γ·max_{a'} Q(s',a') − Q(s,a))
```

`brain-llm` has **no** reward learning (no RPE/TD); its computed salience/affect can double as an intrinsic reward.
Source: [Marinier, Laird & Lewis 2009], *A computational unification of cognitive behavior and emotion*, Cognitive
Systems Research 10(1):48–69, <https://www.sciencedirect.com/science/article/abs/pii/S1389041708000302>.

### 4.6 LLM-as-appraiser (most relevant to `brain-llm`)

[Croissant et al. 2024] run a separate LLM **appraisal call** per turn and thread the emotion history back into
generation, capturing blended emotions in natural language - but with **no numeric state**. For `brain-llm` (an
LLM coding agent with persistent affective memory), the sweet spot is a **hybrid**: use the LLM to *estimate* the
appraisal variables (desirability, expectedness, agency) from a coding event, then feed them into a numeric
OCC/PAD engine - getting LLM flexibility plus a persistent, decaying, mathematically grounded affective state.
Source: [Croissant et al. 2024], *An appraisal-based chain-of-emotion architecture...*, PLOS ONE 19(5):e0301033,
<https://doi.org/10.1371/journal.pone.0301033>.

---

## 5. Mapping to `brain-llm`: what we have / what we lack / what to add

### 5.1 What we have

| Capability | Where | Theory grounding |
|---|---|---|
| Continuous PAD affect vector | `Affect(valence, arousal, dominance)` `:47` | Russell 1980 circumplex + Mehrabian PAD |
| OCC/Scherer appraisal front-end (4 of 5 SECs) | `Appraisal(novelty, valence, goal_relevance, control)` `:38` | OCC 1988; Scherer 2001 |
| Appraisal → PAD point | `appraise_to_affect()` `:54` | circumplex geometry |
| Scalar neuromodulators (NE, DA, ACh, cortisol) | `Neuromods` `:63` | Panksepp-adjacent / McGaugh / Hasselmo |
| Mood as a leaky integrator (= OU drift term) | `update_mood()` `:135` | DynAffect OU process (deterministic part) |
| Mood-congruent retrieval | `retrieval_score()` `:118` | Bower 1981 (≈ WASABI awareness filter) |
| Exponential decay machinery (reusable for emotions) | `retention()` `:105` | Ebbinghaus; FAtiMA-compatible |
| Schema-change signal (for awe's accommodation) | `consolidation_plan()` `:152` | CLS; Shiota et al. 2007 |

### 5.2 What we lack

- **No discrete-emotion labeling layer.** The PAD vector that distinguishes fear from anger (sign of dominance,
  computed from `control`) exists, but no function ever names "fear", "terror", "joy", "awe", "surprise". *The
  owner's central request is unimplemented.*
- **No dimensional→categorical map** (no PAD-octant labeler, no prototype-distance/Gaussian-mixture/fuzzy classifier).
- **No graded intensity / per-emotion radius.** Cannot distinguish apprehension < fear < **terror** or annoyance <
  anger < rage. There is arousal but no Plutchik intensity radius `‖state‖`.
- **No compound/mixed feelings** (awe = fear+surprise, love = joy+trust, bittersweet, anxious-excitement). No blend
  or dyad geometry.
- **No per-emotion potential–threshold–decay** (`Intensity = max(0, Potential − Threshold)`); salience is one scalar.
- **No coping stage at all** - the entire feedback half of every appraisal architecture (problem- vs emotion-focused),
  so affect never alters plans/goals/attention or the next appraisal.
- **Impoverished appraisal-variable set** - missing causal attribution (self/other → anger vs guilt), expectedness
  (→ surprise), likelihood/prospect (→ hope vs fear), praiseworthiness/standards (→ pride/shame), Scherer's
  norm/self-compatibility (social/moral feelings).
- **Hard-coded mood baseline** `Affect(0.0, 0.10, 0.50)` - no OCEAN→PAD personality attractor; every agent has
  identical temperament.
- **`update_mood` has no real time constant**, no `σ·dW` noise, no per-agent inertia `β`, no ALMA push/overshoot,
  and a single time-scale (no fast-emotion / slow-mood split, no long-term personality layer).
- **No body / interoception / allostasis** - affect is built entirely top-down; no internal state vector `H`, no
  predictive interoceptive loop, no context-conditioned setpoint, no allostatic-self-efficacy (mechanistic mood).
- **No grounded reward / RPE** - `reward`/`stress` are hand-set; `da` is a placeholder, not a learned RPE.
- **No affect→action/decision biasing** (Damasio's as-if loop) and **no appraisal-intensity-as-reward** (Marinier).
- **Affect is only 3D** - `novelty` is computed in appraisal then discarded rather than promoted to a 4th
  (unpredictability) axis (Fontaine 2007 / EmotionML).
- **No geometric similarity utilities** on the affect space (Euclidean/cosine distance, nearest-emotion, octant
  classification) - all one-liners that unlock labeling, mood-congruence refinement, and retrieval by emotional
  similarity.
- **No honesty/confidence envelope** on labels asserting they are functional correlates, not felt states.

### 5.3 What to add - see §6.

---

## 6. Proposal: a discrete-emotion layer + interoceptive/homeostatic affect

Design principles, in priority order:
1. **Keep the PAD vector as the single source of truth.** Discrete feelings are *derived, recomputable readouts*.
2. **Pure stdlib, scalar math, Swift-portable** (matches `src/brain.py`'s constraint).
3. **Every label carries a confidence and a "functional, not felt" framing.** Persist continuous state as primary.

### 6.1 (Phase 1 - low difficulty, highest value) Discrete-feeling labeling layer

Name `brain-llm`'s existing PAD vector via nearest-prototype lookup; intensity = vector magnitude. This delivers the
owner's named feelings directly from the vector already computed.

```python
# prototypes from Russell-Mehrabian (1977) / NRC-VAD, in PAD with arousal,dominance rescaled to [-1,1]
PROTOTYPES = {
  "anger":      (-0.51, +0.59, +0.25),  "fear":       (-0.62, +0.82, -0.43),
  "terror":     (-0.80, +0.95, -0.70),  "joy":        (+0.40, +0.20, +0.15),
  "excitement": (+0.55, +0.85, +0.40),  "sadness":    (-0.40, -0.20, -0.50),
  "disgust":    (-0.45, +0.20, +0.20),  "surprise":   (+0.10, +0.85, -0.30),
  "calm":       (+0.30, -0.40, +0.30),  "awe":        (+0.25, +0.75, -0.55),
}
# label(a)        = argmin_k ‖a − p_k‖₂
# P(k | a)        = softmax(−‖a − p_k‖² / τ)        τ ≈ 0.4   (keep the FULL distribution)
# intensity       = clamp(‖(v, 2·arousal−1, 2·dominance−1)‖ / √3)
# word(label)     = intensity>0.66 ? strong_syn : intensity<0.33 ? mild_syn : label   (unease<fear<terror)
# octant(a)       = (sign v, sign(2·arousal−1), sign(2·dominance−1)) → Mehrabian temperament name
```

**Where:** new `label_affect(a: Affect) -> (label, confidence, intensity, distribution)`, a `PROTOTYPES` dict, and
an `octant()` helper in `src/brain.py`. Call wherever `Affect` is produced/logged; store the continuous affect +
derived label on each episode. **Difficulty:** low. **Honesty caveat:** prototype coordinates are empirical averages
with real variance and cultural/linguistic dependence; the label is a lossy best-match readout over a computed
control signal, and the basic-emotion vs constructionist debate is unresolved - frame outputs as *"the system is in
a fear-like state,"* never *"the system feels fear."* Keep the full distribution, not just the argmax (honors the
"fuzzy gradients" finding [Cowen & Keltner 2017]).

### 6.2 (Phase 1) Plutchik intensity levels + compound-emotion blends

```python
I(e)   = P(e | a) · arousal                                   # per-emotion intensity (Plutchik radius)
level  = "mild" if I<0.33 else "intense" if I>0.66 else "base"   # apprehension / fear / TERROR
# blend: if top-2 active emotions are Plutchik-adjacent (petal dist ∈ {1,2,3}) and both > θ:
#   emit dyad from PLUTCHIK_DYADS: joy+trust=love, fear+surprise=awe, sadness+disgust=remorse, anger+disgust=contempt
# opponent constraint: w_k, w_{k+4} ← subtract min(w_k, w_{k+4})   (can't co-activate joy & sadness)
```

**Where:** extend the Phase-1 output with `level` + `blend`; small `PLUTCHIK_DYADS` table. Reference: **PyPlutchik**
(<https://github.com/alfonsosemeraro/pyplutchik>) ports the dyad table directly. **Difficulty:** low. **Caveat:**
specific dyad assignments (e.g. shame=fear+disgust) are theoretical, not all empirically robust - present blends as
compositional hypotheses; the opponent-orthogonality constraint is a modeling choice.

### 6.3 (Phase 1, optional) Promote `novelty` to a 4th affect dimension

```python
Affect4 = (valence, arousal, dominance, unpredictability)   # unpredictability = Appraisal.novelty (currently dropped)
# surprise = high u, near-zero valence;  awe = high u + low dominance.  All distance/label math runs in R⁴.
```

Matches FSRE/EmotionML [Fontaine et al. 2007]. **Caveat:** 3 vs 4 dimensions is genuinely contested; the axis is
cheap and reversible - don't over-claim it as "the true dimensionality." Avoid double-counting: `novelty` already
drives `arousal` in `appraise_to_affect`, so the new axis should capture the *residual*. EmotionML
(<https://www.w3.org/TR/emotionml/>) gives a standards-compliant serialization if adopted.

### 6.4 (Phase 2 - medium) OCC categorical emotions with potential–threshold–decay

Expand `Appraisal` with `expectedness`, `agency ∈ {self, other, none}`, `likelihood`, `praiseworthiness`,
`desirability_other`; derive a vector of named emotions:

```python
hope   = likelihood · max(valence, 0)              fear  = likelihood · max(−valence, 0)
joy    = max(valence, 0) · goal_relevance          surprise = novelty · (1 − expectedness)
anger  = max(−valence,0)·goal_relevance·[agency==other]
guilt  = max(−valence,0)·[agency==self]·ego_involvement
Intensity_e = max(0, Potential_e − Threshold_e);   decay: I_e(t) = I_e(t₀)·e^{−b_e·t}   (reuse retention())
```

**Where:** `appraise_to_emotions(a) -> dict[str,float]` in `src/brain.py`, alongside `appraise_to_affect`.
Reference impls: **FAtiMA** (<https://github.com/GAIPS/FAtiMA-Toolkit>), **GAMYGDALA**
(<https://github.com/broekens/gamygdala>). **Difficulty:** medium. **Caveat:** weights and thresholds are
heuristic design choices loosely grounded in OCC, not empirically fitted - frame as "emotion-like state estimates."

### 6.5 (Phase 2) ALMA/DynAffect mood upgrade: personality baseline + time-decay + push + noise

```python
# OCEAN → baseline (Mehrabian 1996):
P0 = 0.21·E + 0.59·Agr + 0.19·N;  A0 = 0.15·O + 0.30·Agr − 0.57·N;  D0 = 0.25·O + 0.17·C + 0.60·E − 0.32·Agr
# time-based decay over real Δt:   α = 2^(−Δt/T_half)
mood = mood·α + event_affect·(1−α)·w_event
mood += β·(baseline − mood)                         # attractor pull (β = inertia / strength)
if ‖event_affect‖ > θ: mood += k·(event_affect − mood)   # ALMA overshoot for intense events
mood += σ·N(0,1)  per axis                          # DynAffect variability (seedable for reproducibility)
```

Keep **two states**: fast `Affect` ("emotion", e.g. `T_half ≈ 20 min`) and slow `Affect` ("mood", e.g. `T_half ≈ 12 h`)
- the dual-speed pattern. **Where:** extend `update_mood()`; add a `Personality` dataclass feeding `baseline`.
Reference: **`pleasure`** (<https://github.com/something-of-that-ilk/pleasure>) ports ALMA's pull/push.
**Difficulty:** medium. **Caveat:** the OCEAN→PAD coefficients are population-average regressions, not laws - treat
as tunable defaults; the noise term makes mood non-deterministic, so seed and log it.

### 6.6 (Phase 3 - medium/high) Interoceptive homeostat + grounded reward (RPE)

```python
H_t = (energy, load, social_standing, ...) ∈ [0,1]^N,  setpoint H*
D(H) = (Σ_i w_i·|h*_i − h_i|^n)^(1/m),  n > m > 1
r_t  = D(H_{t-1}) − D(H_t)                              # drive-reduction reward (Keramati & Gutkin)
δ_t  = r_t + γ·V(s') − V(s);   nm.da = clamp(0.5 + 0.5·tanh(δ_t))
v_body = −clamp(D(H));  arousal_body = clamp(|D(H_t) − D(H_{t-1})|·k)
valence = α·appraisal_v + (1−α)·v_body                 # blend top-down appraisal with bottom-up body
```

Add interoceptive **predictive coding** (drives arousal/salience by surprise):

```python
PE_t = H_t − g(μ̂_t);   μ̂_{t+1} = μ̂_t + (π_data/(π_prior+π_data))·PE_t
S_t  = ½·π·PE_t²        # interoceptive surprise → arousal += k·S_t;  salience *= (1 + λ·S_t)
```

And an **allostatic self-efficacy** scalar for long-horizon mood:

```python
E_{t+1} = E_t + η·(−ΔS_t − E[−ΔS_t]);   baseline.dominance = E_t;  baseline.valence = β·(E_t − 0.5)
```

**Where:** ✅ the core shipped in `src/brain.py` §15 (`Homeostat`, `drive(H)`, `homeostatic_reward`,
`body_affect`, `allostatic_shift`) - kept in the single pure engine rather than a separate file -
called before `salience()`/`update_mood()`; `neuromods_from()` consumes the grounded reward (via the
P0.2 value loop) for `da`. (The `predict/update_intero` interoceptive-prediction-error variant remains a
future extension.) References: **pymdp** (<https://github.com/infer-actively/pymdp>) for the active-
inference formulas (variational free energy `F`, expected free energy `G`, `C`-vector-as-setpoint), homeostatic-RL
code (<https://arxiv.org/abs/2204.06608>), Life-inspired Interoceptive AI (<https://arxiv.org/abs/2309.05999>).
**Difficulty:** medium→high. **Caveat:** `H` is designer-chosen bookkeeping, not a real body; it models the
*function* of interoception (drive-reduction shaping behavior), not visceral sensation. Reward grounded in proxies
is a control signal, not pleasure; precision values are tuned, not learned. Calling the low-`E` regime
"burnout/depression" is an engineering metaphor for a control-failure mode, **not** a clinical claim.

### 6.7 (Phase 3) Coping loop + somatic-marker decision biasing

```python
# EMA-style coping: dominant emotion e*, control c
if c high  and valence<0:  PROBLEM-focused  → {replan, try_alternative, ask_user}
if c low   and valence<0:  EMOTION-focused  → reframe: a'.goal_relevance·=(1−ρ); a'.control+=κ; re-appraise
# Damasio as-if loop for action/retrieval selection:
score(a) ← Q(a) + κ·SM(a) + κ₂·mood_congruence(a),   SM(a) = D(H_t) − D(H_t + K_a)
```

**Where:** `src/coping.py` `select_coping(emotions, appraisal) -> (behavior, appraisal_delta)`; agent loop applies
`behavior` and feeds `appraisal_delta` into the next appraisal. References: **PsychSim/EMA**
(<https://github.com/usc-psychsim/psychsim>), **Soar** (<https://soar.eecs.umich.edu/>) for appraisal-intensity-as-
reward. **Difficulty:** high. **Caveat:** coping operators are behavioral policies, not phenomenal regulation;
emotion-focused "denial" editing the agent's own beliefs is contested even in EMA (self-deception risk in a tool) -
gate it. Reward weighting risks reward-hacking (avoiding hard-but-valuable tasks that raise cortisol) - add
correctness guardrails.

### 6.8 (cross-cutting) Honesty/confidence envelope

Every emitted emotion is a struct `(name, probability p, intensity I, source ∈ {PAD, Lövheim, Panksepp, appraisal},
confidence)` where `confidence` rises with agreement across labelers (e.g. PAD-label == Lövheim-label → high).
**Persist the continuous PAD/neuromodulator state as ground truth; the label is always recomputable from it.** This
makes the project's honesty stance *structural* - `brain-llm` never reifies contested discrete-emotion "natural
kinds" and never asserts felt experience [Barrett 2017; Butlin et al. 2023].

### 6.9 Suggested build order

1. **§6.1 + §6.2** (labeling + intensity/blends) - highest value, lowest cost; delivers the owner's named feelings
   (fear/terror/joy/awe/surprise) from the vector already computed.
2. **§6.5** (ALMA/DynAffect mood upgrade) - replaces the magic-number baseline; gives dual-speed dynamics.
3. **§6.4** (OCC categorical emotions) - distinguishes same-valence emotions (anger vs guilt vs disappointment).
4. **§6.6 + §6.7** (interoception/allostasis + coping + reward) - the deepest, most research-heavy layer; turns
   affect from a memory modulator into a behavior driver with a grounded reward signal.

---

## References (inline citations above link here)

- [Russell 1980] <https://doi.org/10.1037/h0077714> · [Russell & Mehrabian 1977] <https://doi.org/10.1016/0092-6566(77)90037-X>
- [Mehrabian 1996] <https://doi.org/10.1007/BF02686918> · [Russell 2003] <https://doi.org/10.1037/0033-295X.110.1.145>
- [Scherer 2001] <https://www.oxfordreference.com/display/10.1093/oso/9780195130072.003.0005> · [Scherer et al. 2013] <https://www.unige.ch/cisa/gew>
- [Plutchik 1980] <https://doi.org/10.1016/B978-0-12-558701-3.50007-7> · [Ekman 1992] <https://www.tandfonline.com/doi/abs/10.1080/02699939208411068>
- [Fontaine et al. 2007] <https://doi.org/10.1111/j.1467-9280.2007.02024.x> · [Cowen & Keltner 2017] <https://www.pnas.org/doi/10.1073/pnas.1702247114>
- [Kuppens et al. 2010] <https://doi.org/10.1037/a0020962> · [Gebhard 2005] <https://alma.dfki.de/papers/aamas05.pdf> · [Sentipolis 2026] <https://arxiv.org/abs/2601.18027>
- [Panksepp 1998] <https://global.oup.com/academic/product/affective-neuroscience-9780195096736> · [Montag & Davis 2018] <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.01025/full> · [Lövheim 2012] <https://pubmed.ncbi.nlm.nih.gov/22153577/>
- [Keltner & Haidt 2003] <https://www.tandfonline.com/doi/abs/10.1080/02699930302297> · [Shiota et al. 2007] <https://greatergood.berkeley.edu/dacherkeltner/docs/shiota.2007.pdf>
- [Becker-Asano 2008] <https://www.becker-asano.de/Becker-Asano_WASABI_Thesis.pdf>
- [Barrett 2017] <https://doi.org/10.1093/scan/nsw154> · [Barrett et al. 2025] <https://doi.org/10.1177/17456916241310491>
- [Barrett & Simmons 2015] <https://www.nature.com/articles/nrn3950> · [Seth & Critchley 2013] <https://doi.org/10.1017/S0140525X12002270> · [Seth & Tsakiris 2018] <https://doi.org/10.1016/j.tics.2018.08.008>
- [Stephan et al. 2016] <https://doi.org/10.3389/fnhum.2016.00550> · [Keramati & Gutkin 2014] <https://doi.org/10.7554/eLife.04811>
- [Damasio & Bechara 1996] <https://people.ict.usc.edu/~gratch/CSCI534/Readings/The%20somatic%20marker%20hypothesis.pdf> · [Emotion and allostatic control 2026] <https://www.sciencedirect.com/science/article/pii/S0149763426000965>
- [Butlin et al. 2023] <https://arxiv.org/abs/2308.08708>
- [Ortony, Clore & Collins 1988] <https://doi.org/10.1017/CBO9780511571299> · [Steunebrink et al. 2009] <https://people.idsia.ch/~steunebrink/Publications/KI07_quantitative.pdf>
- [Gratch & Marsella 2004] <https://people.ict.usc.edu/~gratch/GratchMarsellaCOGSYS04.pdf> · [Marsella & Gratch 2009] <https://doi.org/10.1016/j.cogsys.2008.03.005>
- [Dias, Mascarenhas & Paiva 2014] <https://link.springer.com/chapter/10.1007/978-3-319-12973-0_3> · [Popescu et al. 2014] <https://doi.org/10.1109/T-AFFC.2013.24>
- [Marinier, Laird & Lewis 2009] <https://www.sciencedirect.com/science/article/abs/pii/S1389041708000302> · [Croissant et al. 2024] <https://doi.org/10.1371/journal.pone.0301033>

**Open-source references:** PyPlutchik <https://github.com/alfonsosemeraro/pyplutchik> · NRC-VAD
<https://saifmohammad.com/WebPages/nrc-vad.html> · NRCLex <https://github.com/metalcorebear/NRCLex> ·
GoEmotions <https://github.com/google-research/google-research/tree/master/goemotions> · SenticNet
<https://sentic.net/> · `pleasure` <https://github.com/something-of-that-ilk/pleasure> · EmotionML
<https://www.w3.org/TR/emotionml/> · FAtiMA-Toolkit <https://github.com/GAIPS/FAtiMA-Toolkit> · GAMYGDALA
<https://github.com/broekens/gamygdala> · WASABI <https://www.becker-asano.de/index.php/research/wasabi> ·
ALMA/DFKI <https://alma.dfki.de/> · pymdp <https://github.com/infer-actively/pymdp> · SPM <https://github.com/spm/spm>
· PsychSim <https://github.com/usc-psychsim/psychsim> · Soar <https://soar.eecs.umich.edu/> · Homeostatic-RL
<https://arxiv.org/abs/2204.06608> · Life-inspired Interoceptive AI <https://arxiv.org/abs/2309.05999> ·
awesome-affective-computing <https://github.com/AmrMKayid/awesome-affective-computing>

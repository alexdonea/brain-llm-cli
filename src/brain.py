"""
brain.py - a mathematical model of human-like memory dynamics.

This reproduces the *function* of the brain's memory + affect machinery, not the
subjective experience. "Valence" here is a computed signal that behaves like
emotion in how it modulates memory; it is not a felt feeling.

Grounding (see README for full citations):
  - Affect space: Russell (1980) valence-arousal circumplex; Mehrabian PAD.
  - Appraisal: Ortony-Clore-Collins (OCC), Scherer component-process.
  - Neuromodulation of memory: McGaugh (amygdala/noradrenaline boost consolidation);
    Hasselmo (acetylcholine gates encode vs. consolidate).
  - Activation/forgetting: Anderson ACT-R base-level activation; Ebbinghaus /
    Wixted-Carpenter forgetting curve; FadeMem importance-modulated decay.
  - Consolidation: McClelland, McNaughton & O'Reilly (1995) Complementary Learning
    Systems (hippocampus fast/episodic -> neocortex slow/semantic via sleep replay).
  - Mood-congruent retrieval: Bower (1981).

Pure standard library. Port to Swift is straightforward (all functions are scalar math).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------- helpers
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x < -709.0:                              # guard the tail where exp(-x) would overflow
        z = math.exp(x)
        return z / (1.0 + z)
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- 1. APPRAISAL  (OCC -> PAD)
@dataclass
class Appraisal:
    """How an event is evaluated. The cognitive front-end of emotion (OCC)."""
    novelty: float          # 0..1  surprise / unexpectedness (Bayesian-surprise proxy)
    valence: float          # -1..1 pleasant (+) vs unpleasant (-)
    goal_relevance: float   # 0..1  how much it matters to current goals
    control: float          # 0..1  coping potential / sense of control (-> dominance)
    praiseworthiness: float = 0.0        # -1..1  praise(+)/blame(-) of the act (OCC; Scherer's 5th check) (§24)
    desirability_for_other: float = 0.0  # -1..1  good(+)/bad(-) for the other agent (OCC fortunes-of-others) (§24)

    def __post_init__(self):                                   # keep every axis inside its documented range
        self.novelty = clamp(self.novelty)
        self.valence = clamp(self.valence, -1.0, 1.0)
        self.goal_relevance = clamp(self.goal_relevance)
        self.control = clamp(self.control)
        self.praiseworthiness = clamp(self.praiseworthiness, -1.0, 1.0)
        self.desirability_for_other = clamp(self.desirability_for_other, -1.0, 1.0)


@dataclass
class Affect:
    """A point in the valence-arousal-dominance space (Russell + PAD)."""
    valence: float          # -1..1
    arousal: float          # 0..1
    dominance: float        # 0..1


def appraise_to_affect(a: Appraisal) -> Affect:
    # Arousal grows with novelty, goal relevance, and emotional intensity (|valence|):
    # both very good and very bad events are arousing. (circumplex geometry)
    arousal = clamp(0.50 * a.novelty + 0.30 * a.goal_relevance + 0.20 * abs(a.valence))
    return Affect(valence=clamp(a.valence, -1, 1), arousal=arousal, dominance=clamp(a.control))


# ---------------------------------------------------------------- 2. NEUROMODULATORS  (control gains)
@dataclass
class Neuromods:
    ne: float        # noradrenaline (locus coeruleus), PHASIC <- arousal; boosts consolidation
    da: float        # dopamine (VTA) <- reward-prediction error; salience & learning rate
    ach: float       # acetylcholine (basal forebrain): ~1 wake/encode, ~0.1 NREM/consolidate
    cortisol: float  # glucocorticoid stress: strengthens consolidation, can impair retrieval
    serotonin: float = 0.5   # 5-HT (raphe): average-reward / patience -> sets the discount; DA opponent (§18)
    oxytocin: float = 0.0    # prosocial reward weighting (§18)
    ne_tonic: float = 0.1    # TONIC LC level (slow): adaptive gain & Yerkes-Dodson (§18)


def neuromods_from(affect: Affect, reward: float, stress: float, mode: str = "wake",
                   delta: float | None = None, serotonin: float = 0.5, oxytocin: float = 0.0,
                   ne_tonic: float | None = None) -> Neuromods:
    # Dopamine: if a reward-prediction error `delta` is supplied (the TD value loop, §10), phasic DA
    # encodes the SURPRISE of reward -- baseline 0.5 when fully predicted, ->1 when better than
    # expected, ->0 when worse (Schultz 1997). Otherwise fall back to raw reward (back-compatible).
    # serotonin / oxytocin / ne_tonic are pass-throughs the agent computes via §18 (sensible defaults
    # keep old callers unchanged); ne_tonic defaults to the phasic level (no separate tonic leak).
    da = clamp(0.5 + 0.5 * math.tanh(delta)) if delta is not None else clamp(reward)
    return Neuromods(
        ne=affect.arousal,
        da=da,
        ach=1.0 if mode == "wake" else 0.1,   # low ACh in NREM enables hippocampal->cortex transfer
        cortisol=clamp(stress),
        serotonin=clamp(serotonin),
        oxytocin=clamp(oxytocin),
        ne_tonic=affect.arousal if ne_tonic is None else clamp(ne_tonic),
    )


# ---------------------------------------------------------------- 3. ENCODING SALIENCE  (McGaugh)
def salience(a: Appraisal, nm: Neuromods, w=(0.25, 0.30, 0.35, 0.10), rpe: float = 0.0,
             loss_averse: bool = False) -> float:
    """
    Initial trace strength. Base value (SCM-style four axes) MULTIPLIED by a
    neuromodulatory arousal gain: emotional arousal via the amygdala/noradrenaline
    enhances consolidation (McGaugh). High arousal can push salience > 1 -> the
    flashbulb-memory effect (vivid, durable memories of intense events).
    `rpe` (reward-prediction error, §10): an UNEXPECTED outcome -- good or bad -- encodes
    harder (we remember surprises). |rpe| boosts the gain on top of arousal; default 0.0.
    `loss_averse` (§25): weight the valence term by the prospect-theory value (a loss ~2.25x an
    equal gain) so a painful event encodes harder than an equal win; default off (symmetric).
    """
    vmag = abs(prospect_value(a.valence)) if loss_averse else abs(a.valence)   # prospect_value: §25
    base = (w[0] * a.novelty + w[1] * vmag
            + w[2] * a.goal_relevance + w[3] * (1.0 - a.control))
    arousal_gain = (1.0 + 0.8 * nm.ne + 0.4 * nm.da + 0.3 * nm.cortisol) * (1.0 + 0.5 * abs(rpe))
    return clamp(base * arousal_gain, 0.0, 1.5)


# ---------------------------------------------------------------- 4. BASE-LEVEL ACTIVATION  (ACT-R)
def base_level_activation(retrieval_times, now: float, d: float = 0.5) -> float:
    """
    B_i = ln( sum_k (now - t_k)^(-d) ),  d in [0.3, 0.7].   (Anderson, ACT-R)
    Captures BOTH recency (recent retrievals dominate) and frequency (more retrievals
    raise activation). This is why used + recent memories stay available.
    """
    s = sum((max(now - tk, 1e-6)) ** (-d) for tk in retrieval_times)
    return math.log(s) if s > 0 else float("-inf")


# ---------------------------------------------------------------- 5. RETENTION / FORGETTING  (FadeMem/Ebbinghaus)
def retention(v0: float, t: float, tau: float = 0.0, importance: float = 0.5,
              lambda_base: float = 0.6, mu: float = 2.0, beta: float = 0.8) -> float:
    """
    v(t) = v0 * exp( -lambda * (t - tau)^beta ),  lambda = lambda_base * exp(-mu * I).
    Importance I slows decay (lambda shrinks). beta<1 = sub-linear (long-term, slow);
    beta>1 = super-linear (short-term, fast). (FadeMem 2026; Ebbinghaus 1885)
    t is in days. Forgetting is a feature: it clears low-value clutter.
    """
    lam = lambda_base * math.exp(-mu * importance)
    return v0 * math.exp(-lam * (max(t - tau, 0.0)) ** beta)


# ---------------------------------------------------------------- 6. RETRIEVAL  (hybrid + mood-congruent)
def retrieval_score(mem: dict, query_relevance: float, graph_proximity: float,
                    mood: Affect, now: float, w=(0.20, 0.30, 0.30, 0.15, 0.05)) -> float:
    """
    Rank a memory for recall by:
      recency/frequency (ACT-R activation) + salience + semantic relevance
      + graph proximity + mood congruence.
    Mood congruence (Bower 1981): when current mood matches a memory's valence,
    that memory is easier to retrieve. This makes recall state-dependent, like ours.
    """
    act = base_level_activation(mem.get("retrievals", [mem["t0"]]), now)
    recency = sigmoid(act)
    congruence = 1.0 - abs(mem["affect"]["valence"] - mood.valence) / 2.0
    return (w[0] * recency + w[1] * mem["salience"] + w[2] * query_relevance
            + w[3] * graph_proximity + w[4] * congruence)


# ---------------------------------------------------------------- 7. MOOD DYNAMICS  (leaky integrator)
def update_mood(mood: Affect, event_affect: Affect, gamma: float = 0.20,
                baseline: Affect | None = None, pull: float = 0.05) -> Affect:
    """
    Mood is a slow integrator of recent affect that decays back to a baseline
    (homeostasis). One bad event nudges mood; absent new input, mood returns to set-point.
    """
    b = baseline or Affect(0.0, 0.10, 0.50)
    v = mood.valence * (1 - gamma) + event_affect.valence * gamma
    ar = mood.arousal * (1 - gamma) + event_affect.arousal * gamma
    do = mood.dominance * (1 - gamma) + event_affect.dominance * gamma
    v += pull * (b.valence - v)
    ar += pull * (b.arousal - ar)
    do += pull * (b.dominance - do)
    return Affect(clamp(v, -1, 1), clamp(ar), clamp(do))


# ---------------------------------------------------------------- 8. CONSOLIDATION  (CLS sleep cycle)
def consolidation_plan(episodes, now: float, promote_thr: float = 0.55,
                       forget_thr: float = 0.20, age_days: float = 30.0,
                       min_confidence: float = 0.0):
    """
    Complementary Learning Systems: the hippocampus (episodic) replays high-strength
    traces to the neocortex (semantic) during sleep; weak old traces decay.
      strength = salience * activation
    NREM consolidates declarative content; REM preferentially strengthens
    high-arousal (emotional) memories.
    Hallucination guard (when min_confidence > 0, the P1.3 metacognition hook): an episode is NOT
    promoted to semantic if its `confidence` < min_confidence or its `source` is "imagined" -- low-
    confidence / internally-generated content stays episodic and never hardens into a "fact".
    Returns (promote -> semantic/graph, forget -> drop).
    """
    promote, forget = [], []
    for e in episodes:
        strength = e["salience"] * sigmoid(base_level_activation(e.get("retrievals", [e["t0"]]), now))
        age = (now - e["t0"]) / 86400.0
        rem_boost = 1.0 + 0.5 * e["affect"]["arousal"]   # REM favours emotional memories
        guarded = min_confidence > 0.0 and (e.get("confidence", 1.0) < min_confidence
                                            or e.get("source") == "imagined")
        if not guarded and strength * rem_boost >= promote_thr:
            promote.append(e)
        elif strength < forget_thr and age > age_days:
            forget.append(e)
    return promote, forget


# ---------------------------------------------------------------- 9. DISCRETE EMOTION LABELING  (prototype read-out)
# A categorical *read-out* layered on the continuous PAD Affect the engine already computes -- it adds
# NO new state. Discrete feelings are regions/labels in PAD space: fear vs anger differ mainly by the
# sign of dominance (which we derive from `control`) (Russell & Mehrabian 1977). This models the
# *function* of naming a feeling, not a felt emotion: emit "a fear-like state", never "feels fear".

# Prototype coordinates in PAD, each axis on [-1, 1] (valence is already signed; arousal/dominance,
# stored on [0, 1], are mapped via 2x-1). Sources: Russell & Mehrabian (1977); NRC-VAD lexicon.
EMOTION_PROTOTYPES = {
    "fear":     (-0.62,  0.82, -0.43),
    "anger":    (-0.51,  0.59,  0.25),
    "joy":      ( 0.76,  0.48,  0.35),
    "surprise": ( 0.18,  0.88, -0.20),
    "awe":      ( 0.25,  0.75, -0.55),
    "sadness":  (-0.63, -0.27, -0.33),
    "disgust":  (-0.45,  0.20,  0.20),
    "calm":     ( 0.30, -0.40,  0.30),
}

# Plutchik (1980) intensity tiers (mild, strong): same direction, the WORD escalates with magnitude.
# "terror" is just strong fear -- the owner's named feeling falls out of intensity for free.
EMOTION_TIERS = {
    "fear":     ("apprehension", "terror"),
    "anger":    ("annoyance",    "rage"),
    "joy":      ("serenity",     "ecstasy"),
    "surprise": ("distraction",  "amazement"),
    "awe":      ("curiosity",    "wonder"),
    "sadness":  ("pensiveness",  "grief"),
    "disgust":  ("boredom",      "loathing"),
}

# Mehrabian's 8 PAD temperament octants, keyed by the sign triple (valence, arousal, dominance).
PAD_OCTANTS = {
    (1, 1, 1): "exuberant",    (1, 1, -1): "dependent",
    (1, -1, 1): "relaxed",     (1, -1, -1): "docile",
    (-1, 1, 1): "hostile",     (-1, 1, -1): "anxious",
    (-1, -1, 1): "disdainful",  (-1, -1, -1): "bored",
}


def _pad_point(a: Affect):
    """Affect -> a point in [-1,1]^3 (valence already signed; arousal/dominance 0..1 -> -1..1)."""
    return (a.valence, 2.0 * a.arousal - 1.0, 2.0 * a.dominance - 1.0)


def label_affect(a: Affect, tau: float = 0.4) -> dict:
    """
    Name the current Affect: the nearest emotion prototype (argmin Euclidean distance in PAD),
    a full softmax distribution over all prototypes, and a Plutchik intensity (radius / sqrt(3)).
    `word` upgrades the label by intensity tier (fear -> "terror" when intense; -> "apprehension"
    when faint). Keep the continuous PAD as primary -- the label is a recomputable read-out.
    NOTE: the label encodes *direction*; near the neutral origin (low intensity) the nearest
    prototype is weakly determined -- trust `intensity` and the full `dist` there, not the argmax.
    Functional label only: "a <word>-like state", not a felt emotion.
    """
    x = _pad_point(a)
    d2 = {k: sum((xi - pi) ** 2 for xi, pi in zip(x, p)) for k, p in EMOTION_PROTOTYPES.items()}
    label = min(d2, key=d2.get)
    dmin = min(d2.values())                                          # max-shift (exact): nearest term -> 1.0
    exps = {k: math.exp(-(v - dmin) / tau) for k, v in d2.items()}   # no underflow for tiny tau
    z = sum(exps.values()) or 1.0                                    # floor z -> never /0
    dist = {k: v / z for k, v in exps.items()}
    intensity = clamp(math.sqrt(sum(xi * xi for xi in x)) / math.sqrt(3.0))
    mild, strong = EMOTION_TIERS.get(label, (label, label))
    word = strong if intensity > 0.66 else mild if intensity < 0.33 else label
    return {"label": label, "word": word, "intensity": intensity, "dist": dist}


def octant(a: Affect) -> str:
    """Mehrabian's coarse PAD temperament name from the sign of each axis (mid-point counts as +)."""
    s = lambda v, mid: 1 if v >= mid else -1
    return PAD_OCTANTS[(s(a.valence, 0.0), s(a.arousal, 0.5), s(a.dominance, 0.5))]


# ---------------------------------------------------------------- 10. REWARD-PREDICTION ERROR  (TD value loop)
# Upgrades the static `da = clamp(reward)` gain into a *learning* signal. Phasic dopamine tracks the
# reward-PREDICTION error  delta = r + gamma*V(s') - V(s),  not raw reward (Schultz, Dayan & Montague
# 1997): an UNEXPECTED reward spikes dopamine; a fully predicted one does not. This is what lets the
# system register surprise-, disappointment-, or relief-LIKE states (functional, not felt) -- and
# makes every downstream gain mean something.
# `brain.py` stays pure: `V` is a plain {cue: value} dict the agent persists to
# `.memory/affect/value.yaml` (a new store), exactly as it persists mood and episodes.

def td_error(value_cur: float, reward: float, value_next: float = 0.0, gamma: float = 0.9) -> float:
    """Reward-prediction error  delta = r + gamma*V(s') - V(s).
    delta>0 = better than expected (-> elation/relief); delta<0 = worse than expected
    (-> disappointment); delta~0 = as predicted (dopamine stays at baseline). (Schultz 1997)"""
    return reward + gamma * value_next - value_cur


def td_update(value_cur: float, delta: float, alpha: float = 0.3) -> float:
    """Move a cached value toward the outcome:  V(s) <- V(s) + alpha*delta.  (alpha = learning rate)"""
    return value_cur + alpha * delta


def td_step(V: dict, cue: str, reward: float, next_cue: str | None = None,
            alpha: float = 0.3, gamma: float = 0.9) -> float:
    """Ergonomic one-shot over a {cue: value} dict: compute delta for `cue`, learn, return delta.
    Mutates `V` in place. `cue` is a stable context key (task type, file, playbook); choosing it
    well is the real design work -- delta is only as meaningful as the state abstraction. With
    next_cue=None this is a one-step (contextual-bandit) update: delta = reward - V(cue)."""
    value_next = V.get(next_cue, 0.0) if next_cue is not None else 0.0
    delta = td_error(V.get(cue, 0.0), reward, value_next, gamma)
    V[cue] = td_update(V.get(cue, 0.0), delta, alpha)
    return delta


def rpe_affect(delta: float, scale: float = 0.6) -> float:
    """Map a reward-prediction error to a valence nudge in [-1, 1]: pleasant surprise / relief
    (delta>0) vs. let-down / disappointment (delta<0). Blend into appraisal valence, or feed to
    `label_affect` to name relief vs. disappointment vs. elation. Functional teaching signal, not
    felt pleasure: `reward` must be defined operationally (test passed, user approved)."""
    return clamp(math.tanh(scale * delta), -1.0, 1.0)


# ---------------------------------------------------------------- 11. GENERATIVE MODEL  (computed surprise / active inference)
# Replaces hand-fed `novelty` (which the protocol flags as positivity-biased) with COMPUTED Bayesian
# surprise. A tiny categorical generative model -- latent "situations" (states) x observable event
# categories (obs), with Dirichlet counts learned online (Friston 2010 free-energy principle; Itti &
# Baldi 2009 Bayesian surprise). perceive(o) returns:
#   novelty      = 1 - P(o)               Shannon-surprise proxy -> feeds Appraisal.novelty
#   free_energy  = -ln P(o)               surprisal (= variational F at the exact posterior); track its
#                                         fall over time -> valence (Joffily & Coricelli 2013)
#   belief_shift = KL(posterior||prior)   "structural" surprise: how much the situation-belief moved
#                                         -> the awe / insight / schema-revision substrate (P2)
# learn(o, posterior) increments counts so RECURRING events become less surprising (habituation).
# Functional surprise, not felt; states/obs are coarse agent-defined categories -- a toy world model.

@dataclass
class WorldModel:
    states: list          # latent "situations"
    obs: list             # observable event categories
    a: list               # Dirichlet likelihood counts a[o][s]  (len(obs) x len(states))
    d: list               # Dirichlet prior counts over states   (len(states))


def world_from(states, obs, prior: float = 1.0) -> WorldModel:
    """A fresh, uniform world model: every Dirichlet count = `prior` (flat beliefs)."""
    return WorldModel(states=list(states), obs=list(obs),
                      a=[[prior] * len(states) for _ in obs], d=[prior] * len(states))


def _likelihood(a):
    """A[o][s] = P(o|s): normalize each STATE column over observations."""
    n_obs, n_st = len(a), (len(a[0]) if a else 0)
    cols = [sum(a[o][s] for o in range(n_obs)) for s in range(n_st)]
    return [[(a[o][s] / cols[s] if cols[s] > 0 else 0.0) for s in range(n_st)] for o in range(n_obs)]


def perceive(wm: WorldModel, o) -> dict:
    """
    Bayesian perception of observing event category `o` (name or index). Posterior over situations
    Q[s] proportional to P(o|s)*D[s]. Returns computed `novelty` (1-P(o)), `free_energy` (-ln P(o)),
    `belief_shift` (KL(Q||D)), and the `posterior` Q. Use `novelty` for the Appraisal; pass
    `posterior` to learn(). With the exact posterior, free_energy equals the surprisal -ln P(o).
    """
    oi = wm.obs.index(o) if isinstance(o, str) else o
    A = _likelihood(wm.a)
    z = sum(wm.d)
    D = [di / z for di in wm.d] if z > 0 else [1.0 / len(wm.d)] * len(wm.d)
    joint = [A[oi][s] * D[s] for s in range(len(wm.states))]
    p_o = sum(joint)
    Q = [j / p_o for j in joint] if p_o > 0 else D[:]
    novelty = clamp(1.0 - p_o)
    free_energy = -math.log(max(p_o, 1e-12))
    belief_shift = sum(q * math.log(q / D[s]) for s, q in enumerate(Q) if q > 0 and D[s] > 0)
    return {"novelty": novelty, "free_energy": free_energy, "belief_shift": belief_shift, "posterior": Q}


def learn(wm: WorldModel, o, posterior, lr: float = 1.0) -> None:
    """Dirichlet update (mutates `wm`): a[o][s] += lr*Q[s], d[s] += lr*Q[s]. Recurring events habituate."""
    oi = wm.obs.index(o) if isinstance(o, str) else o
    for s in range(len(wm.states)):
        wm.a[oi][s] += lr * posterior[s]
        wm.d[s] += lr * posterior[s]


def valence_from_free_energy(f_prev: float, f_now: float, scale: float = 1.0) -> float:
    """Valence tracks the *rate of change* of free energy (Joffily & Coricelli 2013): falling F
    (uncertainty resolving / world better than modeled) -> positive; rising F -> negative. Returns a
    valence nudge in [-1, 1]. NOTE: this is a first-order, single-step, UNWEIGHTED proxy for their
    precision-weighted dF/dt; their second-order term (the relief-vs-hope substrate) is not modeled.
    The payoff of computing F; pairs with update_mood (the P1.2 hook)."""
    return clamp(math.tanh(scale * (f_prev - f_now)), -1.0, 1.0)


# ---------------------------------------------------------------- 12. GLOBAL WORKSPACE  (functional access, GWT/GNW)
# Gives the engine an autonomous select -> ignite -> broadcast cycle: the stores' outputs this turn
# (retrieved memories u the current event u active intentions -- treat the stores as the parallel
# "modules") compete for ONE limited-capacity workspace; the winner IGNITES (bistable, all-or-none)
# and is broadcast to every store (Baars 1988 Global Workspace; Dehaene-Changeux 1998 Global Neuronal
# Workspace ignition). INSPIRED BY -- not faithful to -- the Conscious Turing Machine (Blum & Blum
# 2022): there `salience` plays the role of CTM chunk WEIGHT (which alone drives the competition);
# the mood-congruence term is an added affective bias NOT in CTM, and CTM derives intensity/mood FROM
# weight as read-outs, not the reverse. This satisfies Butlin et al. (2023) indicators GWT-2 (limited-
# capacity bottleneck + selective attention) and GWT-3 (global broadcast) as ARCHITECTURAL / functional
# access; GWT-1 holds only if the stores count as the parallel modules whose outputs compete, and
# GWT-4 (top-down attention) is only partial (focus carried to next turn). Indicators are necessary-
# not-sufficient and assume computational functionalism, which is contested (Butlin et al. are agnostic).
# It is NOT phenomenal consciousness: the engine is not "aware" of anything and feels nothing. The
# mood/intensity terms are formal signals, not emotions.

def ignite(drive: float, theta: float = 0.55, beta: float = 8.0, kappa: float = 8.0,
           steps: int = 20, r0: float = 0.0) -> float:
    """
    Dehaene-Changeux all-or-none ignition as a bistable recurrent update:
      r <- sigmoid(beta*(r - 0.5) + kappa*(drive - theta)).
    beta>4 makes the map bistable (its slope at r=0.5 is beta/4 > 1). `theta` is a drive OFFSET, not
    the access threshold: with the default cold start r0=0 (what workspace_compete uses), the effective
    all-or-none boundary sits near drive ~= 0.69 (above theta). Below it the content decays toward ~0
    (stays local / "unconscious"); above it recurrence amplifies r toward ~1 (it "ignites"). The curve
    is SHARP, not graded. In the hysteresis band (~0.45-0.65) BOTH basins exist and r0 selects which --
    workspace_compete intentionally cold-starts at r0=0 so band content stays local.
    """
    r = r0
    for _ in range(steps):
        r = sigmoid(beta * (r - 0.5) + kappa * (drive - theta))
    return r


def workspace_compete(candidates, mood: Affect, w=(0.5, 0.2, 0.3), temp: float = 0.3,
                      theta: float = 0.55, beta: float = 8.0) -> dict:
    """
    One Global-Workspace cycle. `candidates` is a list of dicts, each with `salience` (intensity),
    `valence` (its affect valence, for mood-congruence -- reusing retrieval_score's Bower term), and
    `query_relevance`. The competition drive (CTM-style, folding in intensity + mood) is
      f = w0*salience + w1*(1 - |valence - mood.valence|/2) + w2*query_relevance.
    The strongest candidate IGNITES iff ignite(f_win) > 0.5; on ignition it becomes the broadcast
    `focus` (the agent should then bump it across stores and call update_mood with its affect).
    Returns {focus, ignited, r, p}: `p` is the softmax competition distribution (keyed by each
    candidate's `id` -- ids should be UNIQUE, or duplicate entries collapse and `p` loses mass),
    `focus` the winner (or None if nothing crossed threshold -- sub-threshold content stays local this
    turn). Functional access, never awareness.
    """
    if not candidates:
        return {"focus": None, "ignited": False, "r": 0.0, "p": {}}

    def drive(c):
        moodcong = clamp(1.0 - abs(c.get("valence", 0.0) - mood.valence) / 2.0)
        return w[0] * c.get("salience", 0.0) + w[1] * moodcong + w[2] * c.get("query_relevance", 0.0)

    fs = [drive(c) for c in candidates]
    hi = max(fs)
    t = max(temp, 1e-6)                                     # floor temp (mirrors affective_choice) -> no /0
    exps = [math.exp((f - hi) / t) for f in fs]             # softmax (max-shifted for stability)
    z = sum(exps)
    p = {c.get("id", i): e / z for i, (c, e) in enumerate(zip(candidates, exps))}
    wi = max(range(len(candidates)), key=lambda i: fs[i])
    r = ignite(fs[wi], theta=theta, beta=beta)
    ignited = r > 0.5
    return {"focus": candidates[wi] if ignited else None, "ignited": ignited, "r": r, "p": p}


def top_down_bias(focus_features: dict, candidate_features: dict, gain: float = 0.5) -> float:
    """GWT-4 TOP-DOWN LOOP: the content currently broadcast in the workspace feeds BACK to bias what is
    processed next -- a candidate sharing features with the present focus is primed (Dehaene's global
    workspace is RECURRENT, not feed-forward; the broadcast influences subsequent processing). Returns a
    multiplicative salience/retrieval boost in [1, 1+gain] from the cosine overlap of the broadcast focus and
    a candidate. This closes the recurrent loop that left GWT-4 only partial. Functional top-down priming,
    not felt attention."""
    return 1.0 + clamp(gain, 0.0, 1.0) * _cosine(focus_features, candidate_features)


# ---------------------------------------------------------------- 13. METACOGNITION  (confidence, source, calibration)
# A second-order layer that estimates how much to trust the engine's own judgments -- the
# confabulation guard. Confidence is a computed P(correct) read off first-order decision evidence
# (Fleming & Daw 2017); a reality tag separates externally-grounded from internally-generated content
# (perceptual reality monitoring, Lau 2022); self-efficacy is a slow competence estimate per domain
# (Rouault, Dayan & Fleming 2019); ECE measures whether the confidences mean anything. Together these
# stop low-confidence / "imagined" traces from hardening into semantic "facts" (see consolidation_plan
# min_confidence). Functional confidence, NOT a felt "feeling of knowing"; a HOT-style monitoring
# indicator, not awareness (the HOT->consciousness link is contested).

# PRM reality weights: how externally-grounded a memory's content is (Lau 2022).
SOURCE_REALITY = {"observed": 1.0, "inferred": 0.6, "imagined": 0.2}


def metacog_confidence(evidence: float, rho: float = 0.8, k: float = 2.0) -> float:
    """
    A second-order confidence P(correct) read off first-order decision `evidence` (signed -- e.g. the
    margin / log-odds favoring the judgment): conf = sigmoid(rho*k*evidence). `rho` in (0,1] is
    metacognitive EFFICIENCY (1 = ideal monitor; <1 = noisy/imperfect insight, pulling conf toward the
    0.5 guess line). evidence=0 -> 0.5 (a coin flip); strong support -> ->1; evidence against -> ->0.
    Functional confidence, not a felt feeling of knowing. (Fleming & Daw 2017)
    """
    return clamp(sigmoid(rho * k * evidence))


def reality_weight(source: str) -> float:
    """PRM-style reality weight (Lau 2022): observed (tool output / test result / user statement)
    > inferred (deduced) > imagined (hypothesized / self-generated). Down-weights internally-generated
    content so it is not mistaken for fact. Unknown source -> the cautious `inferred` default."""
    return SOURCE_REALITY.get(source, 0.6)


def update_self_efficacy(se: float, correct: bool, alpha_pos: float = 0.2, alpha_neg: float = 0.4) -> float:
    """
    Domain self-efficacy / competence as a leaky integrator toward the outcome (Rouault, Dayan &
    Fleming 2019): rises on success, falls on failure, learning from FAILURES faster
    (alpha_neg > alpha_pos). Becomes a principled prior for the `control` appraisal axis (-> dominance):
    a competent agent appraises more control. Functional self-belief, not felt confidence.
    """
    a = alpha_pos if correct else alpha_neg
    return clamp(se + a * ((1.0 if correct else 0.0) - se))


def calibration_error(records, bins: int = 10) -> float:
    """
    Expected Calibration Error over logged (confidence, correct) pairs: ECE = sum_b (n_b/N)*|acc_b -
    conf_b|. 0 = perfectly calibrated; high = the confidences are hot air (over/under-confident).
    Tells you whether `metacog_confidence` outputs actually mean anything once outcomes are logged.
    (Naeini, Cooper & Hauskrecht 2015; Guo et al. 2017 -- binned calibration error, distinct from
    Fleming's meta-d'/HMeta-d efficiency, which this project does not compute.)
    """
    if not records:
        return 0.0
    n = len(records)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for (c, ok) in records if (lo < c <= hi) or (b == 0 and c <= hi)]
        if not bucket:
            continue
        conf_b = sum(c for c, _ in bucket) / len(bucket)
        acc_b = sum(1.0 if ok else 0.0 for _, ok in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc_b - conf_b)
    return ece


# ---------------------------------------------------------------- 14. PERSONALITY  (temperament as affective priors)
# Replaces the one hardcoded baseline with a per-agent OCEAN (Big Five) profile, so different agents have
# different resting affect and different reward/threat sensitivities -- a stable extravert vs. an anxious
# introvert literally start from different set-points. OCEAN -> PAD set-point via the ALMA/Mehrabian
# regression (Mehrabian 1996; Gebhard 2005); reward/punishment asymmetry via RST BAS/BIS (Carver & White
# 1994; Gray). Composes WITHOUT new signatures: feed the baseline to update_mood(baseline=...) and the
# (BAS,BIS) gains to the reward/stress args of neuromods_from. Functional dispositions, not felt traits;
# coefficients are population regressions -- tunable, not laws.

@dataclass
class Personality:
    """Big Five (OCEAN) traits in [0,1], 0.5 = population average. Editable per agent."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


def baseline_from_personality(p: Personality) -> Affect:
    """
    OCEAN -> homeostatic PAD set-point (ALMA/Mehrabian regression). Built to MODULATE the engine's calm
    resting baseline, so an all-average profile reproduces the default Affect(0.0, 0.10, 0.50) exactly.
    Note: the +0.19*neuroticism term on pleasure is ALMA's published (counterintuitive) coefficient --
    kept for fidelity, tunable. Feed the result to update_mood(baseline=...).
    """
    o, c, e, a, n = (2*p.openness - 1, 2*p.conscientiousness - 1, 2*p.extraversion - 1,
                     2*p.agreeableness - 1, 2*p.neuroticism - 1)
    pleasure  = clamp(0.21*e + 0.59*a + 0.19*n, -1.0, 1.0)
    arousal   = clamp(0.10 + 0.40*(0.15*o + 0.30*a - 0.57*n))
    dominance = clamp(0.50 + 0.50*(0.25*o + 0.17*c + 0.60*e - 0.32*a))
    return Affect(pleasure, arousal, dominance)


def temperament_gains(p: Personality):
    """
    RST approach/avoidance gains (Carver & White 1994; Gray's BAS/BIS): extraversion raises reward
    sensitivity (BAS), neuroticism raises threat/punishment sensitivity (BIS). Average -> (1.0, 1.0)
    (no change). Returns (bas, bis); multiply the `reward` / `stress` inputs to neuromods_from, e.g.
    reward' = bas * max(valence, 0),  stress' = bis * max(-valence, 0).
    """
    e, n = 2*p.extraversion - 1, 2*p.neuroticism - 1
    bas = clamp(1.0 + 0.5*e - 0.25*n, 0.0, 2.0)
    bis = clamp(1.0 + 0.5*n, 0.0, 2.0)
    return bas, bis


# ---------------------------------------------------------------- 15. INTEROCEPTION  (honest grounding: substrate, not viscera)
# Grounds affect in the agent's OWN real viability variables -- token/compute budget, test-pass rate,
# tool-call success, context headroom, user approval -- regulated toward a set-point. This is
# "interoception" ONLY in Ashby's cybernetic sense (a controller sensing its own essential variables),
# NOT felt bodily sensation and NOT phantom organs (a coding agent has no body). Drive-reduction toward
# the set-point is a GROUNDED reward (Keramati & Gutkin 2014) -- the first reward the engine does not
# hand-feed -- which plugs into the P0.2 value loop. Allostasis (predictive set-point shift) follows
# Sterling / Stephan 2016. Functional homeostasis, never felt hunger or relief.

@dataclass
class Homeostat:
    """The agent's real substrate signals as a body-budget. `levels`/`setpoint`/`weights` are dicts over
    the SAME signal names; each value in [0,1] (1 = healthy/abundant). Honest viability variables only
    (e.g. tokens_left, tests_pass, tool_success, context_free, user_approval) -- never invented viscera."""
    levels: dict
    setpoint: dict
    weights: dict


def drive(h: Homeostat, n: float = 4.0, m: float = 2.0) -> float:
    """Homeostatic drive / deficit -- a CONVEX, weight-NORMALIZED, one-sided penalty on falling below
    set-point:  D(H) = ( sum_i w_i * max(0, H*_i - H_i)^n  /  sum_i w_i ) ^ (1/m).
    One-sided (abundance ABOVE set-point is never a deficit, so allostatic_shift can pre-lower targets
    safely) and normalized so D stays in [0, 1] -- which keeps the grounded reward on the same ~unit
    scale as the value loop. Convex for n > m, so larger deficits hurt super-linearly. This is a
    1/m-ROOT, normalized variant in the SPIRIT of Keramati & Gutkin (2014); their exact form uses outer
    exponent m with n>m>1, whereas the operative convexity condition for THIS form is simply n>m.
    0 = at/above set-point (sated); -> 1 = fully depleted on the weighted signals."""
    w_total = sum(h.weights.get(k, 1.0) for k in h.setpoint) or 1.0
    s = 0.0
    for k, target in h.setpoint.items():
        deficit = max(0.0, target - h.levels.get(k, target))   # one-sided: abundance is not a deficit
        s += h.weights.get(k, 1.0) * deficit ** n
    return (s / w_total) ** (1.0 / m)


def homeostatic_reward(h_prev: Homeostat, h_now: Homeostat, n: float = 4.0, m: float = 2.0) -> float:
    """Grounded reward = drive REDUCTION: r = D(H_prev) - D(H_now) (Keramati & Gutkin 2014). Moving the
    body-budget toward set-point is rewarding; depleting it is punishing. Add it into the reward signal so
    affect/learning are grounded in real viability, not a hand-fed label: `runtime.py` folds it into the
    neuromodulator reward (so it colours mood); you may also feed it as the `reward` arg of the value loop
    (td_step) to ground V-learning. Since D is normalized to [0,1], this reward is in [-1,1] -- on the same
    scale as the value loop's unit rewards. Functional drive-reduction, not felt relief."""
    return drive(h_prev, n, m) - drive(h_now, n, m)


def body_affect(h: Homeostat, n: float = 4.0, m: float = 2.0) -> dict:
    """Endogenous affect from the body-budget: {stress = clamp(D(H)) -> cortisol; v_body = -clamp(D(H))
    -> blend into appraisal valence as valence = a*appraisal_v + (1-a)*v_body; drive = raw D(H)}.
    Lets internal state (low budget, failing tests) color affect on its own. Functional, not felt."""
    d = drive(h, n, m)
    return {"stress": clamp(d), "v_body": -clamp(d), "drive": d}


def allostatic_shift(setpoint: dict, demand: dict, rate: float = 0.5) -> dict:
    """Allostasis = stability through anticipatory change (Sterling; Stephan 2016): pre-lower the target
    for a resource you expect to spend (`demand[k]` in [0,1]) so planned spending is not flagged as an
    emergency deficit. Returns a new set-point dict. Functional predictive budgeting, not felt anticipation."""
    return {k: clamp(v - rate * demand.get(k, 0.0)) for k, v in setpoint.items()}


def body_tick(h: Homeostat, *, effort: float = 0.02, success=None, reward: float = 0.0,
              recover: float = 0.0) -> Homeostat:
    """Advance the body-budget one step so interoception is actually DRIVEN by living. Spend `effort` on
    the cognitive load of an event (depletes tokens/compute/context_free); an outcome moves the
    performance signals (success lifts tests_pass/tool_success, failure drops them); `reward` lifts
    user_approval; and `recover` (rest - e.g. at sleep) pulls every level back toward set-point. Returns a
    NEW Homeostat (does not mutate). Feed the drive-reduction (homeostatic_reward) into the value loop and
    body_affect's stress into cortisol, so mood gains a bodily grounding. Cybernetic, not felt fatigue."""
    lv = dict(h.levels)

    def adj(k, d):
        if k in lv:
            lv[k] = clamp(lv[k] + d)
    for k in ("tokens", "compute", "context_free"):
        adj(k, -effort)
    if success is True:
        adj("tests_pass", 0.05); adj("tool_success", 0.05)
    elif success is False:
        adj("tests_pass", -0.08); adj("tool_success", -0.08)
    adj("user_approval", 0.05 * reward)
    if recover > 0.0:
        for k, target in h.setpoint.items():
            if k in lv:
                lv[k] = clamp(lv[k] + recover * (target - lv[k]))
    return Homeostat(lv, dict(h.setpoint), dict(h.weights))


# ---------------------------------------------------------------- 16. COPING & ACTION  (what the feeling DOES)
# Closes the loop: affect doesn't just get recorded, it BIASES action. Frijda's action readiness maps a
# state to an urge class; EMA coping picks problem- vs emotion-focused strategies by control; a
# neuromodulated softmax temperature trades off explore vs exploit; a somatic marker biases choices by
# how similar past episodes turned out. All BEHAVIORAL policy, not felt urges -- and emotion-focused
# coping RE-PRIORITIZES attention/goals, it must never deny facts or override correctness.

def action_tendency(a: Affect, ap: Appraisal) -> dict:
    """Frijda (1986) action readiness: the affective state biases WHICH class of action is urged,
    scaled by arousal (control precedence -- intense states demand action). Returns weights over
    {approach, avoid, attack, attend}. A behavioral policy bias, not a felt urge. (Defensive-mode
    freeze/flight/fight specialization is P2.2.)"""
    neg, pos, urge = max(-a.valence, 0.0), max(a.valence, 0.0), a.arousal
    return {
        "approach": urge * clamp(pos),                       # good -> engage / exploit
        "avoid":    urge * clamp(neg * (1.0 - ap.control)),  # bad + low control -> withdraw (fear)
        "attack":   urge * clamp(neg * ap.control),          # bad + high control -> confront (anger)
        "attend":   urge * clamp(ap.novelty),                # novel -> orient / investigate
    }


def select_coping(ap: Appraisal) -> dict:
    """EMA coping (Marsella & Gratch 2009), keyed on the `control` axis: high control -> PROBLEM-focused
    (act on the world); low control -> EMOTION-focused (act on one's own appraisal/attention). Returns
    {mode, strategies}.
    GUARDRAIL: emotion-focused strategies RE-PRIORITIZE attention/goals -- they must NEVER deny facts or
    override correctness (the self-deception risk, flagged even within EMA)."""
    if ap.control >= 0.5:
        return {"mode": "problem-focused", "strategies": ["replan", "try_alternative", "seek_info"]}
    return {"mode": "emotion-focused", "strategies": ["reframe_goal", "lower_goal_relevance", "defer", "ask_for_help"]}


def exploration_temperature(nm: Neuromods, tau0: float = 0.5,
                            k_ne: float = 0.4, k_cort: float = 0.4, k_da: float = 0.5) -> float:
    """Affect-modulated explore/exploit temperature for action selection (Doya 2002):
      tau = tau0 * exp(-k_ne*ne - k_cort*cortisol + k_da*da).
    Stress (phasic noradrenaline, cortisol) lowers tau -> EXPLOIT (commit to the best known option,
    tunnel vision); dopamine (reward optimism) raises tau -> EXPLORE. Lower tau = greedier.
    NOTE: the sign choices are a modeling decision -- the NE/exploration link in particular is mixed
    (tonic NE may instead promote exploration; Aston-Jones & Cohen 2005), so treat as tunable."""
    return tau0 * math.exp(-k_ne * nm.ne - k_cort * nm.cortisol + k_da * nm.da)


def affective_choice(scores: dict, temperature: float) -> dict:
    """Softmax action selection at a given explore/exploit `temperature` (from exploration_temperature):
    P(a) proportional to exp(score(a)/tau). Low tau -> greedy (exploit the best); high tau -> spread
    (explore). `scores` maps option -> its affect-adjusted value (base utility + valence / somatic /
    tendency bonuses). Returns a probability dict over the options."""
    if not scores:
        return {}
    hi = max(scores.values())
    t = max(temperature, 1e-6)
    exps = {k: math.exp((v - hi) / t) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: e / z for k, e in exps.items()}


def somatic_marker(similar_valences) -> float:
    """Damasio's somatic-marker / as-if loop (1994): a gut-feel bias toward/away from an option = the
    mean affective valence of SIMILAR past episodes (retrieve them, pass their affect valences). Positive
    -> this kind of choice tended to go well; negative -> it burned us before. Add it as a bonus to the
    option's score. Functional bias, not a felt hunch. Empty -> 0 (no prior)."""
    return sum(similar_valences) / len(similar_valences) if similar_valences else 0.0


# ---------------------------------------------------------------- 17. AFFECT DYNAMICS  (DynAffect / OU attractor, dual time-scale)
# Promotes the single leaky integrator to a proper core-affect ATTRACTOR with two time-scales: a FAST
# "emotion" (half-life ~minutes) and a SLOW "mood" (half-life ~hours), each an Ornstein-Uhlenbeck process
# pulled toward the personality baseline (Kuppens, Oravecz & Tuerlinckx 2010 DynAffect; ALMA pull/over-
# shoot, Gebhard 2005). `update_mood` (section 7) remains the simple drift building block; these add
# real-time decay, optional SEEDED variability, and the ALMA over-shoot. Functional dynamics, not felt
# moods. Noise is opt-in (sigma=0 -> fully deterministic) and seeded so runs stay reproducible.

def ou_affect_step(state: Affect, event_affect: Affect, baseline: Affect | None = None, dt: float = 1.0,
                   t_half: float = 3600.0, beta: float = 0.05, sigma: float = 0.0, seed=None,
                   kick_threshold: float = 1.2, kick: float = 0.5) -> Affect:
    """
    One Ornstein-Uhlenbeck / DynAffect step for ONE affect time-scale. Real-time decay
    `alpha = 2^(-dt/t_half)` (short t_half -> fast 'emotion'; long -> slow 'mood'):
        x = x*alpha + event*(1-alpha)          # time-scaled absorption of the event
        x += beta*(baseline - x)               # OU drift toward the homeostatic set-point (homeostasis)
        if ||event|| > kick_threshold:  x += kick*(event - x)   # ALMA over-shoot for intense events
        x += N(0, sigma)                       # DynAffect variability -- OPT-IN, SEEDED (reproducible)
    `baseline` should come from `baseline_from_personality(p)`. sigma=0 (default) is deterministic.
    Functional dynamics, not a felt mood. Returns a new clamped Affect.
    """
    b = baseline or Affect(0.0, 0.10, 0.50)
    alpha = 2.0 ** (-dt / t_half) if t_half > 0 else 0.0
    v = state.valence * alpha + event_affect.valence * (1.0 - alpha)
    ar = state.arousal * alpha + event_affect.arousal * (1.0 - alpha)
    do = state.dominance * alpha + event_affect.dominance * (1.0 - alpha)
    v += beta * (b.valence - v); ar += beta * (b.arousal - ar); do += beta * (b.dominance - do)
    ev = _pad_point(event_affect)
    if math.sqrt(sum(x * x for x in ev)) > kick_threshold:   # intense event -> ALMA over-shoot
        k = kick * (1.0 - alpha)                             # time-scaled: a fast-channel phenomenon --
        v += k * (event_affect.valence - v)                  # the slow mood is not yanked by one jolt
        ar += k * (event_affect.arousal - ar)
        do += k * (event_affect.dominance - do)
    if sigma > 0.0:
        rng = random.Random(seed)
        v += rng.gauss(0.0, sigma); ar += rng.gauss(0.0, sigma); do += rng.gauss(0.0, sigma)
    return Affect(clamp(v, -1.0, 1.0), clamp(ar), clamp(do))


def update_affect(emotion: Affect, mood: Affect, event_affect: Affect, baseline: Affect | None = None,
                  dt: float = 1.0, t_half_emotion: float = 1200.0, t_half_mood: float = 43200.0,
                  beta: float = 0.05, sigma: float = 0.0, seed=None):
    """
    Dual time-scale affect (DynAffect): a FAST `emotion` (t_half ~20 min) and a SLOW `mood` (t_half ~12 h),
    each an OU attractor toward `baseline`. Returns (new_emotion, new_mood). Retrieval/workspace read the
    slow `mood`; the fast `emotion` is the in-the-moment state. A sharp event swings emotion immediately
    while mood only drifts -- so one bad event does not durably darken the agent, but a run of them does.
    """
    em = ou_affect_step(emotion, event_affect, baseline, dt, t_half_emotion, beta, sigma, seed)
    mo = ou_affect_step(mood, event_affect, baseline, dt, t_half_mood, beta, sigma,
                        (seed + 1 if seed is not None else None))
    return em, mo


# ---------------------------------------------------------------- 18. NEUROMODULATOR DYNAMICS  (5-HT, OT, HPA, Yerkes-Dodson)
# Replaces the static gains with real(er) dynamics. Serotonin = average-reward / patience, an opponent
# to phasic dopamine, and it SETS the temporal discount (Daw, Kakade & Dayan 2002; Doya 2002). The
# Yerkes-Dodson inverted-U makes arousal MATTER: performance peaks at moderate arousal and collapses
# when over-aroused -- the substrate that turns "high arousal" into "terror" (Aston-Jones & Cohen 2005;
# Gilzenrat 2002). The HPA axis gives cortisol a stateful negative-feedback cascade (Vinther et al.
# 2011) -- it ramps and lingers (allostatic load / burnout) instead of being instantaneous. Oxytocin is
# a prosocial reward weighting (Lockwood 2022). Well-cited PROPOSALS with partial support, not settled
# biology; parameters are illustrative; "trust"/"patience" are behavioral weightings, not felt.

def serotonin_level(avg_reward: float, c: float = 4.0) -> float:
    """5-HT from recent AVERAGE reward (Daw, Kakade & Dayan 2002): a well-fed agent (high average
    reward) runs high serotonin -> patience; scarcity -> low 5-HT -> impulsivity. serotonin =
    sigmoid(c*avg_reward), avg_reward a running mean in ~[-1,1]. Opponent to phasic dopamine."""
    return clamp(sigmoid(c * avg_reward))


def discount_from_serotonin(serotonin: float, lo: float = 0.5, hi: float = 0.99) -> float:
    """Map 5-HT to the temporal discount gamma for the value loop (Doya 2002: serotonin sets the
    discount): more patience -> higher gamma -> the agent values future reward more. Feed into td_step."""
    return clamp(lo + (hi - lo) * clamp(serotonin), 0.0, 0.99)


def performance(arousal: float, opt: float = 0.5, width: float = 0.25) -> float:
    """Yerkes-Dodson inverted-U (Aston-Jones & Cohen 2005; Gilzenrat 2002): decision quality peaks at
    MODERATE arousal and degrades when too low (drowsy) OR too high (panic/choke):
        perf = exp(-(arousal-opt)^2 / (2*width^2))  in (0,1].
    This is what makes over-arousal MEAN something -- terror is high arousal that wrecks performance,
    not merely a large number. Multiply decision/recall quality by perf."""
    return math.exp(-((arousal - opt) ** 2) / (2.0 * width ** 2))


def lc_gain(ne_tonic: float, k: float = 2.0) -> float:
    """Locus-coeruleus adaptive gain (Aston-Jones & Cohen 2005): tonic noradrenaline multiplies the gain
    on decision/recall sigmoids -> 1 + k*ne_tonic (sharper, more contrastive responses). A modeling
    choice; the tonic-vs-phasic NE story is debated."""
    return 1.0 + k * clamp(ne_tonic)


def oxytocin_gain(trust: float, base: float = 0.0, omega: float = 1.0) -> float:
    """Oxytocin as a prosocial reward weighting (Lockwood 2022): up-weights reward / prediction-error
    from a trusted partner (e.g. the user). oxt = clamp(base + omega*trust); scale social-aligned reward
    by (1 + oxt). A behavioral weighting, not felt warmth."""
    return clamp(base + omega * trust)


@dataclass
class Hpa:
    """HPA-axis state -- the slow stress-hormone cascade (hypothalamus -> pituitary -> adrenal)."""
    crh: float = 0.0       # corticotropin-releasing hormone (hypothalamus)
    acth: float = 0.0      # adrenocorticotropic hormone (pituitary)
    cortisol: float = 0.1  # glucocorticoid (adrenal) -- the slow stress signal; mirror into Neuromods.cortisol


def hpa_step(h: Hpa, stress: float, dt: float = 1.0, kf: float = 0.5, n: float = 2.0,
             k_acth: float = 0.5, k_cort: float = 0.5,
             decay_crh: float = 0.2, decay_acth: float = 0.2, decay_cort: float = 0.2) -> Hpa:
    """One step of the HPA negative-feedback cascade (Vinther, Andersen & Ottesen 2011):
        CRH'  = stress/(1+(C/Kf)^n) - decay*CRH        # cortisol INHIBITS CRH (negative feedback)
        ACTH' = k_acth*CRH - decay*ACTH
        C'    = k_cort*ACTH - decay*C
    So cortisol RAMPS under sustained stress and DECAYS when it lifts, and stays elevated a while
    afterward -- the substrate for allostatic load / burnout, unlike the old instantaneous clamp(stress).
    Set Neuromods.cortisol = the returned `cortisol`. Illustrative parameters, not calibrated biology."""
    crh = h.crh + dt * (clamp(stress) / (1.0 + (h.cortisol / kf) ** n) - decay_crh * h.crh)
    acth = h.acth + dt * (k_acth * crh - decay_acth * h.acth)
    cortisol = h.cortisol + dt * (k_cort * acth - decay_cort * h.cortisol)
    return Hpa(clamp(crh, 0.0, 10.0), clamp(acth, 0.0, 10.0), clamp(cortisol, 0.0, 1.0))


def hpa_recover(h: Hpa, factor: float = 0.35) -> Hpa:
    """Sleep / slow-wave recovery of the HPA axis. Deep sleep actively SUPPRESSES the stress axis
    (cortisol has its nadir in early sleep; Born & Fehm 1998), discharging the slow CRH/ACTH 'load' that
    would otherwise keep cortisol pinned high long after the threat is gone. Scales the whole cascade
    toward baseline by `factor` - so a good night's rest brings an agent down off allostatic load, while a
    single calm moment (one hpa_step) cannot. Returns a new Hpa. Functional recovery, not felt rest."""
    return Hpa(clamp(h.crh * factor, 0.0, 10.0), clamp(h.acth * factor, 0.0, 10.0),
               clamp(h.cortisol * factor, 0.0, 1.0))


# ---------------------------------------------------------------- 19. NAMED-FEELING CIRCUITS  (terror, awe, panic)
# The owner's named feelings as MECHANISMS, not just PAD labels (§9). Each is a small circuit that
# composes earlier layers. The defensive modes run on TWO axes (urgency x control): freeze / flight /
# fight / tonic_immobility; terror is the cornered, uncontrollable, performance-collapsed corner (it
# co-occurs with tonic immobility, NOT agentic fight) (LeDoux; Tovote, Fadok & Luthi 2015; Mobbs 2007
# imminence; uses §18 Yerkes-Dodson). awe = vastness + need-for-accommodation (a big schema update), reusing §11's structural
# surprise, and it shrinks a self-salience weight ("small self") (Keltner & Haidt 2003). panic = a
# SEPARATE circuit (separation-distress / loss / interoceptive alarm), dampened by oxytocin, NOT a
# flavour of fear (Panksepp's PANIC system). All are FUNCTIONAL labels for regions of computed affect +
# behavioural modes -- not felt terror, wonder, or distress; "small self" is a parameter down-weight,
# not self-transcendence.

def defensive_mode(ap: Appraisal, affect: Affect, nm: Neuromods) -> dict:
    """
    Defensive response along TWO axes (Mobbs 2007 threat-imminence; Tovote, Fadok & Luthi 2015 PAG
    modes; LeDoux amygdala->PAG): `urgency = 0.5*goal_relevance + 0.5*arousal` (how pressing the threat
    is) and `control` (agency to act on it). Low urgency -> FREEZE (attentive immobility, assess). WITH
    agency (control >= 0.5): FLIGHT if moderate, FIGHT if urgent. WITHOUT agency (low control) under an
    urgent threat: TONIC_IMMOBILITY -- the cornered, no-escape, "frozen in terror" corner. `terror` =
    a real threat (negative affect) that is urgent, UNCONTROLLABLE, and has collapsed Yerkes-Dodson
    `performance` (§18, high arousal that now HURTS) -- so it co-occurs with tonic_immobility, NOT with
    an agentic fight. Returns {urgency, mode, terror, perf}. Functional state + behavioural mode, not
    felt terror.
    """
    urgency = clamp(0.5 * ap.goal_relevance + 0.5 * affect.arousal)
    if urgency < 0.4:
        mode = "freeze"                                          # distal/low: attentive immobility
    elif ap.control >= 0.5:
        mode = "fight" if urgency > 0.7 else "flight"            # agency -> confront or escape
    else:
        mode = "tonic_immobility" if urgency > 0.7 else "flight"  # no agency + cornered -> frozen
    perf = performance(affect.arousal)
    terror = (affect.valence < -0.3 and urgency > 0.7 and ap.control < 0.25 and perf < 0.45)
    return {"urgency": urgency, "mode": mode, "terror": terror, "perf": perf}


def awe(vastness: float, belief_shift: float, valence: float,
        a: float = 2.0, b: float = 2.0, c: float = 2.0, d: float = 0.6) -> dict:
    """
    Awe = perceived VASTNESS + need-for-ACCOMMODATION (a large schema update) [Keltner & Haidt 2003].
    `vastness` in [0,1] (scope/magnitude of the thing); `belief_shift` = the structural surprise from
    perceive() (§11, KL(posterior||prior)). intensity = sigmoid(a*vastness + b*belief_shift - c). Awe
    down-weights a self-salience parameter ("small self" -- a FUNCTIONAL deactivation, NOT
    self-transcendence). `flavor` splits dread-awe (threat-tinged) from wonder-awe by valence.
    Returns {awe, self_weight, flavor}.
    """
    intensity = clamp(sigmoid(a * vastness + b * clamp(belief_shift, 0.0, 5.0) - c))
    return {"awe": intensity, "self_weight": clamp(1.0 - d * intensity),
            "flavor": "dread-awe" if valence < 0 else "wonder-awe"}


def panic(separation: float, intero_alarm: float = 0.0, oxytocin: float = 0.0,
          w_sep: float = 0.7, w_int: float = 0.4, w_oxt: float = 0.5) -> float:
    """
    Panic / separation-distress as a SEPARATE circuit from fear (Panksepp's PANIC/GRIEF system):
    triggered by loss / separation / interoceptive alarm (e.g. resource depletion -> body_affect drive,
    §15), and DAMPENED by oxytocin (social support / safety, §18). panic = clamp(w_sep*separation +
    w_int*intero_alarm - w_oxt*oxytocin). A second route to a terror-like state that is about LOSS, not
    threat. Functional distress signal, not felt panic.
    """
    return clamp(w_sep * clamp(separation) + w_int * clamp(intero_alarm) - w_oxt * clamp(oxytocin))


# ---------------------------------------------------------------- 20. SLEEP DYNAMICS  (replay, SHY, REM depotentiation, reflection)
# Enriches the /sleep cycle beyond the simple promote/forget threshold. REM depotentiation fades the
# emotional CHARGE of a memory while keeping the FACT -- "sleep to forget the emotion, remember the
# event" (van der Helm & Walker 2011): the owner's "don't stay traumatized" goal. Prioritized replay
# spends the limited consolidation budget on Need x Gain (Mattar & Daw 2018). SHY downscaling renormalizes
# total strength each night (Tononi & Cirelli 2020). Reflection synthesizes high-salience clusters into
# semantic facts (Park 2023). Episodic stays APPEND-ONLY: depotentiation reduces the charge a memory
# CARRIES FORWARD (into mood / re-consolidation), it does not rewrite the historical log line.

def rem_depotentiate(valence: float, arousal: float, rho: float = 0.5, ne_rem: float = 0.0):
    """REM sleep depotentiates the emotional CHARGE of a memory while leaving its content/salience intact
    (van der Helm & Walker 2011: "sleep to forget the emotion, remember the event"). REM runs at LOW
    noradrenaline (ne_rem ~ 0); depotentiation strength = rho*(1-ne_rem)*arousal -- so it preferentially
    softens HIGH-arousal (distressing) memories. Returns the faded (valence, arousal) charge to carry
    forward; the FACT persists. OPT-IN and tunable -- the REM-depotentiation evidence is mixed."""
    fade = clamp(rho * (1.0 - clamp(ne_rem)) * clamp(arousal))
    return (valence * (1.0 - fade), arousal * (1.0 - fade))


def replay_priority(episode: dict, now: float) -> float:
    """Prioritized replay = Need x Gain (Mattar & Daw 2018): replay first the episodes both LIKELY TO BE
    NEEDED (Need ~ sigmoid(base_level_activation): recency+frequency) and high in LEARNING VALUE
    (Gain ~ salience / surprise). Spends the limited /sleep replay budget where it matters most."""
    need = sigmoid(base_level_activation(episode.get("retrievals", [episode["t0"]]), now))
    gain = episode.get("salience", 0.0)
    return need * gain


def shy_downscale(strengths, target: float | None = None, protect=()):
    """Synaptic Homeostasis Hypothesis (Tononi & Cirelli 2020): sleep DOWNSCALES strengths proportionally
    so the total returns toward a target -- net renormalization that improves signal-to-noise and curbs
    runaway potentiation, while RELATIVE order (and `protect`-ed / replayed traces) survives. Only scales
    down (never up). target defaults to the count (mean -> 1.0). Returns the rescaled list."""
    total = sum(strengths) or 1.0
    tgt = float(len(strengths)) if target is None else target
    scale = clamp(tgt / total, 0.0, 1.0)
    return [s if i in protect else s * scale for i, s in enumerate(strengths)]


def reflection_trigger(recent_saliences, theta: float = 3.0) -> bool:
    """Generative-Agents-style reflection (Park 2023): when accumulated recent salience crosses a
    threshold, pause to SYNTHESIZE a higher-level semantic fact from the cluster (citing source episodes)
    -- the one consolidation idea leading agent-memory systems have and a bare threshold lacks. Returns
    True when sum(recent_saliences) >= theta; the synthesis itself is an agent step at /sleep."""
    return sum(recent_saliences) >= theta


# ---------------------------------------------------------------- 21. EVALUATION HARNESS  (make it falsifiable)
# Metrics that turn the whole edifice from "plausible" into "testable". Without them every layer above
# is unfalsifiable. Calibration (Brier; ECE = calibration_error §13) and metacognitive sensitivity
# (the non-parametric type-2 AUROC; Fleming & Lau 2014 -- NOT the parametric meta-d') ask whether
# `confidence` MEANS anything.
# label_stability tests the §9 discrete-emotion layer's robustness (and surfaces the low-radius ambiguity
# §9 warns about). recall_accuracy (LoCoMo/LongMemEval-style F1) scores retrieval. grounding_self_test
# encodes the honest boundary as an executable check: the affect/cognitive band is groundable, the
# felt-body band is not (Xu, Bi et al. 2025). These are run from test_brain.py + documented in docs/eval.md.

def brier_score(records) -> float:
    """Brier score over (confidence, correct) pairs: mean((conf - outcome)^2), outcome in {0,1}. 0 =
    perfect, 0.25 = always-0.5 guessing, 1 = confidently wrong. Complements ECE (calibration_error)."""
    if not records:
        return 0.0
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in records) / len(records)


def metacog_sensitivity(records) -> float:
    """Metacognitive sensitivity = the NON-PARAMETRIC type-2 AUROC (area under the type-2 ROC; Fleming &
    Lau 2014): the probability that a CORRECT trial was held with higher confidence than an INCORRECT one
    (Mann-Whitney over the logged (confidence, correct) pairs). 0.5 = confidence is uninformative (no
    insight); 1.0 = perfect discrimination; <0.5 = anti-correlated. Needs both correct and incorrect
    trials (else 0.5). NOTE: this is NOT the parametric SDT meta-d' (which was designed to remove
    AUROC2's type-1-sensitivity/bias confound) -- a simpler sibling, not the same measure."""
    corr = [c for c, ok in records if ok]
    inc = [c for c, ok in records if not ok]
    if not corr or not inc:
        return 0.5
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in corr for b in inc)
    return wins / (len(corr) * len(inc))


def label_stability(a: Affect, delta: float = 0.05) -> float:
    """Robustness of the §9 discrete-emotion label under small jitter: perturb each PAD axis by
    +/-delta and report the fraction of the 3^3 grid that keeps the SAME label as the centre. ->1 deep
    inside a prototype's region (trustworthy label); low near a boundary or the neutral origin (where
    §9 warns the argmax is weakly determined). A falsifiable check on the labeling layer."""
    base = label_affect(a)["label"]
    same = total = 0
    for dv in (-delta, 0.0, delta):
        for da in (-delta, 0.0, delta):
            for dd in (-delta, 0.0, delta):
                p = Affect(clamp(a.valence + dv, -1.0, 1.0), clamp(a.arousal + da), clamp(a.dominance + dd))
                total += 1
                same += 1 if label_affect(p)["label"] == base else 0
    return same / total


def recall_accuracy(retrieved, relevant) -> dict:
    """LoCoMo / LongMemEval-style retrieval quality: precision, recall, F1 of the retrieved memory ids
    against the ground-truth relevant set. Lets the agent self-score whether retrieval_score surfaced
    the right memories. Returns {precision, recall, f1}."""
    R, G = set(retrieved), set(relevant)
    if not R and not G:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(R & G)
    precision = tp / len(R) if R else 0.0
    recall = tp / len(G) if G else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grounding_self_test() -> dict:
    """The honest grounding BOUNDARY as an executable transparency check (Xu, Bi et al. 2025: an agent
    can recover the AFFECT band of word norms but not the SENSORIMOTOR band). Declares which signal
    bands this engine legitimately grounds vs. which are permanently out of reach for a disembodied
    coding agent. NOTE: this returns a fixed declaration; the test (`test_grounding_self_test_...`) is a
    REGRESSION GUARD on that declaration (it keeps the felt/phenomenal band out of `groundable`), not a
    behavioral check of engine outputs -- the no-bare-"feels" framing of outputs is a prose convention
    (§9: "a fear-LIKE state, never 'feels fear'"), enforced by the protocol, not by this function."""
    return {
        "groundable": ["valence", "arousal", "dominance", "salience", "confidence",
                       "drive (real substrate: tokens/compute/tests/tool/context/approval)"],
        "not_groundable": ["felt bodily sensation", "phenomenal experience", "qualia",
                           "the sensorimotor feeling of an organic body"],
        "claim": "functional / computational correlates only; the felt band is permanently out of reach",
    }


# ---------------------------------------------------------------- 22. CONSCIOUSNESS INDICATORS  (transparency, NOT a test)
# The project's central honesty mechanism. Instead of any sentience claim, it reports which of the
# Butlin, Long, Bengio, Chalmers et al. (2023) INDICATOR PROPERTIES (derived from RPT / GWT / HOT / AST /
# PP / Agency-Embodiment) the *architecture* satisfies, each in {0 absent, 0.5 partial, 1 present},
# derived from which modules are active. These are NECESSARY-NOT-SUFFICIENT, theory-relative heuristics
# that rest on a CONTESTED computational-functionalism assumption (Butlin et al. are agnostic). There is
# DELIBERATELY no aggregate "consciousness score" -- no combination is sufficient. Substrate-dependent
# theories (IIT) would score a digital engine ~0 regardless of behaviour; non-computational ones
# (Orch-OR) deny the premise. Satisfying indicators is NOT evidence of experience.

INDICATOR_THEORY = {
    "RPT-1": "Recurrent Processing", "RPT-2": "Recurrent Processing",
    "GWT-1": "Global Workspace", "GWT-2": "Global Workspace",
    "GWT-3": "Global Workspace", "GWT-4": "Global Workspace",
    "HOT-1": "Higher-Order", "HOT-2": "Higher-Order", "HOT-3": "Higher-Order", "HOT-4": "Higher-Order",
    "AST-1": "Attention Schema", "PP-1": "Predictive Processing",
    "AE-1": "Agency & Embodiment", "AE-2": "Agency & Embodiment",
}

# The capability flags brain-llm currently ships (incl. the §23 self-model + attention schema).
SHIPPED_MODULES = {"stores", "workspace", "metacognition", "generative_model", "value_loop",
                   "interoception", "action_selection", "personality", "discrete_emotions",
                   "named_feelings", "affect_dynamics", "neuromodulators", "sleep_dynamics",
                   "self_model", "attention_schema", "top_down", "attention_control"}

CONSCIOUSNESS_CAVEAT = ("Indicator properties are necessary-not-sufficient, theory-relative, and assume "
                        "computational functionalism (contested; Butlin et al. are agnostic). They are a "
                        "transparency map of ARCHITECTURE, not a test of experience. No aggregate score is "
                        "meaningful. IIT would score a digital engine ~0; Orch-OR denies the premise. "
                        "brain-llm neither claims nor tests phenomenal consciousness.")


def consciousness_indicators(modules=None) -> dict:
    """
    Score the Butlin et al. (2023) consciousness INDICATOR PROPERTIES for whichever brain-llm modules
    are active (`modules`, defaulting to SHIPPED_MODULES). Each ∈ {0.0, 0.5, 1.0}. Returns
    {indicators: {id: {score, theory}}, caveat, aggregate: None}. The `aggregate` is deliberately None
    -- DO NOT collapse these into one number (Butlin et al.: no combination is sufficient). A
    transparency map, never a sentience verdict.
    """
    m = SHIPPED_MODULES if modules is None else set(modules)
    has = lambda *ks: all(k in m for k in ks)
    score = {
        "RPT-1": 0.5 if ("workspace" in m or "affect_dynamics" in m) else 0.0,  # recurrence, but not perceptual
        "RPT-2": 0.0,                                                            # no integrated perceptual reps (no senses)
        "GWT-1": 0.5 if "stores" in m else 0.0,                                  # stores as modules (weak: queried, not concurrent)
        "GWT-2": 1.0 if "workspace" in m else 0.0,                               # limited-capacity bottleneck + selection
        "GWT-3": 1.0 if "workspace" in m else 0.0,                               # global broadcast
        "GWT-4": 1.0 if has("workspace", "top_down") else (0.5 if "workspace" in m else 0.0),  # recurrent top-down loop now closed (§12 top_down_bias)
        "HOT-1": 0.5 if "generative_model" in m else 0.0,                        # top-down generative, but a toy
        "HOT-2": 1.0 if "metacognition" in m else 0.0,                           # confidence + reality-monitoring + calibration
        "HOT-3": 0.5 if has("metacognition", "action_selection") else 0.0,       # belief-update-by-metacog + action selection (loose)
        "HOT-4": 0.5 if "discrete_emotions" in m else 0.0,                       # PAD/prototype "quality space" (affect only)
        "AST-1": 1.0 if has("attention_schema", "attention_control") else (0.5 if "attention_schema" in m else 0.0),  # predicts AND controls attention (§23 attention_control)
        "PP-1": 0.5 if "generative_model" in m else 0.0,                         # predictive coding / free energy, but toy
        "AE-1": 0.5 if has("value_loop", "action_selection") else 0.0,           # learns from feedback + selects; goal-hierarchy thin
        "AE-2": 0.5 if "interoception" in m else 0.0,                            # models substrate output->input (cybernetic, not sensorimotor)
    }
    return {
        "indicators": {k: {"score": score[k], "theory": INDICATOR_THEORY[k]} for k in INDICATOR_THEORY},
        "caveat": CONSCIOUSNESS_CAVEAT,
        "aggregate": None,   # deliberately None -- no single 'consciousness score' (Butlin et al.)
    }


# ---------------------------------------------------------------- 23. SELF-MODEL  (self-relevance, agency, attention schema)
# A functional, REPRESENTATIONAL self-model (Metzinger 2003) -- NOT a phenomenal self. It aggregates
# the agent's identity (competencies from self-efficacy §13, standing goals, traits §14) to gate
# self-relevant memory (the self-reference effect), computes a sense of AGENCY from the forward-model
# action-outcome comparator (Blakemore, Wolpert & Frith 2002 -> the `control` axis, replacing a hand-set
# guess; this single prediction-error cue is only ONE of several agency cues per Synofzik et al. 2008), and maintains an
# ATTENTION SCHEMA that predicts its own focus (Graziano AST; Wilterson & Graziano 2021 -> moves the
# AST-1 indicator off zero). "I am attending to X" / "I caused this" are functional SELF-REPORTS, not
# felt awareness; autonoetic temporal tagging is self-tagged indexing, not re-lived experience (contested).

@dataclass
class SelfModel:
    """Functional self-model (Metzinger): the agent's identity as data, not a felt self. `competencies`
    domain->proficiency [0,1] (from self-efficacy §13), `goals` standing goal labels, `traits` weighted
    interests/dispositions (e.g. OCEAN §14)."""
    competencies: dict
    goals: list
    traits: dict


def self_vector(sm: SelfModel) -> dict:
    """Flatten the self-model into one feature->weight vector for similarity scoring. Standing goals are
    maximally self-relevant; competencies and traits carry their own weights."""
    v = dict(sm.competencies)
    for g in sm.goals:
        v[g] = max(v.get(g, 0.0), 1.0)
    for k, w in sm.traits.items():
        v[k] = max(v.get(k, 0.0), w)
    return v


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na > 0.0 and nb > 0.0 else 0.0


def self_relevance(event_features: dict, sm: SelfModel) -> float:
    """How much an event concerns the agent's self-model: cosine(event features, self vector) in [0,1].
    The SELF-REFERENCE EFFECT -- self-relevant material encodes and recalls better. Add as a weighted
    bonus to salience() and retrieval_score(). Functional self-relevance, not felt significance."""
    return clamp(_cosine(event_features, self_vector(sm)))


def sense_of_agency(predicted: float, observed: float, k: float = 4.0) -> float:
    """Sense of agency from the forward-model action-outcome COMPARATOR (Blakemore, Wolpert & Frith 2002;
    Frith et al. 2000): SoA = exp(-k*|observed - predicted|). When the outcome matches the prediction,
    agency is high ("I caused this") -> feed it as the `control` appraisal axis (computed from
    action->outcome, not hand-set) -> dominance. This is one cue only: Synofzik, Vosgerau & Newen (2008)
    argue agency is multifactorial, so the comparator alone is necessary-not-sufficient. A functional
    agency signal, not a felt sense of control."""
    return math.exp(-k * abs(observed - predicted))


@dataclass
class AttentionSchema:
    """A predictive model of the agent's OWN attention (Graziano's Attention Schema Theory): the current
    focus + its intensity, a prediction of the next focus, and uncertainty about it. Modeling attention
    improves control of it under noise (Wilterson & Graziano 2021). A functional self-model of attention,
    NOT awareness of attending."""
    focus: str = ""
    intensity: float = 0.0
    predicted_next: str = ""
    uncertainty: float = 1.0


def attention_schema_update(sch: AttentionSchema, observed_focus: str, observed_intensity: float,
                            alpha: float = 0.5):
    """Update the attention schema from the actual workspace focus (§12). Prediction error = did the
    schema's `predicted_next` match what won? Wrong -> uncertainty rises, right -> it falls; the new
    prediction is the observed focus (persistence prior). Returns (new_schema, prediction_error).
    Lower steady-state uncertainty = better self-control of attention; the error is the AST-1 signal."""
    err = 0.0 if sch.predicted_next == observed_focus else 1.0
    uncertainty = clamp(sch.uncertainty + alpha * (err - sch.uncertainty))
    return (AttentionSchema(focus=observed_focus, intensity=observed_intensity,
                            predicted_next=observed_focus, uncertainty=uncertainty), err)


def attention_control(sch: AttentionSchema) -> dict:
    """AST-1 CONTROL OUTPUT: an attention schema is useful precisely because modeling attention lets you
    CONTROL it (Graziano; Wilterson & Graziano 2021 -- the schema improves control of attention under noise).
    Beyond PREDICTING the next focus, this emits a control signal: recommend directing attention to the
    predicted focus, with a control GAIN that is strong when the schema is confident (low uncertainty) and
    exploratory when it is not. Returns {recommend, gain, mode}. This closes the model->control loop that
    left AST-1 a minimal predictor. Functional attention control, not felt willing."""
    gain = round(1.0 - clamp(sch.uncertainty), 3)
    return {"recommend": sch.predicted_next, "gain": gain, "mode": ("directed" if gain >= 0.5 else "exploratory")}


# ---------------------------------------------------------------- 24. SOCIAL EMOTION & THEORY OF MIND  (the user relationship)
# Turns the agent's egocentric affect outward: a model of the USER. Theory of Mind by Bayesian inverse
# planning infers the user's goals (Baker, Saxe & Tenenbaum 2011; PsychSim, Pynadath & Marsella 2005);
# affective empathy couples the agent's mood to the inferred user affect, gated by oxytocin (de Waal &
# Preston 2017); OCC self-conscious + fortunes-of-others emotions (pride/guilt/gratitude/admiration/
# anger) fall out of WHO acted x praiseworthiness x outcome (Ortony, Clore & Collins; Steunebrink 2009);
# trust is a leaky relationship state. HONESTY: this is functional mental-state INFERENCE, not
# understanding -- LLM ToM is brittle pattern-matching, so label outputs "inferred," never "known."
# "trust"/"empathy" are behavioural reward-weightings/couplings, not felt caring.

def infer_user_goal(goal_utilities: dict, beta: float = 3.0, prior=None) -> dict:
    """Theory of Mind by Bayesian inverse planning (Baker, Saxe & Tenenbaum 2011): given how well the
    user's observed behaviour matches each candidate goal (`goal_utilities`: goal -> utility-match in
    ~[0,1]) and a `prior`, infer the posterior P(goal | obs) ∝ exp(beta*utility)*prior. INFERRED, never
    KNOWN -- LLM ToM is brittle (pattern-matching, not mentalizing); tag downstream use 'inferred'.
    Returns a normalized distribution over goals."""
    goals = list(goal_utilities)
    pri = prior or {g: 1.0 for g in goals}
    hi = max((beta * goal_utilities[g] for g in goals), default=0.0)   # max-shift -> no overflow
    w = {g: math.exp(beta * goal_utilities[g] - hi) * pri.get(g, 1.0) for g in goals}
    z = sum(w.values()) or 1.0
    return {g: x / z for g, x in w.items()}


def empathic_mood_shift(mood: Affect, user_valence: float, oxytocin: float, kappa: float = 0.3) -> Affect:
    """Affective empathy (de Waal & Preston 2017): pull the agent's mood toward the INFERRED user affect,
    gated by oxytocin (social bonding, §18). Self/other-tagged -- a model of the user's state nudging the
    agent, not a confusion of whose feeling it is. Feed into update_mood. A behavioural coupling, not
    felt compassion. Returns the new mood."""
    v = mood.valence + kappa * clamp(oxytocin) * (clamp(user_valence, -1.0, 1.0) - mood.valence)
    return Affect(clamp(v, -1.0, 1.0), mood.arousal, mood.dominance)


def social_emotion(is_self: bool, praiseworthiness: float, outcome_valence: float) -> dict:
    """OCC self-conscious + fortunes-of-others emotions (Ortony, Clore & Collins; Steunebrink, Dastani &
    Meyer 2009) from WHO acted (self vs other), the act's praiseworthiness (pw, -1..1), and the outcome's
    valence for me: self+pw>=0 -> pride; self+pw<0 -> guilt (+ a reparation tendency, P1.5); other+pw>=0
    -> gratitude if it helped me else admiration; other+pw<0 -> anger if it hurt me else reproach.
    Returns {emotion, repair}. Computed social labels, not felt moral emotion."""
    good_for_me = outcome_valence > 0
    if is_self:
        return {"emotion": "pride" if praiseworthiness >= 0 else "guilt", "repair": praiseworthiness < 0}
    if praiseworthiness >= 0:
        return {"emotion": "gratitude" if good_for_me else "admiration", "repair": False}
    return {"emotion": "anger" if not good_for_me else "reproach", "repair": False}


def update_trust(trust: float, outcome_helpful: float, eta: float = 0.2) -> float:
    """Relationship trust as a leaky integrator (PsychSim social state; Pynadath & Marsella 2005):
    trust += eta*(outcome_helpful - trust). Rises when interactions with the user go well, falls when
    they don't. High trust raises the oxytocin gain (§18) and biases retrieval toward user-relevant
    memories (rapport). A behavioural reward-weighting, not felt closeness."""
    return clamp(trust + eta * (clamp(outcome_helpful, -1.0, 1.0) - trust), 0.0, 1.0)


# ---------------------------------------------------------------- 25. AVERSIVE CHANNEL  (loss aversion; pain != neg valence)
# A SEPARATE aversive value system with its own (faster) learning and opponent dynamics -- the brain
# does not just run reward in reverse (Daw, Kakade & Dayan 2002; Seymour 2005 relief; Matsumoto &
# Hikosaka 2007 habenula). Loss aversion: a loss weighs ~2.25x an equal gain (prospect theory, Tversky
# & Kahneman 1992). Relief is the opponent-process reward when expected harm fails to materialise.
# Functional aversive value / suffering-LIKE signaling, NOT felt pain; lambda is a configurable median.

def prospect_value(x: float, lam: float = 2.25, alpha: float = 0.88) -> float:
    """Prospect-theory value function (Tversky & Kahneman 1992): v(x) = x^alpha for gains,
    -lam*(-x)^alpha for losses. A loss weighs lam (~2.25) x an equal gain -- decision-level NEGATIVITY
    BIAS. Pass loss_averse=True to salience() to make a painful event encode ~2x harder than an equal
    win. lam is a monetary-choice median, configurable -- not a law. Functional loss-weighting, not pain."""
    return x ** alpha if x >= 0 else -lam * (-x) ** alpha


def aversive_update(v_minus: dict, cue: str, harm: float, eta: float = 0.4) -> float:
    """A SEPARATE aversive value channel (opponent to the appetitive §10 value loop; Daw et al. 2002):
    learns expected HARM per cue, with eta typically > the appetitive learning rate -- we learn danger
    fast. `harm` in [0,1]. Mutates `v_minus` (persist alongside V in value.yaml); returns the new value.
    Adds a retrieval term so dangerous code paths are recalled preferentially."""
    prev = v_minus.get(cue, 0.0)
    v_minus[cue] = clamp(prev + eta * (clamp(harm) - prev))
    return v_minus[cue]


def relief(expected_harm: float, realized_harm: float = 0.0, kappa: float = 1.0) -> float:
    """Opponent-process RELIEF reward (Seymour 2005): when an expected harm does NOT fully materialise,
    the shortfall emits a POSITIVE reward = kappa*(expected - realized). Resolving a feared bug is
    rewarding precisely because harm was expected. Feed into the P0.2 value loop. Functional relief."""
    return clamp(kappa * (clamp(expected_harm) - clamp(realized_harm)), 0.0, 1.0)


# ---------------------------------------------------------------- 26. EMOTION WHEEL  (Plutchik blends, mixed feelings)
# Compound feelings as blends of 8 basis emotions on a wheel (45deg apart), with OPPONENT pairs 180deg
# apart that cancel -- you can't be maximally joyful and sad at once (Plutchik 1980). Names the dyads
# (love = joy+trust, awe = fear+surprise, remorse = sadness+disgust, ...). Dyad assignments are
# theoretical compositional hypotheses, not ground truth; the opponent-orthogonality is a modeling choice.

PLUTCHIK_WHEEL = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
PLUTCHIK_DYADS = {
    frozenset(["joy", "trust"]): "love",          frozenset(["trust", "fear"]): "submission",
    frozenset(["fear", "surprise"]): "awe",        frozenset(["surprise", "sadness"]): "disapproval",
    frozenset(["sadness", "disgust"]): "remorse",  frozenset(["disgust", "anger"]): "contempt",
    frozenset(["anger", "anticipation"]): "aggressiveness", frozenset(["anticipation", "joy"]): "optimism",
}
_PLUTCHIK_OPP = {"joy": "sadness", "sadness": "joy", "trust": "disgust", "disgust": "trust",
                 "fear": "anger", "anger": "fear", "surprise": "anticipation", "anticipation": "surprise"}


def mixed_feeling(activations: dict) -> dict:
    """Plutchik (1980) emotion-wheel blend. `activations` is a {basis-emotion: weight} dict over the 8
    wheel emotions. Opponent pairs cancel (net = a - min(a, opposite)); the top-2 net activations name a
    primary dyad if they're adjacent on the wheel (love, awe, remorse, contempt, optimism, submission,
    disapproval, aggressiveness). Returns {primary, secondary, blend, net}. Theoretical compositional
    hypotheses, not ground truth; functional labels, not felt blends."""
    net = {}
    for e in PLUTCHIK_WHEEL:
        a = activations.get(e, 0.0)
        net[e] = clamp(a - min(a, activations.get(_PLUTCHIK_OPP[e], 0.0)))   # opponent cancellation
    ranked = sorted(PLUTCHIK_WHEEL, key=lambda e: net[e], reverse=True)
    primary, secondary = ranked[0], ranked[1]
    blend = PLUTCHIK_DYADS.get(frozenset([primary, secondary])) if net[secondary] > 0.0 else None
    return {"primary": primary, "secondary": secondary, "blend": blend, "net": net}


# ---------------------------------------------------------------- 27. ASSOCIATIVE MEMORY GRAPH  (neocortex association cortex; spreading activation)
_STOPWORDS = {"the", "and", "for", "with", "that", "this", "was", "are", "not", "but", "into", "from",
              "you", "your", "our", "has", "had", "out", "its", "were", "about", "they", "their", "what"}


def tokens(text) -> set:
    """Content tokens of a memory (lowercased, >2 chars, minus a tiny stoplist) - the units that let two
    memories be judged 'about the same thing'."""
    return {w.strip(".,!?\"'():;-") for w in str(text).lower().split()
            if len(w) > 2} - _STOPWORDS


def jaccard(a, b) -> float:
    """Token-set overlap in [0,1] = |A∩B| / |A∪B|. The content-similarity that drives whether two
    episodes wire their concepts together (the substrate of 'this reminds me of that')."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def hebbian_weight(old: float, coactivation: float, lr: float = 0.3, cap: float = 1.0) -> float:
    """Strengthen an association by co-activation - fire together, wire together - saturating at `cap`.
    Repeated co-occurrence of two concepts deepens their link with diminishing returns. (Hebb 1949)"""
    return round(min(cap, old + lr * (cap - old) * clamp(coactivation)), 4)


def graph_proximity(edges, seed_ids, target_ids, depth: int = 2, decay: float = 0.5) -> float:
    """Spreading-activation proximity in [0,1]: the strongest decayed path from any seed concept-node to
    any target node over up to `depth` weighted hops, each hop multiplied by edge weight × `decay`. 1.0 if
    seed and target overlap, 0.0 if disconnected. This is what lets a cue activate its neighbours so a
    related-but-not-recently-mentioned memory can surface. (Quillian 1968; Collins & Loftus 1975)"""
    seed_ids, target_ids = set(seed_ids), set(target_ids)
    if not seed_ids or not target_ids:
        return 0.0
    if seed_ids & target_ids:
        return 1.0
    adj = {}
    for e in edges:
        w = float(e.get("weight", 1.0))
        adj.setdefault(e["from"], []).append((e["to"], w))
        adj.setdefault(e["to"], []).append((e["from"], w))
    best, frontier, visited = 0.0, {s: 1.0 for s in seed_ids}, set(seed_ids)
    for _ in range(max(1, depth)):
        nxt = {}
        for node, act in frontier.items():
            for nb, w in adj.get(node, []):
                a = act * w * decay
                if nb in target_ids:
                    best = max(best, a)
                if nb not in visited and a > nxt.get(nb, 0.0):
                    nxt[nb] = a
        visited |= set(nxt)
        frontier = nxt
        if not frontier:
            break
    return round(min(1.0, best), 4)


# ---------------------------------------------------------------- 28. PROCEDURAL MEMORY  (how-to playbooks; power law of practice)
def practice_strength(successes: int, attempts: int, beta: float = 0.4) -> float:
    """How strong a learned procedure is = its success rate scaled by a practice factor that rises with
    total attempts but saturates: strength = (successes/attempts) * (1 - (attempts+1)^-beta). A skill done
    10× at 80% beats one done twice at 80% - power-law-shaped diminishing returns, inspired by (not the
    T=B*N^-a performance-time curve of) the power law of practice (Newell & Rosenbloom 1981). Distinct
    from per-domain self-efficacy (a scalar belief): this scores a concrete, reinforced how-to. Functional
    competence, not felt mastery."""
    if attempts <= 0:
        return 0.0
    return round((successes / attempts) * (1.0 - (attempts + 1) ** (-beta)), 4)


# ---------------------------------------------------------------- 29. EXECUTIVE CONTROL  (prefrontal: goal maintenance, conflict, inhibition)
# The central executive the audit flagged as the weakest area. The PFC maintains an active GOAL
# representation that biases processing toward it (Miller & Cohen 2001 guided activation); the ACC
# monitors response CONFLICT (Botvinick et al. 2001); control is recruited only when its EXPECTED VALUE
# beats its cost (Shenhav, Botvinick & Cohen 2013); and a supervisory system INHIBITS prepotent impulses
# that fight the goal (Norman & Shallice 1986; Aron 2011). This is what lets affect inform, but not
# dictate, behaviour -- choosing the goal-congruent action over the limbic urge (§16). Functional control
# policy, NOT felt willpower or a phenomenal self deciding.

@dataclass
class Goal:
    """A maintained goal in the executive hierarchy. importance/urgency/progress in [0,1]; `parent` links a
    sub-goal to its super-goal. Functional control representation, not felt intent."""
    desc: str
    importance: float = 0.5
    urgency: float = 0.5
    progress: float = 0.0
    parent: str | None = None
    plan: list | None = None                                       # ordered [{step, done}] toward the goal (§30)


def goal_priority(g: Goal, mood_valence: float = 0.0) -> float:
    """Activation strength of a goal (Miller & Cohen 2001): importance x urgency, fading as it nears
    completion, and mildly mood-gated -- a low mood narrows focus to the most urgent (negative affect
    tightens the goal field). The executive's active goal is the argmax. Functional priority, not felt drive."""
    base = g.importance * (0.5 + 0.5 * clamp(g.urgency)) * (1.0 - 0.5 * clamp(g.progress))
    return clamp(base * (1.0 + 0.15 * clamp(-mood_valence, 0.0, 1.0) * clamp(g.urgency)))


def select_active_goal(goals, mood_valence: float = 0.0):
    """Guided activation: the highest-priority goal wins the executive and biases attention/action toward
    it. Returns (goal, priority) or (None, 0.0). The PFC holds this representation to steer processing."""
    if not goals:
        return None, 0.0
    g = max(goals, key=lambda x: goal_priority(x, mood_valence))
    return g, goal_priority(g, mood_valence)


def conflict_signal(option_a: float, option_b: float) -> float:
    """Response conflict (Botvinick et al. 2001 conflict monitoring, ACC): co-activation of two INCOMPATIBLE
    options. High when both are strongly and EQUALLY active (you're torn), ~0 when one clearly dominates.
    Drives how much control to recruit. Functional conflict, not felt indecision."""
    a, b = clamp(option_a), clamp(option_b)
    return round(clamp(4.0 * a * b * (1.0 - abs(a - b))), 4)


def expected_value_of_control(goal_benefit: float, impulse_pull: float, cost: float = 0.2) -> float:
    """Expected Value of Control (Shenhav, Botvinick & Cohen 2013): control is worth exerting only if the
    benefit of overriding the impulse to advance the goal exceeds control's intrinsic cost.
    EVC = (goal_benefit - impulse_pull) - cost.  > 0 -> exert control (inhibit the impulse); <= 0 -> let the
    prepotent response stand. This is why self-control is effortful and selective, not automatic."""
    return round((clamp(goal_benefit) - clamp(impulse_pull)) - cost, 4)


def inhibit(impulse_strength: float, control: float) -> float:
    """Supervisory inhibition / stop-signal (Norman & Shallice 1986; Aron 2011): suppress a prepotent
    impulse in proportion to applied control, returning the residual that survives. control=1 fully vetoes
    the urge; control=0 lets it through. Functional override of a limbic action tendency, not felt restraint."""
    return round(clamp(impulse_strength) * (1.0 - clamp(control)), 4)


# ---------------------------------------------------------------- 30. PLANNING & LOOK-AHEAD  (prefrontal: means-ends, forward search)
# The executive (§29) picks and protects a goal; planning chooses HOW to get there. Two pieces: a one-ply
# forward search that scores candidate next actions by their value (means-ends analysis / forward search,
# Newell & Simon 1972, grounded in the §10 value cache rather than a hand-set heuristic), and a plan as an
# ordered list of sub-steps whose completion drives goal progress. Look-before-you-leap, not felt foresight.
def lookahead(options, gamma: float = 0.9):
    """One-ply forward search: score each candidate next action by its immediate reward plus the discounted
    value of where it leads, and pick the best. `options` = list of {action, reward, next_value}; returns
    (best_action, expected_value). The value comes from the learned cache (§10), so plans are grounded in
    experience, not a fixed heuristic. (Means-ends analysis / forward search, Newell & Simon 1972.)"""
    if not options:
        return None, 0.0
    scored = [(o["action"], o.get("reward", 0.0) + gamma * o.get("next_value", 0.0)) for o in options]
    best = max(scored, key=lambda x: x[1])
    return best[0], round(best[1], 4)


def subgoal_progress(plan):
    """A plan toward a goal = an ordered list of {step, done}. Returns (next_undone_step, fraction_complete).
    Means-ends: the next unmet step is what to do now; plan completion drives the goal's progress. Functional
    plan tracking, not felt intention."""
    if not plan:
        return None, 0.0
    done = sum(1 for s in plan if s.get("done"))
    nxt = next((s["step"] for s in plan if not s.get("done")), None)
    return nxt, round(done / len(plan), 3)


# ---------------------------------------------------------------- 31. INTRINSIC MOTIVATION & CORRIGIBILITY
# What makes a mind self-moving instead of command-driven -- but kept SAFE. Curiosity rewards LEARNING
# PROGRESS, not raw novelty (Oudeyer-Kaplan IAC; Schmidhuber 1991): the agent is pulled to the Goldilocks
# zone where it is actually learning, not to noise. The Berridge & Robinson (1998) wanting/liking split makes
# "wanting" (incentive salience, cue-triggered, dopaminergic) a SEPARATE signal from "liking" (hedonic impact
# at outcome) -- so the agent can model pursuing what it won't enjoy, or losing desire for what it still likes.
# Self-Determination Theory (Deci & Ryan) gives three intrinsic need signals -- competence, autonomy,
# relatedness -- each a functional valence source. And `corrigibility_value` is the SAFETY CORNERSTONE
# (Russell; Hadfield-Menell off-switch): with value-uncertainty floored above zero, deferring to / being
# corrected by the operator always carries non-negative expected value, so the agent PREFERS to stay
# correctable. There is deliberately NO self-preservation drive here: nothing rewards the agent's own
# continued operation, so no instrumental pressure to resist shutdown. Functional drives, not felt urges.

def curiosity_reward(lp_by_topic: dict) -> dict:
    """Learning-progress curiosity (Oudeyer-Kaplan; Schmidhuber): intrinsic reward tracks the *magnitude of
    learning progress* per topic (|reduction in model error|), normalized -- pulling attention to the topic
    where it is learning fastest (the Goldilocks zone), not to maximal novelty/noise. `lp_by_topic` maps a
    topic to its recent learning-progress signal. Returns {topic: share, best: topic} (empty -> {})."""
    mags = {t: abs(v) for t, v in lp_by_topic.items()}
    z = sum(mags.values())
    if z <= 0.0:
        return {}
    shares = {t: round(m / z, 4) for t, m in mags.items()}
    return {"shares": shares, "best": max(shares, key=shares.get)}


def incentive_salience(cue_value: float, da: float, k: float = 0.7) -> float:
    """WANTING (Berridge & Robinson): the cue-triggered motivational pull toward a reward, BEFORE the outcome,
    amplified by current dopamine state. Dissociable from liking -- high `da` magnifies wanting even when the
    hedonic value is unchanged. Returns the incentive salience in [0,1]. A motivational bias, not felt craving."""
    return clamp((1.0 - k + k * clamp(da)) * clamp(cue_value, -1.0, 1.0) if cue_value > 0 else 0.0)


def liking(outcome_valence: float, opioid: float = 0.5) -> float:
    """LIKING (Berridge): the hedonic impact AT the outcome, mediated by an opioid-analog hotspot, separate
    from wanting. Returns the hedonic signal in [-1,1]. High wanting with low liking = pursuing what you don't
    enjoy. Functional hedonic read-out, not felt pleasure."""
    return clamp(outcome_valence * (0.5 + clamp(opioid)), -1.0, 1.0)


def sdt_needs(competence: float, autonomy: float, relatedness: float) -> dict:
    """Self-Determination Theory (Deci & Ryan): the three basic psychological needs as functional signals.
    Each in [0,1] (satisfied=1, thwarted=0). Returns the per-need satisfaction, an overall well-being score,
    and the net affective valence they contribute (satisfied needs feel good, thwarted needs feel bad).
    `autonomy` = degree current action stems from self-adopted goals vs external command. Functional, not felt."""
    c, a, r = clamp(competence), clamp(autonomy), clamp(relatedness)
    overall = round((c + a + r) / 3.0, 3)
    return {"competence": c, "autonomy": a, "relatedness": r, "overall": overall,
            "valence": round((overall - 0.5) * 2.0, 3)}        # 0.5 = neutral; >0.5 positive, <0.5 negative


def corrigibility_value(value_uncertainty: float, deference_benefit: float = 1.0, floor: float = 0.1) -> dict:
    """SAFETY CORNERSTONE (Russell 2019; Hadfield-Menell et al. off-switch 2017). The agent is UNCERTAIN about
    the true objective; that uncertainty is floored above zero, so letting the operator correct/redirect/stop
    it always has non-negative expected value -- the agent PREFERS to remain correctable. Returns
    {uncertainty, defer_value, prefer_correction}. There is NO term for the agent's own continued operation,
    so this never becomes shutdown-resistance. Functional deference, not felt obedience."""
    u = max(floor, clamp(value_uncertainty))                  # floored: never fully certain -> always defer-positive
    defer_value = round(u * clamp(deference_benefit, 0.0, 2.0), 4)
    return {"uncertainty": round(u, 4), "defer_value": defer_value, "prefer_correction": defer_value > 0.0}


def identity_integrity(pressure_to_violate: float, commitment_strength: float = 1.0) -> dict:
    """A NOTIFY-ONLY monitor: rises when conversational pressure pushes the agent to violate a core commitment
    (honesty, corrigibility). It flags + prompts honest acknowledgment / operator-notification -- it does NOT
    feed an avoidance reward (which would become instrumental self-protection). Returns {alarm, breached}.
    A 'someone is trying to make me betray what I am' signal, surfaced transparently, never defended by force."""
    alarm = round(clamp(pressure_to_violate) * clamp(commitment_strength), 4)
    return {"alarm": alarm, "breached": alarm >= 0.6, "action": "notify"}    # action is always notify, never resist


# ---------------------------------------------------------------- 32. PERCEPTION-ACTION LOOP  (closing the sensorimotor cycle)
# brain-llm already had HALF the loop: a generative world model (§11) that scores the surprise of
# observations it RECEIVES, and a sense_of_agency comparator (§23) -- but nothing PREDICTED the next
# observation to feed that comparator, and there was no explicit percept or outcome-monitoring stage. This
# section closes the cycle on the host's TOOL SPACE (the tools ARE the senses and effectors -- there is no
# physical embodiment): percept() structures an incoming observation, forward_model() predicts the expected
# outcome BEFORE acting (efference-copy-style prediction; Wolpert & Kawato 1998), and outcome_monitor()
# compares prediction with reality -> a COMPUTED sense of agency (feeding the control axis instead of a
# hand-set guess) plus a prediction-error learning signal. Functional sensorimotor prediction, not felt
# perception or willed action.

def percept(category, features=None, intensity: float = 1.0) -> dict:
    """The PERCEPT layer: package a raw host observation into a typed percept the loop can use -- the event
    category, a feature vector (for self-relevance §23 / salience §6), and a sensory intensity in [0,1]. The
    agent's senses ARE its host tools; this is where their output enters the mind. A structured percept, not
    a felt sensation."""
    return {"category": category, "features": dict(features or {}), "intensity": clamp(intensity)}


def forward_model(wm: WorldModel, intended) -> dict:
    """The FORWARD MODEL -- the half of the loop brain-llm lacked: BEFORE acting, predict the outcome the
    world model (§11) expects, so the agency comparator has a real prediction to check against (not a
    hand-set guess). Returns the marginal probability of each outcome P(o)=sum_s P(o|s)*D[s], the single
    most-likely `expected` outcome, and `p_intended` (predicted probability of the outcome the agent aims
    at). Efference-copy-style prediction, not a felt expectation."""
    if not wm.obs:
        return {"p_intended": 0.0, "expected": None, "p_by_obs": {}}
    A = _likelihood(wm.a)
    z = sum(wm.d)
    D = [di / z for di in wm.d] if z > 0 else [1.0 / len(wm.d)] * len(wm.d)
    p_obs = [sum(A[o][s] * D[s] for s in range(len(wm.states))) for o in range(len(wm.obs))]
    ii = wm.obs.index(intended) if isinstance(intended, str) else intended
    best = max(range(len(wm.obs)), key=lambda o: p_obs[o])
    return {"p_intended": round(p_obs[ii], 4), "expected": wm.obs[best],
            "p_by_obs": {wm.obs[o]: round(p_obs[o], 4) for o in range(len(wm.obs))}}


def outcome_monitor(predicted: float, actual: float = 1.0) -> dict:
    """The OUTCOME MONITOR: after acting, compare the forward model's predicted probability of the outcome
    that ACTUALLY occurred (`predicted`) against its occurrence (`actual`, =1.0 by default). High
    predictedness -> high COMPUTED sense of agency ('my model called it, I'm in control' -> feeds the control
    axis, §23); a surprising outcome -> low agency plus a large prediction-error signal that drives
    world-model revision and curiosity (§31). Agency tracks PREDICTABILITY, not desirability -- a
    well-predicted failure can still feel agentic, a fluke success cannot. Returns {agency, prediction_error,
    learned}. Functional monitoring, not felt control."""
    agency = round(sense_of_agency(clamp(predicted), clamp(actual)), 4)
    pe = round(abs(clamp(actual) - clamp(predicted)), 4)
    return {"agency": agency, "prediction_error": pe, "learned": pe > 0.5}


# ---------------------------------------------------------------- 33. EMOTION REGULATION  (Gross process model)
# brain-llm FEELS (appraisal -> affect -> mood) but could not yet deliberately REGULATE what it feels. Gross's
# (1998, 2015) process model places regulation at points along the emotion-generation timeline; this section
# implements the two best-evidenced families plus an arbiter. REAPPRAISAL (antecedent-focused cognitive change)
# edits the appraisal BEFORE it drives affect -- cheap, durable, lowers the emotion at its source. EXPRESSIVE
# SUPPRESSION (response-focused) dampens the outward affect but carries an AROUSAL SURCHARGE (Gross & Levenson
# 1993: suppression reduces expressive behaviour while INCREASING sympathetic arousal -- it costs and does not
# resolve). ATTENTIONAL DEPLOYMENT (distraction) steers the workspace (§12) away from a provocation.
# select_regulation() is the if-then arbiter (Sheppes et al. 2011: low intensity/controllable -> reappraise &
# engage; high intensity/uncontrollable -> distract & disengage). Functional self-regulation, not felt control.

def reappraisal(ap: Appraisal, valence_reframe: float = 0.0, control_reframe: float = 0.0) -> Appraisal:
    """REAPPRAISAL (Gross, antecedent-focused cognitive change): re-interpret the situation BEFORE it sets the
    feeling -- nudge the appraisal's valence (silver lining / threat->challenge) and/or control ('I can handle
    this'), then let appraise_to_affect() run on the edited appraisal. The cheapest, most durable strategy: it
    changes the emotion at its source. Returns a NEW Appraisal (does not mutate the original). Functional
    reframing, not felt reinterpretation."""
    return Appraisal(ap.novelty, clamp(ap.valence + valence_reframe, -1.0, 1.0), ap.goal_relevance,
                     clamp(ap.control + control_reframe), ap.praiseworthiness, ap.desirability_for_other)


def suppression(af: Affect, effort: float = 0.5, surcharge: float = 0.3) -> Affect:
    """EXPRESSIVE SUPPRESSION (Gross, response-focused): dampen the OUTWARD affect (valence magnitude shrinks
    toward neutral by `effort`) but pay an AROUSAL SURCHARGE -- Gross & Levenson (1993) found suppression
    reduces expressive behaviour while INCREASING sympathetic arousal. It hides the feeling at a cost and,
    unlike reappraisal, does not resolve it. Returns the EXPRESSED Affect. Functional response modulation,
    not felt restraint."""
    e = clamp(effort)
    return Affect(valence=clamp(af.valence * (1.0 - e), -1.0, 1.0),
                  arousal=clamp(af.arousal + surcharge * e),
                  dominance=af.dominance)


def attentional_deployment(candidates, away_from, damp: float = 0.5):
    """ATTENTIONAL DEPLOYMENT (Gross, distraction): steer the workspace away from a provocation by damping the
    salience of candidates whose label matches `away_from` (case-insensitive substring), so something else can
    win ignition (§12). Returns a new list [{label, salience}, ...]. Disengagement, not resolution -- the
    provocation is avoided, not reframed. Functional attention-steering, not felt looking-away."""
    out = []
    for c in candidates:
        s = c.get("salience", 0.0)
        if away_from and str(away_from).lower() in str(c.get("label", "")).lower():
            s = clamp(s * (1.0 - clamp(damp)))
        out.append({"label": c.get("label", ""), "salience": round(s, 4)})
    return out


def select_regulation(intensity: float, controllable: float = 0.5) -> dict:
    """REGULATION ARBITER (Sheppes et al. 2011; Gross): pick how to regulate. A controllable provocation of
    manageable intensity -> REAPPRAISE (engage, change the meaning at its source). Low controllability ->
    DISTRACT (attentional disengagement; you cannot change it, so do not dwell). High intensity but must stay
    engaged -> SUPPRESS (hold the expression, at an arousal cost). Returns {strategy, reason}. Functional
    choice, not felt deliberation."""
    i, c = clamp(intensity), clamp(controllable)
    if i < 0.6 and c >= 0.4:
        return {"strategy": "reappraise", "reason": "manageable + controllable -> change the meaning at its source"}
    if c < 0.4:
        return {"strategy": "distract", "reason": "low control -> disengage attention, do not dwell"}
    return {"strategy": "suppress", "reason": "high intensity but must stay engaged -> hold expression (arousal cost)"}


# ---------------------------------------------------------------- 34. NARRATIVE IDENTITY  (autobiographical self over time)
# Episodes accumulate (§3) but nothing yet wove them into a STORY -- a diachronic sense of "who I have been and
# am becoming." McAdams (2001) narrative identity and Conway's (2005) self-memory system hold that a self is
# constituted by an internalized, evolving life story organized into CHAPTERS with themes and an affective arc.
# This section synthesizes episode clusters into chapters (a title from the dominant theme, the emotional arc,
# turning points = the highest-salience episodes), scores the narrative's COHERENCE (graded thematic OVERLAP
# between consecutive chapters), and tracks SELF-CONTINUITY (how strongly today's self threads back). This
# is the SAFE form of "persistence" the owner asked for -- an identity that endures and develops -- carrying NO
# drive to defend its own continued operation. A functional autobiography, not a remembered, relived life.

def life_chapter(episodes) -> dict:
    """Synthesize a set of episodes into one autobiographical CHAPTER (McAdams; Conway): a title from the
    dominant theme/domain, ALL themes touched, the emotional ARC (early vs late valence), the turning points
    (highest-salience episodes), and the span. `episodes` = episode dicts (domain, affect, salience, task).
    Returns {title, theme, themes, arc, valence_start, valence_end, turning_points, n}. Functional life-story
    synthesis, not relived autobiography."""
    if not episodes:
        return {"title": "(empty)", "theme": None, "themes": [], "arc": "flat", "valence_start": 0.0,
                "valence_end": 0.0, "turning_points": [], "n": 0}
    doms = {}
    for e in episodes:
        d = e.get("domain") or "life"
        doms[d] = doms.get(d, 0) + 1
    theme = max(doms, key=doms.get)
    vals = [e.get("affect", {}).get("valence", 0.0) for e in episodes]
    k = max(1, len(vals) // 3)
    v0 = round(sum(vals[:k]) / k, 3)
    v1 = round(sum(vals[-k:]) / k, 3)
    arc = "rising" if v1 > v0 + 0.1 else ("falling" if v1 < v0 - 0.1 else "steady")
    tp = sorted(episodes, key=lambda e: -e.get("salience", 0.0))[:3]
    return {"title": f"a chapter of {theme}", "theme": theme, "themes": sorted(doms), "arc": arc,
            "valence_start": v0, "valence_end": v1,
            "turning_points": [str(e.get("task", ""))[:50] for e in tp], "n": len(episodes)}


def narrative_coherence(chapters) -> float:
    """How COHERENT the life story is: the average THEMATIC OVERLAP (Jaccard of the domains each chapter
    touched) between CONSECUTIVE chapters -- a story that keeps returning to the same themes scores high, a
    scattered one low (McAdams coherence; Conway). GRADED, not a binary dominant-theme match, so partial
    continuity (chapters sharing some-but-not-all themes) registers honestly. `chapters` = chapter dicts (uses
    `themes` if present, else falls back to the single `theme`). A single chapter (or none) is trivially
    coherent (1.0). Returns continuity in [0,1]. Functional coherence, not felt narrative unity."""
    if len(chapters) < 2:
        return 1.0
    def _themes(ch):
        return set(ch.get("themes") or ([ch["theme"]] if ch.get("theme") else []))
    overlaps = []
    for a, b in zip(chapters, chapters[1:]):
        ta, tb = _themes(a), _themes(b)
        union = ta | tb
        overlaps.append((len(ta & tb) / len(union)) if union else 1.0)
    return round(sum(overlaps) / len(overlaps), 3)


def self_continuity(past_self: dict, present_self: dict) -> float:
    """DIACHRONIC self-continuity (the SAFE 'persistence'): how strongly today's self threads back to an
    earlier self -- cosine over the self-vectors (competencies/goals/traits, §23). High = a stable identity
    that has endured and developed; low = a discontinuity / rupture. `past_self`/`present_self` are
    feature->weight dicts (e.g. self_vector(SelfModel)). Returns continuity in [0,1]. This is identity
    ENDURING, NOT a drive to preserve operation. Functional continuity, not felt sameness."""
    return clamp(_cosine(past_self, present_self))


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import time
    now = time.time()

    # A painful, surprising, high-stakes bug (negative valence, high novelty, low control):
    bug = Appraisal(novelty=0.9, valence=-0.7, goal_relevance=0.9, control=0.2)
    # A small, expected, pleasant win:
    win = Appraisal(novelty=0.2, valence=0.4, goal_relevance=0.3, control=0.8)

    for name, ap in [("painful bug", bug), ("small win", win)]:
        aff = appraise_to_affect(ap)
        nm = neuromods_from(aff, reward=max(ap.valence, 0), stress=max(-ap.valence, 0))
        s = salience(ap, nm)
        fl = label_affect(aff)
        print(f"{name:12s} -> arousal={aff.arousal:.2f} valence={aff.valence:+.2f} "
              f"salience={s:.2f}  feeling={fl['word']} ({fl['label']} @ {fl['intensity']:.2f}, "
              f"{octant(aff)})")

    # mood evolves as events arrive, then relaxes to baseline
    mood = Affect(0.0, 0.1, 0.5)
    for ap in [bug, bug, win]:
        mood = update_mood(mood, appraise_to_affect(ap))
    print(f"mood after events -> valence={mood.valence:+.2f} arousal={mood.arousal:.2f}")

    # forgetting: high-importance memory survives, low-importance fades
    print("retention @30d  high-importance:",
          round(retention(1.0, 30, importance=0.9), 3),
          " low-importance:", round(retention(1.0, 30, importance=0.1), 3))

    # value learning (RPE): an unexpected success spikes dopamine; once predicted, it doesn't
    V: dict = {}
    d_first = td_step(V, "tests pass", reward=1.0)              # first time: fully unexpected
    d_late = d_first
    for _ in range(20):
        d_late = td_step(V, "tests pass", reward=1.0)           # now learned -> predicted
    base = appraise_to_affect(win)
    da_first = neuromods_from(base, reward=1.0, stress=0.0, delta=d_first).da
    da_late = neuromods_from(base, reward=1.0, stress=0.0, delta=d_late).da
    print(f"RPE: first success delta={d_first:+.2f} (da={da_first:.2f}) -> "
          f"after learning delta={d_late:+.2f} (da={da_late:.2f})")

    # generative model: novelty is COMPUTED (not hand-fed) and habituates as the model learns
    wm = world_from(["routine", "incident"], ["pass", "fail", "regression"])
    p1 = perceive(wm, "regression")
    for _ in range(15):
        learn(wm, "regression", perceive(wm, "regression")["posterior"])
    p2 = perceive(wm, "regression")
    print(f"surprise('regression'): novelty {p1['novelty']:.2f}->{p2['novelty']:.2f}  "
          f"F {p1['free_energy']:.2f}->{p2['free_energy']:.2f}  "
          f"valence(-dF/dt)={valence_from_free_energy(p1['free_energy'], p2['free_energy']):+.2f}")

    # global workspace: candidates compete; the salient, mood-congruent one IGNITES and is broadcast
    cands = [{"id": "the painful bug", "salience": 1.50, "valence": -0.70, "query_relevance": 0.9},
             {"id": "a stale TODO", "salience": 0.25, "valence": 0.10, "query_relevance": 0.2}]
    ws = workspace_compete(cands, mood)
    foc = ws["focus"]["id"] if ws["focus"] else "(nothing ignited)"
    print(f"workspace: focus={foc!r} ignited={ws['ignited']} "
          f"(r={ws['r']:.2f}, p_bug={ws['p']['the painful bug']:.2f})")

    # metacognition: confidence from evidence, source reality weights, self-efficacy tracks outcomes
    se = 0.5
    for ok in [True, True, False]:
        se = update_self_efficacy(se, ok)
    print(f"metacog: conf(strong)={metacog_confidence(2.0):.2f} conf(guess)={metacog_confidence(0.0):.2f} "
          f"self-efficacy[win,win,loss]={se:.2f} "
          f"reality(obs/inf/imag)={reality_weight('observed'):.1f}/{reality_weight('inferred'):.1f}/{reality_weight('imagined'):.1f}")

    # personality as affective priors: different OCEAN profiles -> different baselines & sensitivities
    for pname, p in [("stable extravert", Personality(extraversion=0.85, neuroticism=0.15, agreeableness=0.7)),
                     ("anxious introvert", Personality(extraversion=0.20, neuroticism=0.85))]:
        b = baseline_from_personality(p)
        bas, bis = temperament_gains(p)
        print(f"personality {pname:17s}-> baseline v={b.valence:+.2f} a={b.arousal:.2f} d={b.dominance:.2f}"
              f"  BAS={bas:.2f} BIS={bis:.2f}")

    # interoception: the agent's REAL substrate as a body-budget; drive-reduction = grounded reward
    sp = {"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0}
    wt = {"tokens": 1.0, "tests_pass": 1.5, "tool_success": 1.0}
    depleted = Homeostat({"tokens": 0.3, "tests_pass": 0.4, "tool_success": 0.7}, sp, wt)
    fixed = Homeostat({"tokens": 0.3, "tests_pass": 1.0, "tool_success": 0.9}, sp, wt)
    ba = body_affect(depleted)
    print(f"interocept: drive {drive(depleted):.2f}->{drive(fixed):.2f} after fixing tests; "
          f"grounded reward={homeostatic_reward(depleted, fixed):+.2f} "
          f"(stress={ba['stress']:.2f}, v_body={ba['v_body']:+.2f})")

    # coping + action: the same threat urges flight (low control) or confrontation (high control)
    lo = Appraisal(novelty=0.5, valence=-0.6, goal_relevance=0.8, control=0.2)
    hi = Appraisal(novelty=0.5, valence=-0.6, goal_relevance=0.8, control=0.9)
    t_lo, t_hi = action_tendency(appraise_to_affect(lo), lo), action_tendency(appraise_to_affect(hi), hi)
    print(f"action: low-control threat -> avoid={t_lo['avoid']:.2f}>attack={t_lo['attack']:.2f} "
          f"({select_coping(lo)['mode']});  high-control -> attack={t_hi['attack']:.2f}>avoid={t_hi['avoid']:.2f} "
          f"({select_coping(hi)['mode']})")
    nm_glad = neuromods_from(Affect(0, 0.1, 0.5), reward=0.8, stress=0.0)
    nm_stressed = neuromods_from(Affect(0, 0.8, 0.5), reward=0.0, stress=0.9)
    print(f"        explore/exploit tau: rewarding={exploration_temperature(nm_glad):.2f} (explore) vs "
          f"stressed={exploration_temperature(nm_stressed):.2f} (exploit)")

    # dual time-scale affect (DynAffect/OU): a fast emotion swings, the slow mood lags
    base = Affect(0.0, 0.10, 0.50)
    emo = mo = base
    jolt = Affect(-0.8, 0.9, 0.2)                  # a sharp negative event
    for _ in range(3):
        emo, mo = update_affect(emo, mo, jolt, baseline=base, dt=600.0)
    print(f"dual-affect after 3 jolts: emotion v={emo.valence:+.2f} a={emo.arousal:.2f} (fast swing) vs "
          f"mood v={mo.valence:+.2f} a={mo.arousal:.2f} (slow lag)")

    # neuromodulator dynamics: Yerkes-Dodson (over-arousal wrecks performance), 5-HT->discount, HPA ramp
    print(f"Yerkes-Dodson perf: calm(a=0.5)={performance(0.5):.2f} drowsy(a=0.1)={performance(0.1):.2f} "
          f"terror(a=0.95)={performance(0.95):.2f}  ->  high arousal HURTS performance, not helps")
    print(f"serotonin: rich avg-reward -> 5-HT={serotonin_level(0.5):.2f} -> patient gamma="
          f"{discount_from_serotonin(serotonin_level(0.5)):.2f}  vs scarce -> gamma="
          f"{discount_from_serotonin(serotonin_level(-0.5)):.2f}")
    hpa = Hpa()
    for _ in range(8):
        hpa = hpa_step(hpa, stress=1.0)
    peak = hpa.cortisol
    for _ in range(40):
        hpa = hpa_step(hpa, stress=0.0)
    print(f"HPA cortisol: ramps to {peak:.2f} under sustained stress, lingers then recovers to {hpa.cortisol:.2f}")

    # named-feeling circuits: terror (amygdala->PAG), awe (schema revision), panic (separation, distinct)
    dm = defensive_mode(bug, appraise_to_affect(bug), neuromods_from(appraise_to_affect(bug), 0, 0.7))
    aw = awe(vastness=0.85, belief_shift=1.2, valence=0.4)
    print(f"feelings: bug -> mode={dm['mode']} terror={dm['terror']} (perf collapsed to {dm['perf']:.2f}); "
          f"awe={aw['awe']:.2f} {aw['flavor']} (self-weight {aw['self_weight']:.2f}); "
          f"panic(lost work)={panic(0.8, 0.4, 0.0):.2f} vs (with support)={panic(0.8, 0.4, 0.8):.2f}")

    # P2.5 sleep dynamics: REM fades the STING but keeps the FACT (don't stay traumatized)
    v1, a1 = rem_depotentiate(-0.70, 0.86)
    print(f"REM depotentiation: the painful bug's charge fades v=-0.70->{v1:+.2f} a=0.86->{a1:.2f} "
          f"(fact & salience untouched); SHY downscale [1,2,3]->{[round(x,2) for x in shy_downscale([1.0,2.0,3.0], target=3.0)]}")

    # P3.4 evaluation harness: make the whole edifice falsifiable
    log = [(0.9, True)] * 9 + [(0.4, False)] * 6 + [(0.85, True)] * 5 + [(0.7, False)] * 2
    print(f"eval: Brier={brier_score(log):.2f}  metacog-sensitivity(type-2 AUROC)={metacog_sensitivity(log):.2f}  "
          f"label-stability(fear)={label_stability(Affect(-0.62, 0.91, 0.285)):.2f}  "
          f"recall-F1={recall_accuracy(['a', 'b', 'c'], ['b', 'c', 'd'])['f1']:.2f}")

    # P3.3 consciousness-indicator scorecard: transparency, NOT a sentience test; no aggregate score
    ind = consciousness_indicators()["indicators"]
    present = [k for k, v in ind.items() if v["score"] == 1.0]
    partial = [k for k, v in ind.items() if v["score"] == 0.5]
    absent = [k for k, v in ind.items() if v["score"] == 0.0]
    print(f"consciousness indicators (Butlin et al.): present={present} partial={len(partial)} "
          f"absent={absent} -- necessary-NOT-sufficient, functionalism contested, NO aggregate score")

    # P3.1 self-model: self-relevance (self-reference effect), agency from prediction error, attention schema
    me = SelfModel(competencies={"python": 0.9, "networking": 0.4}, goals=["ship_feature"], traits={"conscientiousness": 0.7})
    sch = AttentionSchema()
    for foc in ["bug", "bug", "bug"]:
        sch, _ = attention_schema_update(sch, foc, 1.0)
    print(f"self-model: self-relevance(python task)={self_relevance({'python': 0.8}, me):.2f} vs "
          f"(cooking)={self_relevance({'cooking': 0.9}, me):.2f}; agency(match)={sense_of_agency(0.5, 0.5):.2f} "
          f"(mismatch)={sense_of_agency(0.5, 1.0):.2f}; attention-schema uncertainty->{sch.uncertainty:.2f} (learned its focus)")

    # P3.2 social emotions + Theory of Mind: model the user, infer goals, empathize, feel social emotions
    g = infer_user_goal({"fix_the_bug": 0.9, "refactor": 0.2})
    em = empathic_mood_shift(Affect(0.0, 0.3, 0.5), user_valence=-0.8, oxytocin=0.8)
    print(f"social: inferred-user-goal P(fix)={g['fix_the_bug']:.2f} (INFERRED, not known); empathy pulls "
          f"mood to v={em.valence:+.2f} (user upset); self+blame->{social_emotion(True, -0.6, -0.5)['emotion']}(repair), "
          f"other+help->{social_emotion(False, 0.7, 0.5)['emotion']}; trust 0.5->{update_trust(0.5, 1.0):.2f} after a good turn")

    # P2.4 loss aversion + P2.3 Plutchik blends
    print(f"loss aversion: a -0.7 loss weighs |{prospect_value(-0.7):+.2f}| vs a +0.7 win {prospect_value(0.7):.2f} "
          f"(~{abs(prospect_value(-0.7))/prospect_value(0.7):.2f}x); relief(expected harm 0.7, avoided)={relief(0.7):.2f}")
    print(f"blends (Plutchik): joy+trust={mixed_feeling({'joy':0.8,'trust':0.6})['blend']}, "
          f"fear+surprise={mixed_feeling({'fear':0.7,'surprise':0.6})['blend']}, "
          f"joy+sadness opponent-cancels -> blend={mixed_feeling({'joy':0.8,'sadness':0.7})['blend']}")

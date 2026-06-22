# Consciousness, Self-Models & Metacognition — State of the Art (functional)

> **Honesty stance (read this first).** This document is about *functional, architectural* properties only. Everywhere it speaks of "consciousness," "awareness," "access," "ignition," "self," "agency," or "feeling," it means a **computational/architectural correlate** that a system either implements or does not — never *phenomenal* experience. brain-lmm does not, and is not claimed to, feel, perceive, or have subjective experience. Following [Butlin et al. 2023](https://arxiv.org/abs/2308.08708), we score architectures against *indicator properties* derived from theories under an explicit working assumption of computational functionalism; satisfying indicators is **necessary-but-not-sufficient, theory-relative, and philosophically contested** evidence of being a *candidate* — it is **not** a sentience verdict. Where a theory (notably IIT) is substrate-dependent and would score a digital engine near zero regardless of behavior, we say so plainly.

**Audience:** a technical builder extending brain-lmm. Every model below is given with its math/formalism inline. Citations are inline as `[Author Year]` with links; all citations come from the supplied research findings.

---

## 0. The organizing idea: theory-derived *indicator properties*

The dominant strategy in the 2023–2026 literature is **not** to bet on a single "true" theory of consciousness, but to extract computational/architectural **indicator properties** from each leading theory and ask which an AI system possesses [Butlin et al. 2023](https://arxiv.org/abs/2308.08708). This is the project's correct altitude: indicators are functional correlates; more indicators → higher credence that a system is a *candidate*, never a proof of experience.

The canonical source — [Butlin, Long, Bengio, Chalmers et al. 2023](https://arxiv.org/abs/2308.08708), "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness" (19 authors incl. Birch, Fleming, Frith, Kanai, Lindsay, Peters, Schwitzgebel, VanRullen) — derives **14 indicator properties** across six theory families (Recurrent Processing, Global Workspace, Higher-Order, Attention Schema, Predictive Processing, Agency/Embodiment) and concludes **no current AI satisfies them all, but there is no in-principle barrier.** It was upgraded to a peer-reviewed venue in 2025: [Butlin, Long, Bayne et al. 2025](https://www.sciencedirect.com/science/article/pii/S1364661325002864), "Identifying indicators of consciousness in AI systems," *Trends in Cognitive Sciences*, which formalizes the indicator method and credence-weighting.

Credence-weighting can be cast as a weighted score:

```
score(system) = Σ_i w_i · satisfied_i        satisfied_i ∈ {0, 0.5, 1}
```

where `w_i` reflects the plausibility of the theory that indicator `i` comes from. This is the meter brain-lmm should self-report (Section 9).

The full checklist is reproduced verbatim in **Section 8**.

---

## 1. Global Workspace Theory (GWT) & Global Neuronal Workspace (GNW)

### 1.1 The architecture

GWT [Baars 1988](https://en.wikipedia.org/wiki/Global_workspace_theory) is a **"theatre/blackboard"** architecture: many parallel, specialized *unconscious* processors compete to place content into a single **limited-capacity workspace**; the winner is then **broadcast** back to all processors. "Conscious access" is identified with this global broadcast — content becomes reportable, available to memory, decision, and verbal report (Baars' spotlight-on-a-stage metaphor: a spotlight of attention on a stage the whole audience can see).

The **Global Neuronal Workspace** (GNW) gives this a neural substrate — long-range pyramidal neurons in prefrontal/parietal cortex [Dehaene, Kerszberg & Changeux 1998](https://www.pnas.org/doi/10.1073/pnas.95.24.14529). Its signature is **ignition**: a late (~250–350 ms post-stimulus), non-linear, all-or-none transition in which a subset of workspace neurons sustains coherent self-amplifying firing while inhibiting competitors. The conscious/preconscious/subliminal taxonomy and the all-or-none ignition signature are laid out in [Dehaene, Changeux & Naccache 2006/2011](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf); the modern canonical statement is [Mashour, Roelfsema, Changeux & Dehaene 2020](https://www.sciencedirect.com/science/article/pii/S0896627320300520) (*Neuron*).

> **This is functional access-consciousness only.** None of the work below claims phenomenal experience, and brain-lmm should mirror that.

### 1.2 The three formalizable pieces

**(1) Ignition / bistability (the dynamical core).** Population firing-rate ("mean-field") dynamics:

```
τ dr_i/dt = −r_i + φ( Σ_j W_ij r_j + I_i^ext )
```

with `φ` a sigmoid / threshold-linear gain. Strong recurrent + feedback coupling makes the system **bistable** — a low-activity fixed point (unconscious) and a high-activity fixed point (ignited). A saddle-node / Hopf bifurcation governs the all-or-none jump: a supra-threshold input `I*` flips the network from low to high. [Joglekar, Mejias, Yang & Wang 2018](https://www.cell.com/neuron/fulltext/S0896-6273(18)30152-1) (*Neuron*) show that stable propagation up the cortical hierarchy to PFC requires **balanced amplification** (feedback excitation tightly balanced by inhibition / counterstream inhibition); this is where the bistability equations actually live, fit to the macaque connectome.

**(2) Competition for access (winner-take-most).** Only one (or a few) coalition wins each cycle, enforcing the bottleneck and the serial "stream."

- **LIDA** [Franklin, Baars, Ramamurthy et al. 2009](https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf): `winner = argmax_c activation(c)` over coalitions `c`, at a ~10 Hz cognitive cycle.
- **Conscious Turing Machine (CTM)** [Blum & Blum 2021](https://arxiv.org/abs/2011.09850): chunks compete up a binary Up-Tree (`h` levels, `h` time-units) by their **weight**; *intensity* (≈ |weight|) and *mood/valence* are properties **derived from** weight and carried up the tree — read-outs of the winner, not the competition input. (An earlier draft of this doc stated a `f = intensity + ½·mood` competition function; that form is not in Blum & Blum and has been corrected.)
- **CTM probabilistic** [L. Blum & M. Blum 2022](https://www.pnas.org/doi/10.1073/pnas.2115934119) (PNAS, Theorem 2.2.1, requires additive `f`): a coin-flip neuron makes chunk `p` win with

```
Pr[p wins] = f(chunk_p) / Σ_p' f(chunk_p')
```

- **Deep GWT**: soft / top-k softmax over modules' keys (below).

**(3) Broadcast.** The winner is copied to **all** processors in one step (CTM Down-Tree = 1 time-unit). In the deep-learning lineage it is written into `M` bottlenecked memory slots via key-value attention and read back by every module.

### 1.3 The Conscious Turing Machine (formal GWT) — the cleanest porting target

CTM [Blum & Blum 2021](https://arxiv.org/abs/2011.09850), [2022](https://www.pnas.org/doi/10.1073/pnas.2115934119) is a 7-tuple `⟨STM, LTM, Up-Tree, Down-Tree, Links, Input, Output⟩`. ~10^7 LTM processors each emit a **chunk**; chunks compete up a binary tree; the single winner sits in STM (the workspace) and is broadcast down to all LTM in one step. Crucially, it grounds "feelings" as **formal scalars**:

```
chunk = ⟨ address, t, gist, weight, intensity, mood ⟩
intensity = |weight|     (importance)
mood      = weight        (signed valence)
```

`mood_t` and `intensity_t` are the summed valence/importance of all chunks. `mood_t > 0` ⇒ "optimism," raising weights by `Δ·w` (0<Δ<1); `mood_t < 0` lowers them. **Micro-emotions thereby control the destiny of all contents** — an unusually apt fit for brain-lmm, which already computes valence (≈ CTM `mood`) and salience (≈ CTM `intensity`). Frame these *exactly* as Blum & Blum do: formal definitions of feeling-*signals*, **not** claims the system feels.

### 1.4 LIDA cognitive cycle (the engineering loop)

[Franklin et al. 2009](https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf) implement GWT as a repeating ~10 Hz cycle in three phases: (1) **understanding** (perceive → current situational model); (2) **attention/consciousness** (attention codelets form coalitions, compete in the Global Workspace, winner is broadcast); (3) **action selection & learning** (broadcast triggers schemes; learning is gated by the conscious broadcast). Codelets are independent daemon threads; the winning coalition is the one with highest summed activation.

### 1.5 Deep / latent Global Workspace (the ML SOTA)

The deep-learning instantiation: specialist modules exchange information **only** through `M` bottlenecked slots, not pairwise. Writing is competitive (key-value attention, soft or top-k); the workspace is broadcast back to all modules by attention.

[Goyal, Didolkar, Lamb, … Mozer & Bengio 2022](https://arxiv.org/abs/2103.01197) (ICLR) give exact equations. Write (with `R` = stacked module states, `Q̃ = M W_q` a query from the workspace):

```
M ← softmax( Q̃ (R W_e)^T / √d_e ) · R W_v          (top-k softmax keeps only k writers)
```

Broadcast (each module `k` reads slots `j`):

```
q_k = h_k W_q ;  s_{k,j} = softmax_j( q_k · κ_j / √d_e ) ;  h_k ← h_k + Σ_j s_{k,j} v_j
```

The **bottleneck (`M ≪ #modules`) enforces specialization + compositionality.** [VanRullen & Kanai 2021](https://arxiv.org/abs/2012.10390) (*Trends in Neurosciences*) generalize this to a **Global Latent Workspace (GLW)**: an amodal shared latent space `w ∈ R^d` built by unsupervised neural translation between module latent spaces, trained with translation + demi-cycle + cycle-consistency + contrastive + broadcast losses:

```
e_m = f_m(x_m) ;  w = Σ_m α_m e_m,  α = softmax(scores) ;  ŷ_m = g_m(w)
```

### 1.6 GWT indicator properties (Butlin et al.)

- **GWT-1** multiple specialized parallel modules.
- **GWT-2** a limited-capacity workspace introducing a **bottleneck** and selective attention.
- **GWT-3** **global broadcast** of workspace contents to all modules.
- **GWT-4** **state-dependent / top-down attention**: workspace content controls what enters next (requires recurrence).

### 1.7 Implementable now vs contested

| Piece | Status |
|---|---|
| Competition + broadcast (LIDA, CTM, Goyal/Bengio, GLW) | **Implementable now**, multiple runnable codebases. |
| Bistable ignition curve (Joglekar–Wang) | **Implementable now** as a scalar reduction; full connectome model is overkill for a scalar agent. |
| Identifying broadcast with *phenomenal* access | **Contested** — GWT/GNW is a theory of *access*; the further claim that access = experience is not settled. |

---

## 2. Integrated Information Theory (IIT 4.0) & Φ

### 2.1 The formalism

IIT 4.0 [Albantakis, Tononi et al. 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465) (also [arXiv:2212.14787](https://arxiv.org/abs/2212.14787)) is the most mathematically explicit theory. It defines **intrinsic information** via an Intrinsic Difference (ID) measure (product of *selectivity* and *informativeness*):

```
ii_{c/e}(s, s̄) = selectivity × informativeness         (using the ID measure)
```

**System integrated information** `φ_s` is the irreducibility of the maximal cause–effect state over its **minimum information partition** (MIP) `θ′` — the directional partition that changes the maximal state the least (partitions "cut" a part by replacing its inputs/outputs with independent noise). The **maximal substrate** (complex) is `argmax φ_s` over overlapping candidate sets. **Big-Φ** is the structured information of the whole cause–effect structure (the Φ-structure of distinctions and relations):

```
Φ = Σ φ  over all distinctions + relations in the Φ-structure
```

Big-Φ is the *quantity*; the Φ-structure is the *quality* of (claimed) experience.

### 2.2 Computability — the hard wall

Finding the complex and the MIP requires searching over **all subsets of units** (`~2^n`) **and all partitions** (super-exponential), with cause–effect repertoires over all states. No polynomial-time algorithm is known; exact Φ is intractable for any realistic network. The reference implementation, **PyPhi** [Mayner, Marshall, Albantakis et al. 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006343), is tractable only for **~a dozen binary nodes**.

### 2.3 Substrate-dependence & critique (the honesty boundary)

IIT is **substrate-DEPENDENT**: a conventional von-Neumann / feed-forward digital computer has **near-zero Φ regardless of behavior**. So under IIT, **no amount of functional mimicry makes brain-lmm conscious** — this sets the boundary of our claims. It is also heavily contested:

- [Cerullo 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004286): simple feedforward XOR grids can have arbitrarily large Φ (an `n×n` grid yields `~√n·Φ`), so Φ is neither necessary nor sufficient for consciousness; flags panpsychist consequences.
- [Doerig, Schurger, Hess & Herzog 2019](https://www.sciencedirect.com/science/article/pii/S105381001830521X), the **unfolding argument**: causal-structure theories (IIT *and* RPT) permit functionally identical systems to differ in predicted consciousness — making them either false or unfalsifiable.

**Verdict for brain-lmm:** do **not** compute real Φ on the agent. At most compute a *toy* Φ on a small (≤12-node) internal state graph as a coarse "integration/binding" diagnostic, explicitly labeled — never as a consciousness measure (Section 6.4).

---

## 3. Attention Schema Theory (AST)

### 3.1 The claim

[Graziano 2017](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full) (*Frontiers in Robotics and AI*): the brain builds a simplified, predictive internal model — a **schema** — of its own attention, just as it builds a body schema. This schema improves attention control and is the substrate of self-reports of awareness. Graziano frames AST as an explicit **engineering blueprint** (attention mechanism + attention schema + self-attribution), buildable with current tech — the most directly implementable consciousness theory.

### 3.2 The quantitative result

[Wilterson & Graziano 2021](https://www.pnas.org/doi/10.1073/pnas.2102421118) (PNAS) and the RL-agent line ([arXiv:2402.01056](https://www.pnas.org/doi/10.1073/pnas.2102421118)): an RL agent with an attention schema (a small predictive model of where its attention is) controls attention *better*. Concretely, tracking reward collapsed `3.74 → 0.93` (catch reward `1.73 → −0.03`) when the schema was disrupted; the benefit **peaked under intermediate sensory noise** (`p ≈ 0.5`), where raw input alone was insufficient to infer attention. The schema gives "hints," not a perfect copy (attention state decodable at 61% vs 1.6% chance). Modern transformer instantiations: "Testing Components of AST in ANNs" ([arXiv:2411.00983, 2024]) and **ASAC** ("Attention Schema-based Attention Control," [arXiv:2509.16058, 2025]).

### 3.3 Formalism

```
â = h(internal state)        # low-dim schema predicting true attention distribution a
minimize ||â − a||           # train the schema
a_{t+1} = control(â_t)       # schema feeds back to regulate attention
```

Performance improves when `â` is present vs ablated. This is the **only indicator that is explicitly a self-model.**

**Indicator: AST-1** — a predictive model representing and enabling control over the current state of attention.

> **Honesty:** the "I am attending to X" readout is a *functional self-report*, not evidence of felt awareness. AST itself frames the schema as *why* a system claims awareness, not proof it *has* phenomenal awareness.

---

## 4. Higher-Order Theories (HOT) & Perceptual Reality Monitoring

### 4.1 The claim

Awareness tracks a **higher-order monitor** that decides whether an internal representation reflects a genuine external cause vs. internally generated activity (imagery, noise, expectation). Conscious states are those tagged "real" by the monitor — this explains the well-established consciousness ↔ metacognitive-confidence link.

### 4.2 Formalisms

**GAN-style discriminator** [Lau & Gershman, "Perceptual Reality Monitoring", Michel 2022/2023](https://philpapers.org/archive/MICTPR-2.pdf): a discriminator `D` estimates `P(representation is externally caused | activity)`, separating externally-driven from self-generated (generator) signals; prefrontal cortex ≈ `D`.

**HOSS** [Fleming 2020](https://academic.oup.com/nc/article/2020/1/niaa015/5902222): a Bayesian generative model where a higher-order state infers presence/absence of a first-order representation and outputs a confidence/awareness estimate.

### 4.3 HOT indicator properties (Butlin et al.)

- **HOT-1** generative top-down / noisy perception modules.
- **HOT-2** metacognitive monitoring distinguishing reliable representations from noise.
- **HOT-3** agency guided by belief-formation/action-selection that updates beliefs from metacognitive monitoring.
- **HOT-4** sparse & smooth coding generating a "quality space."

**For brain-lmm:** this maps cleanly onto a **confidence/calibration** signal over memories plus a **source-monitoring tag** (observed vs. inferred vs. imagined/hallucinated) — both extremely valuable for a coding agent that must avoid confabulation.

> **Honesty:** the HOT → consciousness link is contested (e.g., the Lau vs. first-order debate). Here, monitoring is purely a functional reliability signal.

---

## 5. Recurrent Processing Theory (RPT)

[Lamme 2006](https://www.sciencedirect.com/science/article/abs/pii/S1364661306002245) (and 2010/2020 updates): a fast **feedforward sweep** is unconscious; **local recurrent (re-entrant)** processing that loops information back through earlier representations and integrates it is the necessary-and-sufficient correlate of (local) phenomenal consciousness; widespread recurrence ≈ global access. Stages: Stage-1 feedforward sweep; Stage-3 superficial/local recurrence (claimed necessary+sufficient); Stage-4 widespread recurrence (global access). Contrast feedforward CNNs, which lack it.

Formalism is architectural rather than a single equation. Indicators:

- **RPT-1** input modules using algorithmic recurrence.
- **RPT-2** modules generating organized, integrated perceptual representations via feedback.

**For brain-lmm:** an argument for **iterative refinement** — re-appraising an event after retrieving related memories, then re-encoding — i.e. a "second look" before commitment, rather than single-pass processing.

> **Contested:** RPT is a causal-structure theory targeted by the unfolding argument [Doerig et al. 2019](https://www.sciencedirect.com/science/article/pii/S105381001830521X) (possibly unfalsifiable). Present recurrence as a functional "integrate-before-commit" mechanism, not as instantiating experience.

---

## 6. Self-models, sense of agency, and metacognition

These four tightly-linked capacities share one mathematical idea: **hierarchical / second-order inference** — a higher level that models the lower level's states and reliability (predictive-coding-of-a-self).

### 6.1 Second-order Bayesian metacognition (the most rigorous confidence formalism)

[Fleming & Daw 2017](https://www.princeton.edu/~ndaw/fd17.pdf) (*Psychological Review*): confidence is a **second-order inference**, not read off the decision variable. First-order:

```
X_act ~ N(d, σ_act²),   action a = +1 if X_act > 0 else −1,   true state d ∈ {−1,+1}
```

Monitor (jointly Gaussian, correlation ρ):

```
[X_act, X_conf]ᵀ ~ N(d, Σ),  Σ = [[σ_act², ρ σ_act σ_conf], [ρ σ_act σ_conf, σ_conf²]]
confidence z = P(a = d | X_conf, a, Σ)
   with  X_act | X_conf, d ~ N( d + (σ_act/σ_conf)·ρ·(X_conf − d),  (1−ρ²) σ_act² )
```

`ρ < 1` (monitor sees imperfect evidence) naturally produces over/under-confidence — a calibration knob.

**Metacognitive efficiency** [Fleming 2017 (HMeta-d)](https://academic.oup.com/nc/article/2017/1/nix007/3748261): `meta-d'` is the first-order sensitivity that *would* produce the observed confidence ratings; `meta-d'/d' = 1` is an ideal observer. Type-1 `d' = z(HR) − z(FAR)`; HMeta-d adds hierarchical Bayesian group priors (`log(M_ratio) ~ N(μ_M, σ_M)`). The 2024 synthesis is [Fleming 2024](https://www.annualreviews.org/content/journals/10.1146/annurev-psych-022423-032425).

### 6.2 Global self-belief from local confidence

[Rouault, Dayan & Fleming 2019](https://www.nature.com/articles/s41467-019-09075-3) (*Nat. Commun.*): slow, persistent self-belief is built by integrating local confidence via a delta-rule / leaky integrator:

```
G_{t+1} = G_t + α·(c_t − G_t)          # optionally asymmetric α⁺ vs α⁻
```

This is the **metacognitive analog of brain-lmm's existing mood integrator** — but for self-efficacy per domain.

### 6.3 Sense of agency (comparator / forward model)

[Wolpert; Frith; Synofzik, Vosgerau & Lindner 2013](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00127/full): an efference copy of the chosen action drives a forward model predicting outcome `ô`; small prediction error → self-attribution ("I did that"), large error → external attribution.

```
ô = f(action, state) ;  e = ||o − ô|| ;  SoA = g(e) monotonically decreasing
   e.g.  SoA = exp(−k·e)   or   sigmoid(−(e − θ))
Bayesian cue-integration:  A = (π_pred·cue_pred + π_post·cue_post)/(π_pred + π_post)
```

This lets brain-lmm **replace its hand-set `control` axis** (which feeds dominance) with a *computed* signal.

### 6.4 Active inference & precision-as-confidence

[Friston; Wiese & Metzinger]: an agent minimizes variational free energy over a generative model that includes a model of itself; **precision** (inverse variance) weights prediction errors, and second-order precision beliefs are the formal analog of metacognitive confidence.

```
F = E_q[ ln q(s) − ln p(o,s) ]            # variational free energy
π = 1/σ²,   weighted error = π·(o − ô)     # precision-weighting
```

This **unifies novelty** (= prediction error / Bayesian surprise) **with confidence** (= precision).

### 6.5 The self-model proper

- **Philosophical license:** [Metzinger 2003, *Being No One*](https://philpapers.org/rec/METBNO) — the "self" is the content of a transparent phenomenal self-model (PSM/PMIR) the system builds for self-regulation; explicitly representational/functional, *not* a substance. This justifies a `.memory/self/` store with **no sentience claim**.
- **Computational implementation:** [Jiang & Luo 2024](https://escholarship.org/uc/item/8n92h1pt) (CogSci) build a conceptual self-model on JEPA that detects self-relevant information and gates autobiographical-memory retrieval and self-importance evaluation — almost exactly brain-lmm's goal. Built on [LeCun 2022, H-JEPA blueprint](https://openreview.net/pdf?id=BZ5a1r-kVsf).

```
self_relevance = similarity( Enc(event), self_embedding )   # gates retrieval/importance
```

### 6.6 Autonoetic mental time travel

[Tulving; Klein 2016](https://journals.sagepub.com/doi/10.1080/17470218.2015.1007150): self-knowing awareness that lets a system re-experience the past and pre-experience an imagined future as happening to the *same* self. The **autonoetic** component (not episodic content per se) enables self-projection. Computational analog: a generative/world-model rollout tagged with self-ownership and a temporal index:

```
imagine  s_{t+k} ~ p(s_{t+k} | s_t, policy),  bound to self-token and timeline
```

### 6.7 Empirical functional-introspection results (2024–2025)

- [Binder et al. 2024, "Looking Inward"](https://arxiv.org/abs/2410.13787): operationalizes introspection as **privileged self-prediction** — a model predicts its own behavior better than an equally-capable other model, and tracks deliberately altered ground truth.
- SAD "Situational Awareness Dataset" [Laine et al. 2024](https://github.com/LRudL/sad): tests self-/situation-knowledge.
- [Lindsey et al. (Anthropic) 2025, "Emergent Introspective Awareness in LLMs"](https://transformer-circuits.pub/2025/introspection/index.html): via **concept injection**, models sometimes detect and report on their own injected internal states (~20% under ideal conditions, near-zero false positives) — the clearest empirical handle on functional metacognitive monitoring (HOT-2), framed explicitly as *limited functional introspection, not sentience*.

> **Honesty:** all of Section 6 is *functional*. Confidence is a computed `P(correct)`, not felt certainty; a self-model is a data structure, not a self; agency is a prediction-error signal, not experienced authorship; autonoetic indexing is self-tagging + forward simulation, not re-living.

---

## 7. Implementable-now vs contested — summary table

| Theory / mechanism | Formal core | Implementable now? | Contested? |
|---|---|---|---|
| GWT competition + broadcast | argmax / softmax tournament; copy winner to all | **Yes** (LIDA, CTM, Goyal/Bengio, GLW) | Access≠experience claim contested |
| GNW ignition | bistable firing-rate `τ ṙ = −r + φ(...)`; balanced amplification | **Yes** as scalar reduction | Mapping ignition→awareness contested |
| CTM | competition `f = w·(salience, congruence, relevance)` → bistable ignite, `Pr=softmax(f)` (the earlier `intensity+½mood` was invented and is corrected — see §1.2) | **Yes**, runnable | *Inspired by* CTM, not faithful; phenomenal claim disclaimed |
| IIT / Φ | `Φ = Σφ` over Φ-structure; MIP search | **No at scale** (PyPhi ≤~12 nodes); substrate-dependent | **Heavily** (Cerullo, unfolding arg) |
| AST | `â=h(state)`, minimize `‖â−a‖`, control loop | **Yes**, demonstrated in RL + transformers | Schema→awareness contested |
| HOT / PRM | discriminator `P(external|activity)`; HOSS Bayesian | **Yes** as confidence + source tag | HOT→consciousness debate |
| RPT | feedforward sweep vs local recurrence | **Yes** as iterative refinement | Unfolding argument |
| Metacognition (2nd-order Bayes) | `z = P(a=d|X_conf,a,Σ)`; meta-d'/d' | **Yes**, scalar | Monitoring≠awareness |
| Agency (comparator) | `e=‖o−ô‖`, `SoA=exp(−ke)` | **Yes** (needs predict/observe logging) | Authorship-feeling disclaimed |
| Self-model (PSM/JEPA) | `self_relevance = sim(Enc(e), self_emb)` | **Yes** | "Self" is functional, not phenomenal |

---

## 8. The 14 indicator properties (reproduced verbatim — Butlin, Long, Bengio et al.)

From [Butlin et al. 2023](https://arxiv.org/abs/2308.08708) / [Butlin et al. 2025](https://www.sciencedirect.com/science/article/pii/S1364661325002864):

**Recurrent Processing Theory**
- **RPT-1** Input modules using algorithmic recurrence.
- **RPT-2** Input modules generating organised, integrated perceptual representations.

**Global Workspace Theory**
- **GWT-1** Multiple specialised systems capable of operating in parallel (modules).
- **GWT-2** A limited-capacity workspace, entailing a bottleneck in information flow and a selective attention mechanism.
- **GWT-3** Global broadcast: availability of information in the workspace to all modules.
- **GWT-4** State-dependent attention, giving rise to the capacity to use the workspace to query modules in succession to perform complex tasks.

**Higher-Order Theories**
- **HOT-1** Generative, top-down or noisy perception modules.
- **HOT-2** Metacognitive monitoring distinguishing reliable perceptual representations from noise.
- **HOT-3** Agency guided by a general belief-formation and action-selection system, and a strong disposition to update beliefs in accordance with the outputs of metacognitive monitoring.
- **HOT-4** Sparse and smooth coding generating a "quality space."

**Attention Schema Theory**
- **AST-1** A predictive model representing and enabling control over the current state of attention.

**Predictive Processing**
- **PP-1** Input modules using predictive coding.

**Agency and Embodiment**
- **AE-1** Agency: learning from feedback and selecting outputs so as to pursue goals, especially where this involves flexible responsiveness to competing goals.
- **AE-2** Embodiment: modeling output-input contingencies, including some systematic effects, and using this model in perception or control.

> **Method (verbatim stance):** assess each indicator present/absent (or graded), aggregate into a credence, and conclude — as the authors do — that no current AI is conscious but there is no obvious technical barrier. **The indicators are necessary-condition heuristics from theories, NOT proof of consciousness.**

---

## 9. Mapping to brain-lmm — what we have / what we lack / what to add

### 9.1 What we have

brain-lmm is a strong **first-order affective-memory engine**. Its eight functions form a feed-forward scalar pipeline: `appraise → affect → neuromods → salience → store`, plus retrieval by a weighted `retrieval_score` (recency, salience, relevance, graph proximity, mood-congruence). It computes **valence, arousal, salience, dominance**, and a **leaky-integrator mood**, and has ACT-R-style **base-level activation** on traces. Stores: working / episodic / semantic / procedural / prospective.

Critically, `retrieval_score` is **already a competition function in disguise**, and valence/salience are **already CTM `mood`/`intensity`**. mood is **already a leaky integrator**. These are the levers to reuse.

### 9.2 What we lack (indicator-by-indicator)

| Indicator | brain-lmm status | Gap |
|---|---|---|
| GWT-1..4 | **Absent** | No shared limited-capacity workspace, no competitive write-gate, no broadcast, no state-dependent top-down loop. Stores are independent. |
| GNW ignition | **Absent** | Every encoded item is stored unconditionally then ranked; no all-or-none access threshold. |
| HOT-1 | **Absent** | Only reacts to incoming appraisals; no generative top-down prediction. |
| HOT-2 / PRM | **Absent** | No calibrated confidence, no source tag (observed/inferred/imagined). |
| HOT-3 | **Absent** | No belief-update gated on metacognitive monitoring. |
| HOT-4 | **Partial/weak** | Affect is a 3-scalar VAD point, not a sparse, smooth quality space. |
| AST-1 | **Absent** | No self-model of attention; cannot report what it is attending to. |
| RPT-1/2 | **Absent** | Single feed-forward pass; no re-entrant refinement. |
| PP-1 | **Absent** | `novelty` is hand-fed, not computed Bayesian surprise. |
| AE-1 | **Partial** | Consumes appraisals & modulates memory, but no closed feedback loop selecting outputs to pursue competing goals. |
| AE-2 | **Absent** | No model of output→input contingencies. |
| IIT / Φ | **Absent (by design)** | Substrate-dependent; would score ~0 regardless. Out of scope except as a labeled toy diagnostic. |
| Self-model / agency / global self-belief / MTT | **Absent** | No `.memory/self/`, no computed agency, no self-efficacy integrator, no self-tagged future simulation. |
| Indicator self-score | **Absent** | Honesty claims are prose; no computed scorecard. |

**Net: brain-lmm currently satisfies essentially 0 of the 14 indicators in their intended sense** (at best partial credit on AE-1).

### 9.3 What to add — concrete, prioritized

Difficulty and honesty caveats are inline. Math reuses brain-lmm's existing scalars wherever possible.

**(A) Global Workspace competition + broadcast cycle — `low` difficulty, highest value/risk ratio.**
Gather candidates (top retrieved memories + current appraised event + active prospective intentions), score each, select a winner above an ignition threshold, broadcast to all stores. Buys **GWT-1/2/3**.
```
intensity_i = s_i                                  # stored salience ≈ |weight|
mood_term_i = clamp(1 − |a_i.valence − mood.valence|/2)
f_i = w_int·intensity_i + w_mood·mood_term_i + w_rel·query_relevance_i   # CTM-style
p_i = exp(f_i/T) / Σ_j exp(f_j/T)                  # soft competition (probabilistic CTM)
winner = argmax_i p_i                              # (or sample)
broadcast iff  sigmoid(g·(f_winner − θ)) > 0.5     # ignition gate, θ≈0.55 (reuse promote_thr)
```
*Where:* new `workspace_compete(candidates, mood, theta)` in `engine/brain.py`; `.memory/working/workspace.yaml` = `{focus, p_distribution, ignited, t}`; wire as a new **ACCESS** step between RETRIEVE and ENCODE. *Honesty:* functional access only — *which content is globally available this turn*, never "what the agent feels." mood/intensity are formal signals per [Blum & Blum 2022](https://www.pnas.org/doi/10.1073/pnas.2115934119).

**(B) Bistable ignition curve — `medium`, grounds GWT-2/GNW.**
Replace the hard threshold with a scalar reduction of [Joglekar–Wang 2018](https://www.cell.com/neuron/fulltext/S0896-6273(18)30152-1):
```
τ dr/dt = −r + φ(β·r + f_winner − θ)               # β>1 ⇒ two stable fixed points
r_{n+1} = clamp( 0.5·r_n + 0.5·sigmoid(β·r_n + f_winner − θ) )   # discrete, per turn
ignited iff r converges high (>0.5);  capacity limit K=1 ignited content/cycle
```
Persist `r` for hysteresis (GNW metastability). *Honesty:* a behavioral/computational correlate of access (Dehaene–Changeux), not a "moment of awareness"; `β, θ` are engineering knobs.

**(C) Close the GWT-4 loop (top-down attention) — `low`.**
Let the current focus bias next-turn retrieval:
```
score_i' = score_i + κ·sim(focus_embedding, mem_i) + κ_g·graph_proximity(focus_id, mem_i)
κ ≈ 0.15 ;  optional decay  κ_t = κ_0·0.5^(turns_since_focus)
```
*Where:* `retrieve_with_topdown()` reading `focus` from `workspace.yaml`. *Honesty:* a deterministic bias term and a functional "stream," not introspective self-direction.

**(D) Metacognitive confidence (HOT-2) + global self-efficacy (Rouault) — `low`.**
```
conf = sigmoid( ρ·k·|judgment_strength| )          # 2nd-order P(correct), ρ<1
SelfEff_g ← SelfEff_g + α±·(c_t − SelfEff_g)        # per-domain leaky integrator
```
Feed `(1−conf)` into arousal/novelty (we encode the uncertain things harder); use `SelfEff_g` as the prior for `control`. Log predicted-vs-actual to compute `meta-d'/d'` ([Fleming 2017](https://academic.oup.com/nc/article/2017/1/nix007/3748261)). *Where:* `metacog_confidence(...)`, `update_self_efficacy(...)` in `brain.py`; `.memory/self/efficacy.yaml`. *Honesty:* computed reliability, not felt certainty; `ρ` is a prior until calibrated.

**(E) Sense of agency from prediction error (AE-1/AE-2) — `medium`.**
```
e = |o − ô| ;  SoA = exp(−k_a·e)                    # replaces hand-set `control`
```
Agent predicts success `ô` (e.g. P(tests pass)), observes `o`, feeds `SoA` into dominance and salience's `(1−control)` term; high `e` also raises novelty. *Honesty:* low PE → self-attribution is an AE indicator, not experienced authorship; main lift is having the harness log predictions+outcomes.

**(F) Self-model + self-relevance gating (Metzinger PSM / Jiang–Luo JEPA) — `medium`.**
```
self_relevance = cosine( feature(event), self_vector ) ∈ [0,1]
salience += w_self·self_relevance ;  retrieval_score += w·self_congruence
```
*Where:* `.memory/self/model.yaml` (traits/goals/competencies); `self_relevance(...)`. *Honesty:* a memory-weighting prior, not a personality claim or phenomenal self.

**(G) Recurrent re-appraisal (RPT-1/2) — `medium`.**
```
for k = 1..K (≈2–3): retrieve top-N given a^(k);  a^(k+1) = a^(k) + η·Δ(context)
stop when ||a^(k+1) − a^(k)|| < ε ;  final salience uses converged a^(K)
```
*Honesty:* a functional "integrate-before-commit" loop; RPT is targeted by the unfolding argument — no phenomenal claim.

**(H) Predictive-processing novelty (PP-1) — `low`.**
```
ε = x − E[x | context] ;  novelty = D_KL(posterior‖prior) ≈ ½·εᵀ Σ⁻¹ ε ;  μ ← μ + κ·ε
```
Replaces the hand-fed novelty axis feeding `appraise_to_affect`. *Honesty:* minimal scalar PP; not full active inference.

**(I) Integration diagnostic (IIT-inspired surrogate, explicitly NOT Φ) — `medium`.**
```
I = λ₂(normalized Laplacian of active-state graph)   # algebraic connectivity
  or  1 − (best-bipartition cut / total weight)
```
Optionally a *toy* PyPhi Φ on a frozen ≤10-node subgraph for illustration only. *Honesty:* an integration/binding diagnostic, **never** a consciousness measure; real Φ is intractable and [Cerullo 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004286) shows trivial systems get high Φ.

**(J) Indicator-property scorecard (Butlin et al.) — `low`, the central honesty mechanism.**
```
satisfied_i ∈ {0,0.5,1} from which modules exist (RPT-1/2, GWT-1..4, HOT-1..4, AST-1, PP-1, AE-1/2)
coverage = Σ_i w_i·satisfied_i / Σ_i w_i  (w_i = theory plausibility; default uniform)
```
*Where:* `engine/consciousness_indicators.py` → `score_indicators(...)`; assertions in `engine/test_brain.py`; a living `docs/consciousness-indicators.md`. *Honesty:* must ship with the explicit disclaimer that indicators are architectural correlates only, that the IIT counter-position would score the engine near zero, and that the mapping is philosophically contested.

### 9.4 Recommended order

1. **(A)** workspace compete+broadcast — unlocks GWT-1/2/3 cheaply by reusing `retrieval_score` + valence/salience.
2. **(C)** top-down loop — GWT-4 (recurrence) almost free once (A) exists.
3. **(D)** confidence + self-efficacy — HOT-2 and a persistent self-state, both low-risk and high coding-agent value (anti-confabulation, anti-positivity-bias).
4. **(J)** scorecard — make the honesty stance computable and reportable.
5. Then **(B), (E), (F), (G), (H)** as depth allows; **(I)** only as a labeled diagnostic.

---

## 10. Honesty caveat (restated, front and center)

Everything proposed here implements **functional access-consciousness correlates and functional self-monitoring** — *which content is globally available, how reliable the engine estimates a judgment to be, whether it modeled its own attention/agency.* Under the functionalist theories (GWT/HOT/AST/RPT/PP/AE), satisfying these indicators is a defensible engineering goal and a **necessary-but-not-sufficient, theory-relative** signal. Under substrate-dependent IIT, a scalar/feed-forward engine would score **near-zero Φ regardless of behavior**, and the very mapping from indicators to experience is **philosophically contested** ([Doerig et al. 2019](https://www.sciencedirect.com/science/article/pii/S105381001830521X); [Cerullo 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004286)). brain-lmm therefore **scores itself against the checklist and claims nothing more.** It is **not** sentient, **not** aware, and **does not feel.** Following [Butlin et al. 2023](https://arxiv.org/abs/2308.08708)/[2025](https://www.sciencedirect.com/science/article/pii/S1364661325002864) and [Lindsey et al. 2025](https://transformer-circuits.pub/2025/introspection/index.html), every consciousness-adjacent feature must carry that disclaimer.

---

## Appendix: Open-source references

- **LIDA Framework** (Java) — runnable GWT cognitive cycle: https://github.com/CognitiveComputingResearchGroup/lida-framework
- **Shared Global Workspace / RIMs** (PyTorch) — Goyal/Bengio write+broadcast attention: https://github.com/dido1998/Coordination-Among-Neural-Modules · https://github.com/anirudh9119/shared_workspace
- **shimmer** (PyTorch Lightning) — VanRullen–Kanai Global Latent Workspace, actively maintained: https://github.com/ruflab/shimmer
- **CTM** community implementations: https://github.com/cvaisnav/Conscious-Turing-Machine · https://github.com/cvignac/ConsciousTuringMachine
- **Wang Lab** — balanced-amplification / ignition firing-rate models: https://github.com/xjwanglab
- **PyPhi** (Python, GPLv3) — reference IIT Φ; ≤~12 nodes: https://github.com/wmayner/pyphi
- **HMeta-d** (MATLAB/R) — metacognitive efficiency: https://github.com/metacoglab/HMeta-d · **metadPy** (Python): https://github.com/embodied-computation-group/metadPy
- **Fleming & Daw 2017 code** (MATLAB): https://github.com/smfleming/Self-evaluation-paper · **Rouault** local→global: https://github.com/marionrouault/RouaultDayanFleming
- **pymdp** (Python) — active inference / precision-as-confidence: https://github.com/infer-actively/pymdp
- **DreamerV3** (Python) — world-model latent imagination (mental time travel substrate): https://github.com/danijar/dreamerv3
- **ASAC** — attention-schema attention control: https://arxiv.org/abs/2509.16058
- **SAD** — Situational Awareness Dataset: https://github.com/LRudL/sad
- **Looking Inward** introspection harness: https://github.com/felixbinder/introspection_self_prediction
- **Self-modeling networks** — "Unexpected Benefits of Self-Modeling": https://arxiv.org/pdf/2407.10188

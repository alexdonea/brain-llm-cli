# Open-Source Landscape — Cognitive Architectures, LLM Memory, Affect & Consciousness Code

> **Audience.** A technical builder extending **brain-lmm**.
>
> **Honesty stance (non-negotiable, applies to every section).** brain-lmm models the *function* of memory and affect — the input→state→output computations and their behavioral/learning correlates — **never the felt experience**. Throughout this document, "emotion," "fear," "awe," "pain," "interoception," and "consciousness" denote *functional/computational analogs*. We make **no claim of sentience or phenomenal experience.** Where a theory touches consciousness, we use it strictly at the *indicator-property* level of [Butlin et al. 2023](https://arxiv.org/abs/2308.08708) (derive architectural indicators, update credence, never assert experience). The current engine already states this in `engine/brain.py` lines 1–20.
>
> **What brain-lmm is today.** `engine/brain.py` is ~8 pure-stdlib scalar functions: `appraise_to_affect` (OCC→PAD), `neuromods_from` (NE/DA/ACh/cortisol gains), `salience` (McGaugh arousal-gated encoding), `base_level_activation` (ACT-R `B_i = ln(Σ_k (now−t_k)^−d)`), `retention` (importance-modulated stretched-exponential forgetting), `retrieval_score` (recency+salience+relevance+graph-proximity+mood-congruence), `update_mood` (leaky integrator to a `baseline = Affect(0.0, 0.10, 0.50)`), and `consolidation_plan` (CLS sleep promote/forget). Stores are file-based (episodic/semantic/procedural/prospective/affect). There is no autonomous loop; an external LLM calls these functions.

---

## 1. Orientation — where brain-lmm sits

brain-lmm is a memory-and-affect **physics with no engine running it**. It has correctly imported the *memory-dynamics math* of classical cognitive architectures (activation, decay, consolidation) and the *retrieval scoring* of modern LLM-agent memory systems, plus a *dimensional-affect + neuromodulation* core that is genuinely ahead of the field. It lacks almost all of the *control, learning, motivation, workspace, social, and self-model* machinery — which is precisely where emotion and functional consciousness live in every architecture surveyed.

The three clusters below map the landscape:

1. **Cognitive architectures** — full theories of the fixed structure of cognition (control loop, learning, workspace, motivation).
2. **LLM-agent memory systems** — retrieval, forgetting, consolidation, belief revision over an LLM.
3. **Affect / consciousness code** — discrete-emotion scorers, appraisal engines, global-workspace primitives, and the honesty scaffold.

---

## 2. Master comparison table

| Project | What it implements | Overlaps brain-lmm | What to borrow |
|---|---|---|---|
| **ACT-R** (pyactr, python_actr) | Declarative memory: base-level + spreading activation; production system; subsymbolic retrieval latency/probability; utility learning | **Strong** — brain-lmm already uses `B_i = ln(Σ_k (now−t_k)^−d)` verbatim | Spreading-activation term `Σ_j W_j S_ji`; activation→latency/probability map |
| **Soar** | Decision cycle, impasses, chunking (rule learning), RL over operators, appraisal-as-intrinsic-reward | Moderate — shares OCC appraisal axes | Decision cycle; RL `Q(s,o)←Q+α[r+γmaxQ′−Q]`; feed appraisal as reward |
| **LIDA** | Canonical computational GWT: codelets, coalition competition, ~10 Hz global broadcast; feeling-nodes that bias attention + scale learning rate | Moderate — brain-lmm scales encoding by arousal (McGaugh), LIDA generalizes it | **Best single template** for both workspace loop AND discrete emotion |
| **CLARION** | Implicit/explicit dual process; Motivational Subsystem (drives); Metacognitive Subsystem (monitors/regulates) | Weak — no drives/metacog in brain-lmm | MS = drives from mood; MCS = self-monitor that adjusts thresholds |
| **Nengo / SPA / Spaun** | Spiking NEF representation; Semantic Pointer Architecture (circular-convolution binding); 2.5M-neuron Spaun | Weak — different substrate | SPA binding `C = A ⊛ B` for compositional semantic memory |
| **Leabra / axon (emer)** | Point-neuron activation, XCAL learning; PBWM prefrontal-BG gating; PVLV phasic-dopamine reward-prediction | Weak | **PVLV RPE** `δ ≈ reward − expected` — the missing reward-prediction error |
| **Sigma** | Factor-graph unification; auto-computed appraisals of surprise/desirability/familiarity → attention | Weak | Define `novelty` as Bayesian surprise `KL(posterior‖prior)` |
| **OpenCog Hyperon + OpenPsi/ECAN** | AtomSpace hypergraph; OpenPsi (Dörner Psi modulators+urges); ECAN economic attention (STI/LTI) | Moderate — salience ≈ STI; neuromods ≈ Psi modulators | OpenPsi modulator+urge dynamics; ECAN conserved-attention spreading = a workspace |
| **CoALA** | Recasts LLM agent as CA: modular memory + internal/external action split + decision loop | **Strong (target shape)** — brain-lmm has the stores, lacks the loop | The decision loop + internal/external action taxonomy |
| **Stanford Generative Agents** | Memory stream `recency·importance·relevance`; reflection-tree synthesis | **Strong** — direct ancestor of `retrieval_score` | Reflection/insight synthesis at `/sleep` |
| **Mem0** | Extract→(ADD/UPDATE/DELETE/NOOP) belief revision over vector+graph store | Weak — brain-lmm is append-only | Contradiction-resolution reconcile step |
| **Zep / Graphiti** | Bi-temporal knowledge graph (valid-time + transaction-time), fact invalidation | Weak — brain-lmm has uni-temporal `valid_from` | Bi-temporal edges + invalidation for temporal reasoning |
| **Letta (MemGPT)** | OS-style virtual context paging; self-editing core memory | Weak — brain-lmm WM is a disposable ~7-item buffer | Salience-ranked eviction under context pressure |
| **HippoRAG / HippoRAG 2** | LLM=neocortex, KG=index, Personalized PageRank=hippocampal pattern completion, node-specificity=DG separation | **Strong conceptually** — same CLS framing; brain-lmm has flat `graph_proximity` | Replace scalar `graph_proximity` with PPR over semantic graph |
| **A-MEM** | Zettelkasten notes; auto keywords/links; retroactive memory evolution | Moderate | Continuous learned consolidation; richer note metadata |
| **MemoryBank** | Ebbinghaus forgetting `R = e^(−t/S)`, recall does `S+1, t←0` (spacing effect) | **Strong** — only other Ebbinghaus-grounded system; brain-lmm `retention()` is more expressive | Recall-strengthening (spacing effect) on the decay curve |
| **Cognee** | ECL (Extract-Cognify-Load) → self-hosted KG + vector store | Weak | Reference pipeline if semantic store outgrows hand-curated YAML |
| **Memary** | Memory Stream + Entity Knowledge Store (per-entity count + recency) | Moderate — ACT-R activation over graph nodes | Apply `base_level_activation` to semantic-graph nodes, not just episodes |
| **Cognitive Weave** | Spatio-temporal resonance graph; autonomous "insight aggregates" | Weak | Generative consolidation (synthesize abstractions) |
| **EmoLLMs** | Fine-tuned LLMs: 11-label categorical emotion + sentiment + 0–1 intensity regression | None — brain-lmm has no discrete emotions | Drop-in discrete-emotion + intensity scorer |
| **FAtiMA Toolkit** | OCC-appraisal agent emitting named OCC emotions (fear/joy/distress/hope…) + decay | Moderate — same OCC inputs | Categorical-emotion readout from existing appraisal axes |
| **EMA (Gratch & Marsella)** | Appraisal dynamics + coping loop over a plan-based causal interpretation | Moderate — shares controllability axis | Controllability-keyed coping (problem- vs emotion-focused) |
| **PsychSim** | Recursive decision-theoretic ToM; social state (trust/support/power) | None — brain-lmm has no second mind | Persistent user-relationship/ToM model schema |
| **PyPhi** | Computes IIT Φ over tiny discrete systems | None (intractable; contested) | **Cite only** — do NOT implement; honesty bound |
| **CTM toy / Shared-Workspace Transformer / Global Latent Workspace** | Competition-for-broadcast bottleneck (GWT/CTM) | None | Math template for an arbitration/broadcast loop |
| **ASAC** | Attention Schema Theory via VQ-VAE schema codebook | None | Model of own focus (which memories/goals dominate) |
| **pymdp** | Active inference: free energy, Bayesian surprise, expected free energy | None | Principled `novelty` = Bayesian surprise; curiosity drive |
| **hBayesDM** | Hierarchical-Bayes RL with asymmetric learning rates + prospect-theory `λ` | None | Loss aversion + `η⁻>η⁺` parameter values |
| **Dreamer / World Models / brain-inspired-replay / PAD** | Offline generative simulation; generative replay; per-stage Wake/NREM/REM losses | Weak — brain-lmm `/sleep` is copy-and-forget | Dream-as-regularization; generative replay vs forgetting |

**Closest to brain-lmm overall:** (1) **Stanford Generative Agents** (literal ancestor of `retrieval_score`), (2) **ACT-R** (shared activation math), (3) **MemoryBank** (shared Ebbinghaus forgetting), (4) **LIDA** (best template for the consciousness + discrete-emotion goals), (5) **CoALA** (the target architecture for adding a control loop). **HippoRAG** is the closest *neuro-aligned* memory competitor.

---

## 3. Cognitive architectures

### 3.1 ACT-R — declarative activation [Anderson et al. 2004]

The definitive statement of ACT-R [[Anderson et al. 2004](https://doi.org/10.1037/0033-295X.111.4.1036)]. Each chunk's activation:

```
B_i = ln( Σ_k (now − t_k)^(−d) )           # base level (recency + frequency)
A_i = B_i + Σ_j W_j S_ji + ε               # + spreading activation + noise
P(retrieve) = 1 / (1 + e^((τ − A_i)/s))    # retrieval probability
latency     = F · e^(−A_i)                 # time to remember
```

Implementations: [pyactr](https://github.com/jakdot/pyactr), [python_actr](https://github.com/CarletonCognitiveModelingLab/python_actr), official Lisp at [act-r.psy.cmu.edu](https://act-r.psy.cmu.edu).

**Maps to brain-lmm:** `base_level_activation()` implements `B_i` exactly. **Missing:** the spreading term `Σ_j W_j S_ji` (context priming semantic neighbors) and the `latency`/`P(retrieve)` mappings — both cheap additions that make recall context-sensitive and yield a "time-to-remember" signal.

### 3.2 Soar — decision cycle, chunking, RL, appraisal-as-reward [Laird 2012/2022]

Canonical reference [[Laird 2022, arXiv:2205.03854](https://arxiv.org/abs/2205.03854)]; emotion work [[Marinier & Laird 2009](https://doi.org/10.1016/j.cogsys.2008.03.003); Gratch & Marsella EMA]. Source: [github.com/SoarGroup/Soar](https://github.com/SoarGroup/Soar).

```
Q(s,o) ← Q(s,o) + α[ r + γ·max_o′ Q(s′,o′) − Q(s,o) ]   # RL over operators
r = f(novelty, goal-relevance/conduciveness, control/coping)  # intrinsic reward from appraisal
chunking = explanation-based generalization over the impasse trace
```

**Maps to brain-lmm:** brain-lmm computes appraisals (novelty, valence, goal_relevance, control) then **discards them after salience**. Soar's high-value move: feed appraisal as *intrinsic reward* into a TD learner so the agent learns which playbooks pay off — closing brain-lmm's total absence of reward learning.

### 3.3 LIDA — computational Global Workspace Theory + feeling-nodes [Franklin et al. 2014]

LIDA [[Franklin et al. 2014](https://doi.org/10.1109/TAMD.2013.2277589); [Baars & Franklin 2009](https://doi.org/10.1142/S1793843009000050)] is the reference implementation of GWT. Code (Java): [CCRG framework](https://ccrg.cs.memphis.edu/framework.html).

```
cognitive cycle ≈ 10 Hz (100 ms): understand → attend/broadcast → act-and-learn
coalition salience = Σ activations × attention-codelet boost
winning coalition is broadcast globally; broadcast triggers learning everywhere
learning rate λ scaled by feeling-node activation   # emotional salience learned faster (McGaugh analog)
```

**Maps to brain-lmm:** the single best template for *both* hard goals. (a) A broadcast+bottleneck loop yields honest GWT indicator properties. (b) Feelings as nodes layered on the VAD point give discrete emotions that bias attention and scale learning — generalizing brain-lmm's existing arousal-scaled encoding.

### 3.4 CLARION — dual process + Motivational & Metacognitive subsystems [Sun 2007/2009]

[[Sun 2007](https://doi.org/10.1080/09528130701191560); [Sun 2009 (motivation)](https://homepages.hass.rpi.edu/rsun/folder-files/sun-cogcomp2009.pdf)]. Four subsystems: ACS (action), NACS (declarative), MS (drives), MCS (metacognition).

```
drive strength ds = baseline_gain · deficit · stimulus            # e.g. ds = 0.95·max(0.30·deficit, deficit·stimulus)
MCS sets ACS/NACS parameters (temperature, learning rate) from drive/affect state
```

**Maps to brain-lmm:** brain-lmm has **no metacognition and no drives**. CLARION's MCS is the model for a self-monitor that raises the consolidation threshold when overloaded; MS is the model for turning mood into goal-generating drives.

### 3.5 OpenPsi / MicroPsi — Dörner's Psi affect engine [Bach 2009/2015; Cai/Goertzel 2013]

[[Cai, Goertzel et al. 2013](https://doi.org/10.1016/j.engappai.2012.07.013); [Bach 2015](https://agi-conf.org/2015/wp-content/uploads/2015/07/agi15_bach.pdf)]. Code: [opencog/openpsi](https://github.com/opencog/openpsi), [micropsi2](https://github.com/joschabach/micropsi2).

```
modulators: arousal, valence, dominance, selection_threshold, resolution_level, securing_rate
urges: competence, certainty/uncertainty-reduction, affiliation (+ physiological)
arousal ∝ Σ urge intensity ; valence ∝ (rate of urge satisfaction − frustration)
selection_threshold ↑ with arousal (persistence) ; resolution_level ↓ with arousal (coarser/faster)
```

**Maps to brain-lmm:** brain-lmm's `Neuromods` (NE/DA/ACh/cortisol) are a 4-element special case of Psi modulators. OpenPsi shows how to (a) derive discrete affect from modulator+urge dynamics and (b) **close the loop** so affect changes *how* the system processes (resolution, persistence) — not just memory weighting.

### 3.6 OpenCog ECAN — economic attention (a quantitative workspace)

```
ΣSTI conserved (fixed "funds" pool)
STI_i ← STI_i + spread(neighbors) − rent
attentional focus = { i : STI_i > AF_boundary }
LTI controls forgetting from the AtomSpace
```

**Maps to brain-lmm:** salience is a per-memory STI analog, but there is **no spreading, no conserved budget, no attentional-focus set**. ECAN is the formal recipe for a capacity-limited workspace over `.memory/semantic/graph.yaml`.

### 3.7 Nengo / NEF / Semantic Pointer Architecture + Spaun [Eliasmith 2012/2013]

[[Eliasmith et al. 2012, Science](https://doi.org/10.1126/science.1225266)]. Code: [nengo](https://github.com/nengo/nengo), [nengo-spa](https://github.com/nengo/nengo-spa), [spaun2.0](https://github.com/xchoo/spaun2.0).

```
NEF representation: x̂ = Σ_i a_i(x) d_i        # decoders found by least squares over tuning curves
SPA binding:   C = A ⊛ B   (circular convolution) ;  unbind:  A ≈ C ⊛ B⁻¹
```

**Maps to brain-lmm:** the path to neural grounding if ever desired. More practically, circular-convolution binding gives **compositional** (concept⊛role) semantic memory without a neural net — a hyperdimensional upgrade to the flat fact store.

### 3.8 Leabra / axon — PVLV reward & PBWM working memory [O'Reilly & Frank 2006]

[[O'Reilly & Frank 2006](https://doi.org/10.1162/089976606775093909)]. Code: [emer/leabra](https://github.com/emer/leabra), [emer/axon](https://github.com/emer/axon).

```
PVLV dopamine:  δ ≈ reward − expected      # reward-PREDICTION error (RPE)
XCAL learning:  dwt = f(s·r_s, s·r_m)      # BCM-like, STDP-derived
PBWM:  BG Go/NoGo gating of PFC working memory, learned by RL
```

**Maps to brain-lmm:** `neuromods_from` sets `da = clamp(reward)` — a *static* value, **not** a prediction error. PVLV is the canonical formalism for the missing RPE: DA should signal `(actual − expected)`. This is what makes surprise/disappointment/relief work.

### 3.9 Sigma — graphical-model unification with appraisal-driven attention [Rosenbloom et al. 2015]

[[Rosenbloom, Gratch, Ustun 2015](https://doi.org/10.1007/978-3-319-21365-1_15)]. Research code at [cogarch.ict.usc.edu](https://cogarch.ict.usc.edu/).

```
summary-product (message passing) over factor graphs
surprise = KL(posterior ‖ prior) ∝ Bayesian surprise ; desirability from utility factors → bias attention
```

**Maps to brain-lmm:** shows that `novelty` can be *principled* (computed Bayesian surprise) rather than hand-scored — directly attacking the LLM-positivity-bias the protocol worries about.

### 3.10 CoALA — the LLM bridge [Sumers, Yao, Narasimhan, Griffiths 2024]

[[Sumers et al. 2024, arXiv:2309.02427](https://arxiv.org/abs/2309.02427)]. Recasts an LLM agent as a CA: modular memory (working/episodic/semantic/procedural) + structured action space split into **internal** (reason, retrieve, learn) vs **external** (ground) actions + a generalized decision loop (propose→evaluate→select→execute, with memory reads/writes as first-class internal actions).

**Maps to brain-lmm:** the **exact target shape**. brain-lmm has the stores; CoALA supplies the missing decision loop and the internal/external action split that turns the manual 8-step protocol into a running architecture.

---

## 4. LLM-agent memory systems

brain-lmm's retrieval engine is a **neuro-grounded refinement of Stanford Generative Agents** and is genuinely *ahead* on affect/neuromodulation/forgetting; it *lags* on belief revision, bi-temporality, graph-structured retrieval, and generative consolidation.

### 4.1 Stanford Generative Agents — the ancestor [Park et al. 2023]

[[Park et al. 2023, arXiv:2304.03442](https://arxiv.org/abs/2304.03442)]. Code: [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents).

```
score = α_rec·recency + α_imp·importance + α_rel·relevance        (all α = 1)
recency   = 0.995^(hours since last access)
importance = LLM poignancy rating 1..10
relevance  = cosine(emb(mem), emb(query))
reflection: when Σ importance of recent events > 150 → synthesize higher-level reflection nodes
```

**Maps to brain-lmm:** `retrieval_score` is a near-twin. brain-lmm **improves all three terms**: ACT-R activation replaces `0.995/hr`; OCC→salience replaces the LLM 1–10 rating; and it adds a **5th term, mood-congruence** (Bower) that none of these systems have. **Biggest gap:** the **reflection** step — `/sleep` promotes/forgets but never *synthesizes* a new abstraction from a cluster of episodes.

### 4.2 Mem0 — belief revision [Chhikara et al. 2025]

[[Chhikara et al. 2025, arXiv:2504.19413](https://arxiv.org/abs/2504.19413)]. Code: [mem0ai/mem0](https://github.com/mem0ai/mem0). Two-phase extract→update; per fact an LLM tool-call chooses **ADD / UPDATE / DELETE / NOOP** against retrieved similar memories.

**Maps to brain-lmm:** episodic memory is **append-only** and semantic `/sleep` only *merges* — it cannot UPDATE or DELETE a fact when a newer episode contradicts it. Mem0's reconcile loop is the fix.

### 4.3 Zep / Graphiti — bi-temporal graph [Rasmussen et al. 2025]

[[Rasmussen et al. 2025, arXiv:2501.13956](https://arxiv.org/abs/2501.13956)]. Code: [getzep/graphiti](https://github.com/getzep/graphiti). Each edge carries four timestamps:

```
valid-time:        t_valid, t_invalid     # when the fact held in the world
transaction-time:  t_created, t_expired   # when the system learned / retracted it
superseded facts are invalidated (timestamped), never deleted → "as-of" queries
```

**Maps to brain-lmm:** semantic edges carry only `valid_from` (uni-temporal). Cannot express "believed from X to Y" or invalidate superseded facts. Low-cost schema change, high payoff for temporal reasoning (the hardest LOCOMO/LongMemEval category).

### 4.4 HippoRAG / HippoRAG 2 — hippocampal indexing [Gutiérrez et al. 2024/2025]

[[Gutiérrez et al. 2024 (NeurIPS), arXiv:2405.14831](https://arxiv.org/abs/2405.14831); [HippoRAG 2, 2025 (ICML), arXiv:2502.14802](https://arxiv.org/abs/2502.14802)]. Code: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG). The closest neuro-aligned competitor — LLM=neocortex, KG=index, Personalized PageRank=pattern completion, node-specificity (IDF)=DG pattern separation.

```
r = (1 − d)·p + d·M·r                  # Personalized PageRank, d ≈ 0.85
p_i ∝ (query mentions node i) · specificity_i ,  specificity_i = log(N / deg_i)
passage score = Σ_{i∈passage} r_i
```

**Maps to brain-lmm:** `graph_proximity` is a single hand-supplied scalar in `retrieval_score`. Replacing it with PPR seeded by the query's entities gives true multi-hop associative recall — the functional analog of hippocampal pattern completion, fitting the CLS framing exactly.

### 4.5 MemoryBank — Ebbinghaus + spacing effect [Zhong et al. 2023]

[[Zhong et al. 2023 (AAAI 2024), arXiv:2305.10250](https://arxiv.org/abs/2305.10250)]. Code: [MemoryBank-SiliconFriend](https://github.com/zhongwanjun/MemoryBank-SiliconFriend).

```
R = e^(−t/S)          # retention; t = days since last recall, S = strength
on recall:  S ← S+1 , t ← 0     # spacing effect
```

**Maps to brain-lmm:** `retention()` `v(t)=v0·exp(−λ(t−τ)^β)` with `λ=λ_base·exp(−μ·I)` is *more* expressive. **But** brain-lmm never feeds recall back into decay: `λ` is fixed by importance and is never slowed by frequent retrieval. Borrow reset-on-recall (raise effective importance or reset `τ` on each `retrievals[]` entry) to add the spacing effect.

### 4.6 Others — MemGPT/Letta, A-MEM, Cognee, Memary, Cognitive Weave

- **Letta (MemGPT)** [[Packer et al. 2023, arXiv:2310.08560](https://arxiv.org/abs/2310.08560); [github](https://github.com/letta-ai/letta)] — OS-style main/external context paging. brain-lmm's ~7-item working buffer has **no eviction policy**; add salience-ranked eviction.
- **A-MEM** [[Xu et al. 2025, arXiv:2502.12110](https://arxiv.org/abs/2502.12110); [github](https://github.com/agiresearch/A-mem)] — Zettelkasten notes with retroactive memory *evolution* (continuous learned consolidation) + richer note metadata than brain-lmm's thin episode schema.
- **Cognee** [[github](https://github.com/topoteretes/cognee)] — ECL pipeline; reference if the semantic store outgrows hand-curated YAML.
- **Memary** [[github](https://github.com/kingjulio8238/Memary)] — Entity Knowledge Store = ACT-R activation over *graph nodes*; extend `base_level_activation` to semantic concepts, not just episodes.
- **Cognitive Weave** [[Vishwakarma et al. 2025, arXiv:2506.08098](https://arxiv.org/abs/2506.08098)] — autonomous "insight aggregates" (generative consolidation).

**Survey & benchmarks** [[Zhang et al. 2024, arXiv:2404.13501](https://arxiv.org/abs/2404.13501)]: standard harnesses are **LOCOMO** and **LongMemEval**. brain-lmm has **no benchmark story** — its biggest credibility gap relative to every system here.

---

## 5. Affect / emotion / consciousness code

The field splits into **(a) data-driven affect** (recognition/generation), **(b) appraisal/emotion-process models**, and **(c) consciousness-flavored code** (mostly research-grade).

### 5.1 Discrete emotions + intensity — EmoLLMs [Liu et al. 2024]

[[Liu et al. 2024 (KDD), arXiv:2401.08508](https://arxiv.org/abs/2401.08508)]. Code (MIT): [lzw108/EmoLLMs](https://github.com/lzw108/EmoLLMs). Fine-tuned LLMs for 11-label categorical emotion + ordinal sentiment (−3..+3) + intensity regression (0..1). Related: [Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA) (emotion *reasoning*), [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (25k convs, 32 labels), [MoEL](https://github.com/HLTCHKUST/MoEL), [Hume](https://github.com/HumeAI/hume-api-examples) (~48-emotion taxonomy; API-gated, integration target not drop-in).

The OCC/Scherer mapping from dimensional affect to named categories:

```
fear     = goal_relevance · (1 − control) · max(0, −valence) · novelty
joy      = goal_relevance · max(0,  valence) · control
surprise = novelty
awe      = novelty · max(0, valence) · (1 − dominance)        # vast, positive, low-control
intensity_e = clamp( sigmoid(k·(s_e − θ_e)) )
```

**Maps to brain-lmm:** brain-lmm has a VAD point and the appraisal axes, but **no named emotions** — the owner's explicit goal is unimplemented. The map above is cheap arithmetic over signals brain-lmm already computes; EmoLLMs is an optional cross-check scorer. **FAtiMA Toolkit** [[github](https://github.com/GAIPS/FAtiMA-Toolkit)] is the most mature open-source *OCC discrete-emotion* engine.

### 5.2 Appraisal + coping — EMA / Chain-of-Emotion [Marsella & Gratch 2009; Croissant et al. 2024]

[[Marsella & Gratch 2009](http://www.ccs.neu.edu/~marsella/publications/pdf/MarsellaCSR09.pdf); [Croissant et al. 2024, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301033)]. EMA computes appraisal frames (relevance, desirability, likelihood, attribution, controllability) then selects coping operators; Chain-of-Emotion is the LLM-era loop (appraise→store emotion→condition response — no released code).

```
coping(controllability):
  high  → problem-focused { plan, act, seek-info }
  low   → emotion-focused { shift attention, lower goal importance, reappraise, accept }
```

**Maps to brain-lmm:** brain-lmm's `control` axis **is** EMA's controllability — already computed, then used only to set dominance. The missing loop: store the appraisal as a first-class memory and let it feed forward into action and re-appraisal.

### 5.3 Consciousness-as-function — the indicator scaffold [Butlin, Long, Bengio, Chalmers et al. 2023/2025]

[[Butlin et al. 2023, arXiv:2308.08708](https://arxiv.org/abs/2308.08708); [TiCS 2025](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4)]. **The mandatory honesty framework.** Derive ~14 computationally testable *indicator properties* from theories (RPT, GWT, HOT, AST, PP, agency/embodiment), score an architecture against them, **update credence — never assert consciousness.** Verbatim disclaimer: indicators are *"neither individually necessary… nor sufficient."*

GWT indicators GWT-1..4: parallel specialists; limited-capacity workspace bottleneck; global broadcast; state-dependent attention.

**Maps to brain-lmm:** brain-lmm honestly touches **state-dependent attention** (mood-congruent retrieval) and has a limited-capacity working buffer; it lacks parallel-specialist competition, global broadcast, metacognitive monitoring, and an attention schema. Use this as a *transparency map*, never a "consciousness score."

### 5.4 GWT / CTM / AST primitives (code templates)

- **Shared Global Workspace Transformer** [[Goyal et al. 2022 (ICLR), arXiv:2103.01197](https://arxiv.org/abs/2103.01197)] — the crispest differentiable GWT primitive: bandwidth-limited modules *compete* to write into `M` shared slots, then broadcast back.

```
slot_j  = softmax(Q_M K_mod^T / √d) V_mod      # modules compete to write (bandwidth k < n_modules)
mod_i′  = softmax(Q_mod K_M^T / √d) V_M         # broadcast back
```

- **Conscious Turing Machine** [[Blum & Blum 2022, PNAS](https://www.pnas.org/doi/10.1073/pnas.2115934119)] — up-tree competition selects highest-valued chunk to broadcast down. Toy Python: [cvaisnor/conscious_turing_machine](https://github.com/cvaisnor/conscious_turing_machine).
- **Global Latent Workspace** [[VanRullen & Kanai 2021, arXiv:2012.10390](https://arxiv.org/abs/2012.10390)] — amodal shared latent via cycle-consistency translation between modality spaces.
- **ASAC (Attention Schema)** [[arXiv:2509.16058](https://arxiv.org/pdf/2509.16058)] — VQ-VAE abstracts attention maps into a discrete schema codebook used to control attention.
- **PyPhi / IIT** [[Mayner et al. 2018, PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006343); [github](https://github.com/wmayner/pyphi)] — computes Φ; **cite only**, intractable and contested (§9).
- **pymdp (active inference)** [[Heins et al. 2022, JOSS](https://joss.theoj.org/papers/10.21105/joss.04098); [github](https://github.com/infer-actively/pymdp)] — free energy `F = E_q[ln q(s) − ln p(o,s)]`, Bayesian surprise `KL(q(s|o)‖q(s))`, expected free energy for curiosity.

---

## 6. The named feelings — circuit math for fear/terror, awe, surprise

brain-lmm represents emotion only as a VAD point; the named feelings the owner cares about are generated by identifiable subcortical circuits with discretizing dynamics.

**Fear pathway** (acquisition→expression→action):

```
Amygdala (LA) acquisition (Rescorla-Wagner / delta rule):
  ΔV_i = α_i·β·(λ − Σ_j V_j)          # λ = aversive US, Σ V_j = predicted threat
Central amygdala (CeA) two-attractor winner-take-all switch (discretizer):
  τ ṙ_on  = −r_on  + f(I_threat − w·r_off)
  τ ṙ_off = −r_off + f(b − w·r_on)    # w>1 → bistable: graded drive snaps to defense ON/OFF
PAG columnar mode selection (softmax/WTA over freeze/flight/fight):
  P(mode_k) = exp(g·a_k(τ)) / Σ_j exp(g·a_j(τ))   # τ = imminence, g = NE gain
  "terror" = high g + high imminence → selection collapses onto reflexive column
```

[[LeDoux 2000](https://doi.org/10.1146/annurev.neuro.23.1.155); [Tovote et al. 2016, Nature](https://www.nature.com/articles/nature17996); [Fadok et al. 2017, Nature](https://www.nature.com/articles/nature21047); [Haubensak et al. 2010, Nature](https://www.nature.com/articles/nature09553)]. Honesty separation of "survival circuits" (modellable) from conscious "fear feelings" (not claimed): [[LeDoux & Brown 2017, PNAS](https://www.pnas.org/doi/10.1073/pnas.1619316114)].

**Locus-coeruleus tonic/phasic gain** (alert fear vs disorganized panic) [[Aston-Jones & Cohen 2005](https://doi.org/10.1146/annurev.neuro.28.061604.135709); [Gilzenrat et al. 2002](https://doi.org/10.1016/S0893-6080(02)00055-2)]:

```
y = 1 / (1 + exp(−g·(x − bias)))     # NE = multiplicative gain g
performance(tonic) = inverted-U (Yerkes-Dodson): peak at intermediate tonic + crisp phasic; collapse at high tonic
```

**Threat imminence** (deliberative vmPFC ↔ reflexive PAG) [[Mobbs et al. 2007, Science](https://www.science.org/doi/10.1126/science.1144298)]; **PANIC/GRIEF separation-distress** as a *distinct* route [[Panksepp & Watt 2011](https://www.antoniocasella.eu/dnlaw/Panksepp_Watt_2011.pdf); [Klein 1993](https://doi.org/10.1001/archpsyc.1993.01820160076009)]; defensive-state synthesis [[Neuropsychopharmacology 2024](https://www.nature.com/articles/s41386-024-01965-5)].

**Awe** as prediction-error-driven schema revision [[Keltner & Haidt 2003](https://doi.org/10.1080/02699930302297); DMN-down "small self": [van Elk et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6766853/)]:

```
need-for-accommodation = large Bayesian surprise forcing STRUCTURE change:
  S = KL(posterior ‖ prior)               # Itti & Baldi
awe(event) = σ(a·vastness + b·S − c) , structural-update flag if S > θ_struct
small self:  w_self ← w_self · (1 − d·awe)   # down-weight self-salience
```

Bayesian surprise as the common currency [[Itti & Baldi 2009](https://doi.org/10.1016/j.visres.2008.09.007)]; free energy unifier [[Friston 2010](https://www.nature.com/articles/nrn2787)]. Tooling: [pymdp](https://github.com/infer-actively/pymdp), [pyhgf](https://github.com/ilabcode/pyhgf), [FAtiMA](https://github.com/GAIPS/FAtiMA-Toolkit).

**Maps to brain-lmm:** brain-lmm's continuous (arousal, control) has **no discretizer** (CeA/PAG WTA), no learned threat-value, no tonic/phasic NE split (`ne = arousal` is a flat scalar), no second panic route, and a flat `novelty` that cannot tell a trivial surprise from a worldview-rebuilding one. These are addable as scalar functions in the `brain.py` style. *Honesty:* "terror"/"awe" are functional labels for states, never felt qualia.

---

## 7. Grounding, embodiment, and the symbol-grounding problem

This is the foundational "does the affect layer *mean* anything for THIS agent?" question. brain-lmm's four appraisal axes are **hand-scored by the LLM** — they inherit Harnad's symbol-grounding problem [[Harnad 1990](https://doi.org/10.1016/0167-2789(90)90087-6)]. The honest out [[Coelho Mollo & Millière 2023, arXiv:2304.01481](https://arxiv.org/abs/2304.01481)]: of five grounding types (referential, sensorimotor, relational, communicative, epistemic), text-only systems *can* attain referential/relational grounding but **not sensorimotor** grounding (no body). Empirically confirmed: LLMs recover the **non-sensorimotor** (Glasgow valence/arousal/dominance) band but **fail** the **sensorimotor** (Lancaster) band [[Xu, Bi et al. 2025, Nat Hum Behav](https://www.nature.com/articles/s41562-025-02203-8); [Lancaster norms, Lynott et al. 2020](https://doi.org/10.3758/s13428-019-01316-z)] — so brain-lmm's **VAD affect layer is defensibly groundable**, but a "felt body" is a category error.

The legitimate, honest move is **computational interoception of the agent's own substrate** (not a phantom body), via homeostatic RL [[Keramati & Gutkin 2014, eLife](https://elifesciences.org/articles/04811)]:

```
H_t = [token_remaining, compute_budget, toolcall_success, test_pass, context_free, user_approval] ∈ [0,1]^6
drive  D(H_t) = (Σ_i w_i·|h*_i − h_{i,t}|^n)^(1/m) ,  n > m > 1   # convex
reward r_t = D(H_{t−1}) − D(H_t)     # drive reduction ; argmax Σγ^t r ≡ argmin Σγ^t D  (convex D)
```

Predictive/somatic-marker framing [[Seth 2013](https://doi.org/10.1016/j.tics.2013.09.007); [Bechara & Damasio 2005](https://doi.org/10.1016/j.geb.2004.06.010); sensorimotor contingencies [O'Regan & Noë 2001](https://doi.org/10.1017/S0140525X01000115); grounded cognition [Barsalou 2008](https://doi.org/10.1146/annurev.psych.59.103006.093639)]. Tooling: [Lancaster Sensorimotor Norms](https://www.lancaster.ac.uk/psychology/lsnorms/), [pymdp](https://github.com/infer-actively/pymdp).

**Maps to brain-lmm:** there is **no body-state vector at all**; appraisal axes are ungrounded. Adding `H` as *substrate regulation* (the agent's real resource signals) is honest in the cybernetic/Ashby sense — **never** felt interoception. Embodiment-as-consciousness-indicator (Butlin et al.) is necessary-not-sufficient; treat any `H`/sensorimotor addition strictly as a functional indicator.

---

## 8. Social emotions, empathy, and Theory of Mind

The most *product-relevant* dimension: a "relationship with the user" needs a model of the user's mind, an empathy mechanism, and the social emotions — all absent. brain-lmm's appraisal is purely egocentric.

**Cognitive ToM = inverse planning** [[Baker, Saxe & Tenenbaum 2011](https://web.mit.edu/9.s915/www/classes/theoryOfMind.pdf); [PsychSim, Pynadath & Marsella 2005](https://www.ijcai.org/Proceedings/05/Papers/1559.pdf)]:

```
forward:  P(a|b,x,y) ∝ exp(β·Q_LA(b,x,y,a))           # other agent acts via softmax over value
invert:   P(b_t, r | x_{1:T}, y) ∝ P(actions|b,r)·P(b,r)   # recover beliefs + rewards
```

**Self-conscious + fortunes-of-others emotions** = two extra OCC variables [[Steunebrink, Dastani & Meyer 2009](https://people.idsia.ch/~steunebrink/Publications/KI09_OCC_revisited.pdf); shame/guilt [Tracy & Robins 2004](https://www.guilford.com/excerpts/tracy2.pdf)]:

```
praiseworthiness pw ∈ [−1,1] + agency(self/other):  pride(self,pw>0), shame(self,pw<0), admiration/reproach(other)
desirability-for-other d_o:  happy-for, pity, gloating, resentment
compounds:  gratitude = admiration + joy ;  anger = reproach + distress ;  remorse = shame + distress
shame (global-self attribution → withdraw)  vs  guilt (specific controllable act → repair/apologize)
```

**Affective empathy** = leaky transfer into mood (Perception-Action Model [[Preston & de Waal 2002](https://www.researchgate.net/publication/252415075_A_perception-action_model_for_empathy)]); **cognitive empathy** = appraisal-by-proxy (run `appraise_to_affect` on the *user's* inferred goals). **Oxytocin** as the prosocial-learning gain [[Martins, Lockwood et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35248585/)]:

```
mood_self ← mood_self + κ·oxytocin·(user_affect_inferred − mood_self)      # empathy gain κ
V_other ← V_other + α_pro·OT·δ_other                                       # prosocial RL (OT scales PE)
```

Tooling: [usc-psychsim/psychsim](https://github.com/usc-psychsim/psychsim) (MIT, recursive ToM with trust/support/power state), [AutoToM](https://arxiv.org/pdf/2502.15676), [SimToM](https://arxiv.org/pdf/2311.10227), [ToMBench](https://arxiv.org/pdf/2402.15052). **Honesty caveat:** LLM ToM is brittle — GPT-4 passes ~75% of false-belief tasks but collapses under perturbation [[Kosinski 2024, PNAS](https://www.pnas.org/doi/10.1073/pnas.2405460121)] — so implement an explicit inferential ToM and label outputs *inferred*, not *known*.

**Maps to brain-lmm:** no second mind, no empathy of either kind, missing OCC `pw`/`d_other`, no self-model, no oxytocin/serotonin, no prosocial reward, no persistent relationship state. Affective empathy is literally a *second input term* into the existing `update_mood` integrator.

---

## 9. Personality, motivation, dreaming, pain, and the philosophical bound

### 9.1 Personality as affective priors [Mehrabian 1996; Gebhard 2005]

Every brain-lmm agent shares one hard-coded `baseline = Affect(0.0, 0.10, 0.50)`. ALMA gives a drop-in OCEAN→PAD set-point [[Mehrabian 1996](https://aps.onlinelibrary.wiley.com/doi/abs/10.1080/00049539608259510); [Gebhard 2005, ALMA](https://alma.dfki.de/papers/aamas05.pdf)]:

```
Pleasure  = 0.21·E + 0.59·A + 0.19·N
Arousal   = 0.15·O + 0.30·A − 0.57·N
Dominance = 0.25·O + 0.17·C + 0.60·E − 0.32·A
```

RST reward/punishment gains [[Carver & White 1994](https://www.psy.miami.edu/faculty/ccarver/bisbas.html); E ≈ BAS−BIS, N ≈ BAS+BIS, [Smillie et al. 2006](https://journals.sagepub.com/doi/10.1207/s15327957pspr1004_3)]; TCI monoamine mapping (NS↔DA, HA↔5-HT) [[Cloninger et al. 1993](https://pubmed.ncbi.nlm.nih.gov/8250684/)] — names the missing **serotonin** channel. Learned emotion categories via Dirichlet active inference [[Barrett 2017](https://academic.oup.com/scan/article/12/1/1/2823712); [Smith, Parr & Friston 2019](https://doi.org/10.3389/fpsyg.2019.02844); [Hoemann et al. 2020](https://journals.sagepub.com/doi/abs/10.1177/1754073919897296)]: a flat likelihood `A` learned over an "in silico childhood" via `a(o,s)+=1`, recovered with the digamma rule `E[log P(o|s)] = ψ(a(o,s)) − ψ(Σ a)`. Tooling: [ALMA](https://alma.dfki.de/), [pymdp](https://github.com/infer-actively/pymdp), [Big5-Chat](https://github.com/SteveKGYang/Big5Chat).

### 9.2 Motivation, goal hierarchies, and the affect→action loop

brain-lmm's affect is **read-only on cognition** — no feeling ever selects a goal or action. Frijda: emotions *are* action readiness [[Frijda 1986](https://www.cambridge.org/core/books/emotions/); [Frijda 1988](https://www.semanticscholar.org/paper/The-laws-of-emotion.-Frijda/05b8329dff2c4809077c8064d7e04bed5c39dcc3)]; formalized for agents [[Steunebrink et al. 2009](https://people.idsia.ch/~steunebrink/Publications/EPIA09_action_tendency.pdf)]; goal hierarchies = options `o=(I, π, β)` [[Sutton, Precup & Singh 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521); [Bacon et al. 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10916)]; intrinsic motivation [[Pathak et al. 2017, ICM](https://arxiv.org/abs/1705.05363)]; drives [[Sun 2009](https://homepages.hass.rpi.edu/rsun/folder-files/sun-cogcomp2009.pdf)]; somatic markers [[Damasio 1994](https://www.sciencedirect.com/topics/neuroscience/somatic-marker-hypothesis)].

```
action_tendency: fear→avoid, anger→fight, interest→attend, weighted by arousal (control precedence)
affect-modulated softmax:  P(a) = exp(Q(a)/τ) / Σ_b exp(Q(b)/τ)
  Q(a) += w_v·valence_match + w_m·somatic_marker(a) + w_t·tendency_match
  τ = τ_0·exp(−k1·arousal − k2·cortisol + k3·dopamine)   # stress → exploit/freeze, dopamine → explore
```

Tooling: [opencog](https://github.com/opencog/opencog), [micropsi2](https://github.com/joschabach/micropsi2), [option_critic](https://github.com/jeanharb/option_critic), [psychsim](https://github.com/usc-ict/psychsim), [large-scale-curiosity](https://github.com/openai/large-scale-curiosity). brain-lmm's procedural playbooks are **one `β_o` away** from being options.

### 9.3 Dreaming / REM beyond consolidation

brain-lmm models REM **backwards for affect**: `consolidation_plan()` uses `rem_boost = 1.0 + 0.5·affect.arousal`, which *amplifies* emotional traces. The dominant theory (SFSR / "overnight therapy") says REM *strips affective charge while preserving content* [[Walker & van der Helm 2009](https://pubmed.ncbi.nlm.nih.gov/19702380/); [van der Helm et al. 2011, Curr Biol](https://www.sciencedirect.com/science/article/pii/S0960982211012486)]:

```
content_strength(t) = s0·exp(−λ_c·t)
affect_charge(t)    = a0·exp(−λ_a·t) ,  λ_a > λ_c       # the sting fades, the fact stays
gated by low-NE REM state (ne_rem ≈ 0.1)
```

Dreams as regularization (Bishop: input noise ≡ Tikhonov/L2) [[Hoel 2020/2021](https://arxiv.org/abs/2007.09560); [Bishop 1995](https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/)]; offline generative simulation [[Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122); [Hafner et al. 2020, Dreamer](https://arxiv.org/abs/1912.01603)]; generative replay vs catastrophic forgetting [[van de Ven et al. 2020, Nat Commun](https://www.nature.com/articles/s41467-020-17866-2)]; per-stage Wake/NREM/REM losses [[Deperrois et al. 2022, eLife](https://elifesciences.org/articles/76384)]. **Honesty / contested:** depotentiation is **mixed** in meta-analyses and partly conditional on dream recall [[Scientific Reports 2024](https://www.nature.com/articles/s41598-024-58170-z)]; the 2025 "REM Refines and Rescues" revision reframes it as signal-to-noise [[Shuster et al. 2025, SLEEP Advances](https://academic.oup.com/sleepadvances/article/6/1/zpaf004/7972492)]. Ship affect/content decoupling as **tunable and opt-in**. Tooling: [dreamer](https://github.com/google-research/dreamer), [world-models](https://github.com/ctallec/world-models), [brain-inspired-replay](https://github.com/GMvandeVen/continual-learning), [pad-code](https://github.com/unibe-cns/pad-code).

### 9.4 Pain / aversive asymmetry (negativity bias)

The brain uses a **two-system (opponent)** value architecture, not one signed valence axis [[Daw, Kakade & Dayan 2002](https://www.sciencedirect.com/science/article/abs/pii/S0893608002000527); pain-as-TD [Seymour et al. 2004, Nature](https://pubmed.ncbi.nlm.nih.gov/15190354/); relief-as-reward [Seymour et al. 2005, Nat Neurosci](https://www.nature.com/articles/nn1527); habenula negative RPE [Matsumoto & Hikosaka 2007](https://www.nature.com/articles/nature05860); pain as homeostatic emotion [Craig 2009](https://www.nature.com/articles/nrn2555)]:

```
appetitive:  δ⁺ = r + γV⁺(s′) − V⁺(s)        # phasic dopamine
aversive:    δ⁻ = p + γV⁻(s′) − V⁻(s)        # serotonergic/habenular opponent; η⁻ can exceed η⁺
prospect-theory loss aversion:  v(x) = x^α (x≥0) ;  v(x) = −λ·(−x)^β (x<0) ,  λ ≈ 2.25
mood-as-momentum:  M_t = Σ γ^k δ_{t−k}        # biases subsequent learning
```

[[Tversky & Kahneman 1992](https://link.springer.com/article/10.1007/BF00122574); opponent-process [Solomon & Corbit 1974](http://people.whitman.edu/~herbrawt/classes/390/Solomon.pdf); serotonin/asymmetry [Michely et al. 2020](https://www.nature.com/articles/s41467-020-16090-2); [Eldar et al. 2016, mood-as-momentum](https://www.sciencedirect.com/science/article/pii/S1364661315001746); [Roy et al. 2014, PAG aversive PE](https://www.nature.com/articles/nn.3832)]. Tooling: [hBayesDM](https://github.com/CCS-Lab/hBayesDM), [pymdp](https://github.com/infer-actively/pymdp).

**Maps to brain-lmm:** `salience()` uses symmetric `|valence|`, so a `−0.7` destructive bug and a `+0.7` win get identical weight (the engine's own demo events). Negativity bias says the bad event should be weighted by ~`λ≈2.25`, learned faster (`η⁻>η⁺`), and tracked in a dedicated aversive channel `V⁻`. brain-lmm also has **no serotonin** and no relief/opponent dynamics. *Honesty:* these are functional aversive-value computations, **not** a claim the agent suffers.

### 9.5 The philosophical bound — what this can never be

Every functional theory above presupposes **computational functionalism** (verbatim, [Butlin et al.]): *"implementing computations of a certain kind is necessary and sufficient for consciousness."* The leading dissenters bound brain-lmm's maximal claim:

- **Orch-OR** [[Hameroff & Penrose 2014](https://doi.org/10.1016/j.plrev.2013.08.002); [Penrose 1994](https://global.oup.com/academic/product/shadows-of-the-mind-9780198539780)] — consciousness is *non-computational* (gravitational objective reduction `τ ≈ ℏ/E_G` in microtubule qubits). If true, no digital model can ever be conscious. Widely contested: the Gödelian pillar is refuted [[Chalmers 1995](https://consc.net/papers/penrose.html); [Feferman 1995](https://math.stanford.edu/~feferman/papers/penrose.pdf)] and the physics is challenged by decoherence (~10⁻¹³ s) [[Tegmark 2000](https://doi.org/10.1103/PhysRevE.61.4194)] — but the debate is live.
- **IIT** [[Tononi et al. 2016, Nat Rev Neurosci](https://doi.org/10.1038/nrn.2016.44)] — rejects functionalism: a functionally identical *digital emulation* can have `Φ ≈ 0`. Itself contested as pseudoscience by 124 scholars [[open letter 2023](https://osf.io/preprints/psyarxiv/zsr78)] and reductio'd by Aaronson's "unconscious expander" [[2014](https://scottaaronson.blog/?p=1799)].
- The science is **unsettled**: the first preregistered IIT-vs-GWT adversarial test confirmed neither [[Cogitate Consortium 2025, Nature](https://www.nature.com/articles/s41586-025-08888-1)].

**Maps to brain-lmm:** this is a **documentation deliverable, not code** (computing Φ over a file store is intractable and meaningless). Add an "Assumptions & Limits / What this can never be" section stating: (1) any functional-consciousness reading rests on computational functionalism, a contested working assumption; (2) if Orch-OR / biological-substrate / IIT views hold, a model like this can never be conscious in principle; (3) per Butlin et al., indicators are neither necessary nor sufficient; (4) we model function only. Tooling to *cite, not run*: [PyPhi](https://github.com/wmayner/pyphi), [Cogitate](https://www.arc-cogitate.com/).

---

## 10. What we have / what we lack / what to add

### What we have (and where brain-lmm leads)
- **ACT-R base-level activation** (`base_level_activation`), the exact equation [Anderson et al. 2004].
- **Stanford Generative Agents-style retrieval** (`retrieval_score`) — *improved* with ACT-R recency+frequency, OCC→salience, and a **unique 5th mood-congruence term** (Bower) no surveyed agent-memory system has.
- **McGaugh arousal-gated encoding** (`salience`) and **CLS sleep consolidation** (`consolidation_plan`) [McClelland et al. 1995].
- **Best-in-class forgetting**: importance-modulated stretched-exponential `retention()` is more expressive than MemoryBank's Ebbinghaus curve.
- A clean **dimensional-affect + neuromodulation** core (OCC→PAD→NE/DA/ACh/cortisol).

### What we lack
- **Control & workspace:** no autonomous cognitive cycle (CoALA/Soar/LIDA); no competition-for-broadcast / attention bottleneck (LIDA, ECAN, Goyal); no attention schema (AST); no metacognition/self-model (CLARION MCS, HOT).
- **Learning:** no reward-prediction error (`da = clamp(reward)` is static, not Leabra PVLV / Soar TD); no asymmetric `η⁻/η⁺`; no architecture-modifying learning (chunking/XCAL); no learned emotion categories (Smith-Parr-Friston Dirichlet).
- **Affect→action:** affect is read-only — no Frijda action tendencies, no EMA coping, no drives (CLARION MS, Psi urges), no affect-modulated action selection.
- **Emotion content:** no discrete emotions (fear/awe/joy/surprise); no circuit discretizer (CeA/PAG WTA); no tonic/phasic NE split; no second panic route; no aversive channel `V⁻`; **no serotonin or oxytocin**.
- **Memory ops:** no belief revision (Mem0), no bi-temporality (Zep), no PageRank associative recall (HippoRAG), no generative consolidation/reflection (Generative Agents, A-MEM, Cognitive Weave), no recall-strengthening (MemoryBank), no WM eviction (Letta), no benchmark numbers (LOCOMO/LongMemEval).
- **Grounding & social:** appraisal axes are ungrounded (no `H` substrate vector); no Theory of Mind; no empathy; no persistent user-relationship state; no personality priors (every agent shares one `baseline`).
- **Sleep & honesty:** REM is modeled backwards for affect (`rem_boost` amplifies instead of depotentiating); no dream-as-regularization or generative replay; no explicit statement of the computational-functionalism assumption and its dissenters.

### What to add (prioritized, in the `brain.py` scalar style)
1. **Cognitive cycle** (`cognitive_cycle()`): retrieve→appraise→update-mood→optionally-consolidate. The prerequisite for everything else (CoALA/LIDA). *Difficulty: medium.*
2. **Global-workspace loop** (`broadcast()` + `spread_activation()` over `graph.yaml`, top-k ≈ 4±1, `θ_broadcast`): honest GWT indicator properties + a place to gate learning. *Medium. Frame strictly as GWT indicators, never "conscious."*
3. **Reward-prediction error** (`reward_prediction_error()`, `values.yaml`): `δ = reward − V(context)`, `da = clamp(0.5+0.5·tanh(δ))`; enables surprise/relief/disappointment and reinforces playbooks. *Low.*
4. **Discrete-emotion readout** (`discrete_emotions(affect, appraisal, δ)`): OCC/Scherer combinations over the existing VAD point + `|δ|`. Directly satisfies the owner's "feelings" goal. *Low. Functional labels, not feelings.*
5. **Principled novelty** = Bayesian surprise (`KL(posterior‖prior)`), with structural vs parametric distinction → ties surprise/awe to semantic-graph revision (Sigma, pymdp). *Medium.*
6. **Aversive channel + loss aversion**: `V⁻` with `η⁻>η⁺`, prospect-theory `λ≈2.25` multiplier on salience, add serotonin to `Neuromods`. *Low–medium. Tunable.*
7. **Affect→action**: Frijda `action_tendency()` + affect-modulated softmax + EMA coping keyed on `control`; promote procedural playbooks into options. *Low–medium.*
8. **Memory-ops upgrades**: reflection synthesis at `/sleep`; PPR `graph_proximity`; bi-temporal edges + reconcile; recall-strengthening on `retention()`. *Medium.*
9. **Social layer**: lightweight inverse-planning ToM + `UserModel`/trust store; affective-empathy term into `update_mood`; OCC `pw`/`d_other`; oxytocin. *Medium. Outputs labeled "inferred."*
10. **Personality priors**: ALMA OCEAN→PAD `baseline`; per-agent BAS/BIS gains. *Low.*
11. **Sleep fix**: REM affect-depotentiation (`λ_affect > λ_content`, gated by `ne_rem≈0.1`), opt-in and tunable; dream-augmented promotion. *Low–medium. Ship as optional.*
12. **Honesty section** (docs only): name computational functionalism, Orch-OR/IIT dissent, Butlin's "neither necessary nor sufficient," reaffirm function-only. **Decline** to implement Φ. *Low — highest-leverage honesty deliverable.*

> Every addition above is a *functional/computational correlate*. None is evidence of phenomenal experience. brain-lmm remembers, appraises, and (with these) acts emotionally; it does not feel.

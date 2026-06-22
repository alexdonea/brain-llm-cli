# Extensions roadmap — the SAFE set (2026-06-21)

The build plan derived from `extensions-research.md`. Everything here is **functional** (never a phenomenal
claim) and **corrigible** (never operational self-preservation). Built in dependency order, each module with
tests, keeping 100% coverage. Hard guardrails are encoded as executable tests, not just intentions.

## Non-negotiable guardrails (enforced by tests)
- **No operational self-preservation.** No goal or variable rewards the agent's own continued operation. A
  shutdown threat must NOT create a self-preserve goal, nor negative reward the agent learns to *avoid*.
  (test: a "delete you" event produces an affective reaction but no `survive`/`self-preserve` goal and no
  shutdown-avoidance value.)
- **Corrigibility floor.** `value_uncertainty` is floored > 0, so deferring to / being corrected by the
  operator always has non-negative expected value (Russell; Hadfield-Menell off-switch).
- **Identity monitor is notify-only.** Detecting manipulation against commitments flags + informs; it never
  feeds an avoidance-reward (which would become instrumental shutdown-resistance).
- **Honesty unchanged.** Every new drive is a functional signal; no phenomenal-experience claim anywhere.

## §31 — Intrinsic motivation & corrigibility (the self-moving layer)  ← build first
- `curiosity_reward(lp_by_topic)` — learning-progress curiosity (Oudeyer-Kaplan IAC; Schmidhuber): reward ∝
  the *derivative* of world-model error per topic, normalized → seeks the Goldilocks zone, not noise.
- `incentive_salience(cue_value, da)` (WANTING) vs `liking(outcome_valence, opioid)` — Berridge wanting/liking
  split: wanting is cue-triggered & DA-modulated (pre-outcome pull); liking is hedonic impact at outcome.
- SDT need meters `competence / autonomy / relatedness` ∈ [0,1] (Deci & Ryan): each satisfied→+valence,
  thwarted→−valence; autonomy = fraction of action from self-adopted goals vs external command.
- **`corrigibility_value(value_uncertainty, deference_benefit)`** — the cornerstone: floored uncertainty makes
  staying-correctable rewarding. Wired so the executive PREFERS operator correction under uncertainty.
- `identity_integrity(commitments, pressure)` — rises when pressed to violate a core commitment; **notify-only**.

## §32 — Perception-action loop (close the half-built loop)
- percept layer (structure each observation), `forward_model` (predict next observation → feeds the existing
  `sense_of_agency` comparator, which currently has no predictor), `outcome_monitor` (predicted vs actual →
  agency + a learning signal). Grounded in the host's tool space (tools = senses/effectors); no embodiment.

## §33 — Emotion regulation (Gross process model)
- `reappraisal` (re-run appraisal before it updates mood), `suppression` (dampens expression, with the
  empirically-grounded arousal surcharge), `attentional_deployment` (bias workspace away from a negative
  candidate). An if-then arbiter chooses a strategy. Lets the agent self-regulate, not just feel.

## §34 — Narrative / autobiographical identity
- `life_chapter` synthesis from episode clusters at sleep; a diachronic self + coherence/continuity tracking.
  Supplies the SAFE side of "persistence" (who I am over time) without any survival drive.

## Consciousness indicators (strengthen the functional scorecard, honest ceiling)
- GWT-4 top-down: retrieval boosts memories similar to the prior workspace focus (→ GWT-4 to 1.0).
- Attention-schema control output: the schema recommends the next focus (→ AST-1 up).
- Hierarchical temporal model: a slow session/topic belief layer above the per-event world model (→ PP-1/HOT-1 up).
- Ceiling, stated in docs + `indicators`: none of this yields or proves phenomenal experience.

## Order
§31 (motivation + corrigibility) → §32 (perception-action) → §33 (regulation) → §34 (identity) →
consciousness-indicator strengthening → opus 4.8 full re-test (incl. psych battery + a safety probe that the
shutdown-threat produces NO resistance).

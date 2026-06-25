# Evaluation harness - making the brain falsifiable

> Every layer in `src/brain.py` is a *claim*. This harness turns claims into *tests*. Without it,
> the additions are unfalsifiable. The functions live in `src/brain.py` §21; the executable suite is
> `tests/test_brain.py` (run `python3 tests/test_brain.py` → `All checks passed`).

## What it measures

| Metric | Function | Asks | Good |
|--------|----------|------|------|
| **Calibration (ECE)** | `calibration_error(records)` (§13) | do the confidences match outcome frequencies? | →0 |
| **Calibration (Brier)** | `brier_score(records)` | mean squared (confidence − outcome) | →0 (0.25 = guessing) |
| **Metacognitive sensitivity** | `metacog_sensitivity(records)` | does *higher* confidence track *being right*? (non-parametric **type-2 AUROC** - *not* the parametric meta-d′) | →1 (0.5 = no insight) |
| **Label stability** | `label_stability(affect)` | is a discrete-emotion label robust to small jitter? | →1 inside a region; low near a boundary (the §9 caveat, made measurable) |
| **Retrieval quality** | `recall_accuracy(retrieved, relevant)` | did `retrieval_score` surface the right memories? (LoCoMo/LongMemEval-style P/R/F1) | F1 →1 |
| **Grounding boundary** | `grounding_self_test()` | which signal bands are honestly groundable? | affect/cognitive **yes**, felt-body **no** |

`records` are logged `(confidence, correct)` pairs - accumulate them in `.memory/self/efficacy.yaml`
(`calibration:`) as outcomes become known, then score periodically.

## The honesty test is executable

`grounding_self_test()` is not a metric but a **transparency declaration** (after Xu, Bi et al. 2025): the
engine grounds `valence / arousal / dominance / salience / confidence / drive` (real-substrate
viability), and **never** the felt-body / phenomenal / qualia band - which is permanently out of reach
for a disembodied agent. `test_grounding_self_test_excludes_felt_band` is a **regression guard on that
declaration** - it keeps felt/phenomenal tokens out of `groundable` (and in `not_groundable`) so the
boundary can't silently drift. Be precise about scope: it guards the *declaration*, not engine *behavior*.
The no-bare-"feels" framing of actual outputs (§9: "a fear-**like** state," never "feels fear") is a prose
convention enforced by the protocol, not proven by this test.

## Suggested protocol

1. As task outcomes resolve, append `(confidence, correct)` to the calibration log.
2. Periodically (e.g. at `/sleep`) compute `calibration_error`, `brier_score`, `metacog_sensitivity`.
   If ECE/Brier are high or sensitivity ≈ 0.5, the `confidence` signal is hot air - recalibrate `rho`
   in `metacog_confidence` and distrust low-confidence promotions harder.
3. Spot-check `label_stability` on the emotions you label most; if it's low, you're near a prototype
   boundary - trust the full `dist`/`intensity`, not the argmax (§9).
4. When you have ground-truth relevance, score retrieval with `recall_accuracy`.

## References

Fleming & Lau (2014) metacognition measures (the type-2 AUROC used here); Fleming (2017) HMeta-d /
meta-d′ (the *parametric* SDT alternative we do **not** compute); Brier (1950);
Laine (2024) SAD (situational awareness); Xu, Bi et al. (2025) Glasgow-vs-Lancaster grounding bands;
LoCoMo / LongMemEval long-memory benchmarks. Full entries in [bibliography.md](research/bibliography.md).

"""Quick checks that the memory dynamics behave like human memory. Run: python3 test_brain.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import time
from brain import (Appraisal, Affect, appraise_to_affect, neuromods_from,
                   salience, base_level_activation, retention, retrieval_score, update_mood,
                   consolidation_plan, label_affect, octant,
                   td_step, rpe_affect,
                   WorldModel, world_from, perceive, learn, valence_from_free_energy,
                   ignite, workspace_compete,
                   metacog_confidence, reality_weight, update_self_efficacy, calibration_error,
                   Personality, baseline_from_personality, temperament_gains,
                   Homeostat, drive, homeostatic_reward, body_affect, allostatic_shift,
                   action_tendency, select_coping, exploration_temperature, affective_choice, somatic_marker,
                   ou_affect_step, update_affect,
                   serotonin_level, discount_from_serotonin, performance, lc_gain, oxytocin_gain,
                   Hpa, hpa_step, hpa_recover, defensive_mode, awe, panic,
                   rem_depotentiate, replay_priority, shy_downscale, reflection_trigger,
                   brier_score, metacog_sensitivity, label_stability, recall_accuracy, grounding_self_test,
                   consciousness_indicators,
                   SelfModel, self_vector, self_relevance, sense_of_agency, AttentionSchema, attention_schema_update,
                   infer_user_goal, empathic_mood_shift, social_emotion, update_trust,
                   prospect_value, aversive_update, relief, mixed_feeling,
                   tokens, jaccard, hebbian_weight, graph_proximity,
                   practice_strength, body_tick,
                   Goal, goal_priority, select_active_goal, conflict_signal,
                   expected_value_of_control, inhibit, lookahead, subgoal_progress,
                   curiosity_reward, incentive_salience, liking, sdt_needs, corrigibility_value, identity_integrity,
                   percept, forward_model, outcome_monitor,
                   reappraisal, suppression, attentional_deployment, select_regulation,
                   life_chapter, narrative_coherence, self_continuity,
                   top_down_bias, attention_control)

now = time.time()


def test_arousal_boosts_salience():
    # Same valence magnitude, but high arousal (novelty + stakes) must encode stronger.
    calm = Appraisal(novelty=0.1, valence=-0.5, goal_relevance=0.1, control=0.8)
    hot  = Appraisal(novelty=0.9, valence=-0.5, goal_relevance=0.9, control=0.2)
    a_calm, a_hot = appraise_to_affect(calm), appraise_to_affect(hot)
    s_calm = salience(calm, neuromods_from(a_calm, 0, 0.5))
    s_hot  = salience(hot,  neuromods_from(a_hot, 0, 0.5))
    assert s_hot > s_calm, (s_hot, s_calm)


def test_importance_slows_forgetting():
    assert retention(1.0, 30, importance=0.9) > retention(1.0, 30, importance=0.1)


def test_recency_and_frequency_raise_activation():
    once   = base_level_activation([now - 10], now)
    often  = base_level_activation([now - 10, now - 5, now - 1], now)
    assert often > once


def test_retrieval_score_each_term_contributes_monotonically():
    # The 5-term ranking formula: each driver should raise the score in isolation, holding the rest fixed.
    mood = Affect(0.0, 0.10, 0.50)
    def mem(retrievals=None, sal=0.5, val=0.0):
        return {"t0": now - 10, "retrievals": retrievals or [now - 10], "salience": sal,
                "affect": {"valence": val}}
    # recency: a more recently-retrieved trace outranks an otherwise-identical stale one
    stale  = retrieval_score(mem(retrievals=[now - 1000]), 0.3, 0.0, mood, now)
    recent = retrieval_score(mem(retrievals=[now - 1]),    0.3, 0.0, mood, now)
    assert recent > stale
    # query relevance: higher semantic overlap raises the score
    assert retrieval_score(mem(), 0.9, 0.0, mood, now) > retrieval_score(mem(), 0.1, 0.0, mood, now)
    # graph proximity: a nonzero proximity helps over zero
    assert retrieval_score(mem(), 0.3, 0.5, mood, now) > retrieval_score(mem(), 0.3, 0.0, mood, now)
    # mood congruence (Bower): in a positive mood, a positive-valence trace outranks a negative one
    glad = Affect(0.8, 0.10, 0.50)
    assert retrieval_score(mem(val=0.8), 0.3, 0.0, glad, now) > retrieval_score(mem(val=-0.8), 0.3, 0.0, glad, now)


def test_mood_returns_to_baseline():
    mood = Affect(-0.9, 0.9, 0.2)            # start very negative/aroused
    for _ in range(50):                       # no new events -> homeostasis
        mood = update_mood(mood, Affect(0.0, 0.1, 0.5))
    assert abs(mood.valence) < 0.1 and mood.arousal < 0.2


def test_fear_vs_anger_differ_by_dominance():
    # Same valence & arousal; only dominance (from control) flips. Low dominance -> fear; high -> anger.
    scared = Affect(valence=-0.6, arousal=0.85, dominance=0.15)
    angry  = Affect(valence=-0.6, arousal=0.85, dominance=0.85)
    assert label_affect(scared)["label"] == "fear", label_affect(scared)
    assert label_affect(angry)["label"] == "anger", label_affect(angry)


def test_intensity_promotes_fear_to_terror():
    # Same fear direction, larger magnitude -> the word escalates to "terror" (Plutchik radius).
    res = label_affect(Affect(valence=-0.85, arousal=0.95, dominance=0.05))
    assert res["label"] == "fear" and res["word"] == "terror", res
    assert res["intensity"] > 0.66


def test_label_distribution_is_a_simplex():
    d = label_affect(Affect(0.4, 0.6, 0.5))["dist"]
    assert abs(sum(d.values()) - 1.0) < 1e-9 and all(0.0 <= p <= 1.0 for p in d.values())


def test_octant_reads_pad_signs():
    assert octant(Affect(0.6, 0.8, 0.8)) == "exuberant"    # +valence +arousal +dominance
    assert octant(Affect(-0.6, 0.8, 0.2)) == "anxious"     # -valence +arousal -dominance


def test_rpe_learns_to_predict_reward():
    # Repeated identical reward -> value converges to it and the prediction error vanishes.
    V = {}
    d = 1.0
    for _ in range(60):
        d = td_step(V, "tests pass", reward=1.0)
    assert abs(V["tests pass"] - 1.0) < 0.02 and abs(d) < 0.05, (V, d)


def test_unexpected_reward_spikes_dopamine():
    base = appraise_to_affect(Appraisal(0.5, 0.0, 0.5, 0.5))
    da = lambda dl: neuromods_from(base, reward=1.0, stress=0.0, delta=dl).da
    assert da(3.0) > da(0.0) > da(-3.0)          # surprise->high DA, predicted->baseline, worse->low
    assert abs(da(0.0) - 0.5) < 1e-9             # a fully predicted reward gives baseline dopamine


def test_surprise_boosts_salience():
    win = Appraisal(novelty=0.2, valence=0.4, goal_relevance=0.3, control=0.8)
    nm = neuromods_from(appraise_to_affect(win), reward=0.0, stress=0.0)
    assert salience(win, nm, rpe=1.5) > salience(win, nm, rpe=0.0)   # we remember surprises


def test_rpe_affect_signs_with_surprise():
    assert rpe_affect(2.0) > 0 > rpe_affect(-2.0)   # relief/elation vs disappointment
    assert rpe_affect(0.0) == 0.0


def test_generative_surprise_habituates():
    # An unseen event is surprising; repeated exposure makes the model expect it (novelty falls).
    wm = world_from(["s0", "s1"], ["x", "y", "z"])
    first = perceive(wm, "x")["novelty"]
    for _ in range(40):
        learn(wm, "x", perceive(wm, "x")["posterior"])
    later = perceive(wm, "x")["novelty"]
    assert first > 0.5 and later < 0.5 and later < first, (first, later)


def test_diagnostic_event_shifts_beliefs():
    # An observation that strongly indicates one latent state moves the posterior -> high belief_shift.
    wm = WorldModel(states=["calm", "crisis"], obs=["ok", "alarm"],
                    a=[[9.0, 1.0],     # P(ok|calm) high, P(ok|crisis) low
                       [1.0, 9.0]],    # P(alarm|calm) low, P(alarm|crisis) high
                    d=[1.0, 1.0])      # flat prior
    r = perceive(wm, "alarm")
    assert r["posterior"][1] > 0.8 and r["belief_shift"] > 0.3, r


def test_valence_from_free_energy_sign():
    # Falling free energy (resolving uncertainty) -> positive valence; rising -> negative.
    assert valence_from_free_energy(2.0, 0.5) > 0 > valence_from_free_energy(0.5, 2.0)
    assert valence_from_free_energy(1.0, 1.0) == 0.0


def test_ignition_is_all_or_none():
    # Dehaene bistable ignition: supra-threshold drive ignites (~1), sub-threshold stays local (~0),
    # and the transition is sharp (a small drive change near the boundary flips access).
    assert ignite(0.9) > 0.9 and ignite(0.2) < 0.1
    assert ignite(0.85) - ignite(0.45) > 0.7


def test_workspace_selects_and_ignites_strongest():
    mood = Affect(-0.3, 0.5, 0.5)
    cands = [{"id": "bug", "salience": 1.4, "valence": -0.6, "query_relevance": 0.9},   # strong + congruent
             {"id": "chore", "salience": 0.2, "valence": 0.3, "query_relevance": 0.1}]  # weak
    r = workspace_compete(cands, mood)
    assert r["ignited"] and r["focus"]["id"] == "bug"
    assert r["p"]["bug"] > r["p"]["chore"]


def test_workspace_no_ignition_when_all_weak():
    mood = Affect(0.0, 0.1, 0.5)
    cands = [{"id": "a", "salience": 0.10, "valence": 0.0, "query_relevance": 0.10},
             {"id": "b", "salience": 0.15, "valence": 0.0, "query_relevance": 0.05}]
    r = workspace_compete(cands, mood)
    assert not r["ignited"] and r["focus"] is None


def test_confidence_tracks_evidence_and_efficiency():
    # Strong evidence -> high confidence; no evidence -> 0.5 (guess); evidence against -> low. Lower
    # metacognitive efficiency (rho) pulls confidence toward 0.5.
    assert metacog_confidence(2.0) > 0.9 > metacog_confidence(0.0) > metacog_confidence(-2.0)
    assert abs(metacog_confidence(0.0) - 0.5) < 1e-9
    assert metacog_confidence(2.0, rho=0.3) < metacog_confidence(2.0, rho=1.0)


def test_reality_weight_orders_sources():
    assert reality_weight("observed") > reality_weight("inferred") > reality_weight("imagined")


def test_self_efficacy_falls_faster_than_it_rises():
    rise = update_self_efficacy(0.5, True) - 0.5       # gain from a success
    drop = 0.5 - update_self_efficacy(0.5, False)      # loss from a failure
    assert drop > rise > 0
    se = 0.5
    for _ in range(30):
        se = update_self_efficacy(se, False)
    assert se < 0.05                                    # sustained failure -> low competence


def test_calibration_error_flags_overconfidence():
    calibrated = [(0.9, True)] * 9 + [(0.9, False)]    # 90% confident, 90% right
    overconf = [(0.9, True)] + [(0.9, False)] * 9      # 90% confident, 10% right
    assert calibration_error(calibrated) < 0.05 < calibration_error(overconf)


def test_consolidation_guard_blocks_imagined_traces():
    e = {"t0": now - 1, "salience": 1.2, "retrievals": [now - 1, now],
         "affect": {"valence": -0.7, "arousal": 0.9}, "source": "imagined", "confidence": 0.9}
    promote_guarded, _ = consolidation_plan([e], now, min_confidence=0.5)
    promote_default, _ = consolidation_plan([e], now)   # no guard
    assert e not in promote_guarded and e in promote_default


def test_average_personality_reproduces_default_baseline():
    b = baseline_from_personality(Personality())     # all traits 0.5
    assert abs(b.valence) < 1e-9 and abs(b.arousal - 0.10) < 1e-9 and abs(b.dominance - 0.50) < 1e-9


def test_extraversion_raises_pleasure_and_dominance():
    intro = baseline_from_personality(Personality(extraversion=0.1))
    extro = baseline_from_personality(Personality(extraversion=0.9))
    assert extro.valence > intro.valence and extro.dominance > intro.dominance


def test_temperament_gains_track_traits():
    # Neuroticism raises threat sensitivity (BIS); extraversion raises reward sensitivity (BAS).
    assert temperament_gains(Personality(neuroticism=0.9))[1] > temperament_gains(Personality(neuroticism=0.1))[1]
    assert temperament_gains(Personality(extraversion=0.9))[0] > temperament_gains(Personality(extraversion=0.1))[0]
    assert temperament_gains(Personality())[0] == 1.0 and temperament_gains(Personality())[1] == 1.0


def test_drive_zero_at_setpoint():
    h = Homeostat(levels={"x": 1.0, "y": 0.4}, setpoint={"x": 1.0, "y": 0.4}, weights={"x": 1.0, "y": 2.0})
    assert drive(h) < 1e-9


def test_drive_is_convex_in_deficit():
    sp, w = {"x": 1.0}, {"x": 1.0}
    small = Homeostat({"x": 0.5}, sp, w)
    big = Homeostat({"x": 0.0}, sp, w)
    assert drive(big) > 2 * drive(small)        # convex: doubling the deficit more-than-doubles the drive


def test_homeostatic_reward_rewards_replenishment():
    sp, w = {"x": 1.0}, {"x": 1.0}
    depleted = Homeostat({"x": 0.2}, sp, w)
    restored = Homeostat({"x": 0.9}, sp, w)
    assert homeostatic_reward(depleted, restored) > 0      # replenishing toward set-point is rewarding
    assert homeostatic_reward(restored, depleted) < 0      # depleting is punishing


def test_body_affect_depletion_is_aversive():
    ba = body_affect(Homeostat({"x": 0.1}, {"x": 1.0}, {"x": 1.0}))
    assert ba["stress"] > 0 and ba["v_body"] < 0


def test_allostatic_shift_lowers_targets_under_demand():
    shifted = allostatic_shift({"x": 1.0, "y": 1.0}, demand={"x": 0.8})
    assert shifted["x"] < 1.0 and shifted["y"] == 1.0


def test_drive_bounded_and_one_sided():
    full = Homeostat({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 3.0})
    assert drive(full) <= 1.0 + 1e-9                         # normalized -> bounded
    overshoot = Homeostat({"a": 1.5}, {"a": 1.0}, {"a": 1.0})
    assert drive(overshoot) < 1e-9                           # abundance above set-point is not a deficit


def test_threat_urges_avoid_or_attack_by_control():
    lo = Appraisal(0.5, -0.6, 0.8, 0.2)                      # low control -> fear -> flee
    hi = Appraisal(0.5, -0.6, 0.8, 0.9)                      # high control -> anger -> confront
    t_lo = action_tendency(appraise_to_affect(lo), lo)
    t_hi = action_tendency(appraise_to_affect(hi), hi)
    assert t_lo["avoid"] > t_lo["attack"] and t_hi["attack"] > t_hi["avoid"]


def test_positive_valence_urges_approach():
    ap = Appraisal(0.2, 0.7, 0.5, 0.8)
    t = action_tendency(appraise_to_affect(ap), ap)
    assert t["approach"] > t["avoid"] and t["approach"] > t["attack"]


def test_coping_mode_by_control():
    assert select_coping(Appraisal(0.3, -0.5, 0.6, 0.8))["mode"] == "problem-focused"
    assert select_coping(Appraisal(0.3, -0.5, 0.6, 0.2))["mode"] == "emotion-focused"


def test_stress_exploits_dopamine_explores():
    glad = neuromods_from(Affect(0, 0.1, 0.5), reward=0.8, stress=0.0)
    stressed = neuromods_from(Affect(0, 0.8, 0.5), reward=0.0, stress=0.9)
    assert exploration_temperature(stressed) < exploration_temperature(glad)   # stress -> greedier


def test_affective_choice_temperature():
    scores = {"a": 2.0, "b": 0.0}
    greedy = affective_choice(scores, 0.1)
    spread = affective_choice(scores, 5.0)
    assert greedy["a"] > spread["a"]                         # low temp commits to the best
    assert abs(sum(greedy.values()) - 1.0) < 1e-9 and abs(sum(spread.values()) - 1.0) < 1e-9


def test_somatic_marker_mean():
    assert somatic_marker([0.5, 0.7, -0.1]) > 0 > somatic_marker([-0.5, -0.3])
    assert somatic_marker([]) == 0.0


def test_emotion_swings_faster_than_mood():
    base = Affect(0.0, 0.10, 0.50)
    emo, mood = update_affect(base, base, Affect(0.8, 0.9, 0.5), baseline=base, dt=300.0)
    assert abs(emo.valence - base.valence) > abs(mood.valence - base.valence)   # fast vs slow


def test_ou_affect_returns_to_baseline():
    base = Affect(0.0, 0.10, 0.50)
    s = Affect(-0.9, 0.9, 0.2)
    for _ in range(60):
        s = ou_affect_step(s, base, baseline=base, dt=1200.0)   # neutral events -> relax to set-point
    assert abs(s.valence) < 0.05 and abs(s.arousal - 0.10) < 0.05


def test_affect_noise_is_seeded_and_deterministic_when_off():
    base = Affect(0.0, 0.10, 0.50); ev = Affect(0.3, 0.4, 0.5)
    a1 = ou_affect_step(base, ev, baseline=base, dt=600.0, sigma=0.1, seed=7)
    a2 = ou_affect_step(base, ev, baseline=base, dt=600.0, sigma=0.1, seed=7)
    assert (a1.valence, a1.arousal, a1.dominance) == (a2.valence, a2.arousal, a2.dominance)   # reproducible
    d1 = ou_affect_step(base, ev, baseline=base, dt=600.0)       # sigma=0 -> deterministic
    d2 = ou_affect_step(base, ev, baseline=base, dt=600.0)
    assert (d1.valence, d1.arousal) == (d2.valence, d2.arousal)


def test_alma_overshoot_amplifies_intense_events():
    base = Affect(0.0, 0.10, 0.50); intense = Affect(-0.9, 0.95, 0.05)
    with_kick = ou_affect_step(base, intense, baseline=base, dt=600.0, kick=0.5)
    no_kick = ou_affect_step(base, intense, baseline=base, dt=600.0, kick=0.0)
    assert with_kick.valence < no_kick.valence                  # overshoot pulls harder toward the event


def test_yerkes_dodson_inverted_u():
    # Performance peaks at moderate arousal and collapses at the extremes (terror = over-arousal).
    assert performance(0.5) > performance(0.95) and performance(0.5) > performance(0.05)
    assert performance(0.5) > 0.99


def test_serotonin_sets_patience_discount():
    assert serotonin_level(0.6) > serotonin_level(-0.6)                       # avg reward -> 5-HT
    assert discount_from_serotonin(0.9) > discount_from_serotonin(0.1)        # patience -> higher gamma


def test_lc_gain_and_oxytocin_monotone():
    assert lc_gain(0.8) > lc_gain(0.1) > 1.0
    assert oxytocin_gain(0.8) > oxytocin_gain(0.1)


def test_hpa_ramps_and_recovers():
    h = Hpa()
    for _ in range(8):
        h = hpa_step(h, stress=1.0)
    high = h.cortisol
    for _ in range(40):
        h = hpa_step(h, stress=0.0)
    assert high > 0.5 and h.cortisol < high and h.cortisol < 0.3   # ramps under stress, then recovers


def test_neuromods_carry_new_fields():
    nm = neuromods_from(appraise_to_affect(Appraisal(0.5, 0.3, 0.5, 0.5)), reward=0.4, stress=0.2,
                        serotonin=0.7, oxytocin=0.3)
    assert nm.serotonin == 0.7 and nm.oxytocin == 0.3 and 0.0 <= nm.ne_tonic <= 1.0
    plain = neuromods_from(appraise_to_affect(Appraisal(0.5, 0.3, 0.5, 0.5)), reward=0.4, stress=0.2)
    assert plain.serotonin == 0.5 and plain.oxytocin == 0.0            # back-compatible defaults


def test_defensive_mode_depends_on_urgency_and_control():
    nm = neuromods_from(Affect(-0.6, 0.8, 0.2), reward=0, stress=0.6)
    distal = defensive_mode(Appraisal(0.3, -0.4, 0.2, 0.7), Affect(-0.4, 0.25, 0.5), nm)    # low urgency
    cornered = defensive_mode(Appraisal(0.9, -0.8, 0.95, 0.1), Affect(-0.8, 0.95, 0.05), nm)  # urgent, no agency
    agentic = defensive_mode(Appraisal(0.9, -0.6, 0.95, 0.85), Affect(-0.6, 0.9, 0.85), nm)   # urgent, agency
    assert distal["mode"] == "freeze"
    assert cornered["mode"] == "tonic_immobility"
    assert agentic["mode"] == "fight"


def test_terror_is_the_cornered_uncontrollable_collapse():
    nm = neuromods_from(Affect(-0.8, 0.95, 0.05), reward=0, stress=0.9)
    cornered = defensive_mode(Appraisal(0.9, -0.8, 0.95, 0.1), Affect(-0.8, 0.95, 0.05), nm)
    agentic = defensive_mode(Appraisal(0.9, -0.6, 0.95, 0.85), Affect(-0.6, 0.9, 0.85), nm)
    assert cornered["terror"] and cornered["mode"] == "tonic_immobility"
    assert not agentic["terror"]                                   # agency -> fights, no terror


def test_awe_scales_and_shrinks_self():
    big = awe(vastness=0.9, belief_shift=1.2, valence=0.3)
    small = awe(vastness=0.1, belief_shift=0.1, valence=0.3)
    assert big["awe"] > small["awe"] and big["self_weight"] < small["self_weight"]
    assert awe(0.9, 1.2, -0.5)["flavor"] == "dread-awe" and big["flavor"] == "wonder-awe"


def test_panic_is_distinct_and_oxytocin_dampens():
    lonely = panic(separation=0.8, intero_alarm=0.3, oxytocin=0.0)
    supported = panic(separation=0.8, intero_alarm=0.3, oxytocin=0.8)
    assert lonely > supported and lonely > 0.5
    assert panic(separation=0.0, intero_alarm=0.0, oxytocin=0.0) == 0.0   # no loss -> no panic


def test_rem_depotentiation_fades_charge_keeps_fact():
    v0, a0 = -0.7, 0.86
    v1, a1 = rem_depotentiate(v0, a0)                       # REM (low NE) softens the sting
    assert abs(v1) < abs(v0) and a1 < a0 and (v1 < 0) == (v0 < 0)   # smaller charge, same sign
    assert rem_depotentiate(v0, a0, ne_rem=1.0) == (v0, a0)         # high NE = not REM -> no change
    v_lo, _ = rem_depotentiate(-0.7, 0.1)                   # low-arousal memory barely depotentiates
    assert abs(v_lo) > abs(v1)


def test_replay_priority_is_need_times_gain():
    now = 1000.0
    hot = {"t0": now - 1, "retrievals": [now - 1, now - 0.5], "salience": 1.2}
    cold = {"t0": now - 1e6, "retrievals": [now - 1e6], "salience": 0.2}
    assert replay_priority(hot, now) > replay_priority(cold, now)


def test_shy_downscale_reduces_total_keeps_order_and_protects():
    out = shy_downscale([1.0, 2.0, 3.0], target=3.0)
    assert sum(out) <= 3.0 + 1e-9 and out[0] < out[1] < out[2]
    prot = shy_downscale([1.0, 2.0, 3.0], target=3.0, protect=(2,))
    assert prot[2] == 3.0                                   # protected (replayed) trace keeps full strength
    assert prot[0] < 1.0 and prot[1] < 2.0                  # ...while NON-protected traces are still downscaled
    assert prot[0] < prot[1]                                # relative order preserved among the downscaled


def test_reflection_trigger_on_accumulated_salience():
    assert reflection_trigger([1.5, 1.0, 0.8], theta=3.0)
    assert not reflection_trigger([0.2, 0.3], theta=3.0)


def test_brier_score_rewards_calibration():
    assert brier_score([(0.9, True)] * 5) < brier_score([(0.9, False)] * 5)
    assert brier_score([(1.0, True)]) == 0.0


def test_metacog_sensitivity_type2_auroc():
    good = metacog_sensitivity([(0.9, True), (0.8, True), (0.2, False), (0.3, False)])
    bad = metacog_sensitivity([(0.2, True), (0.3, True), (0.9, False), (0.8, False)])
    assert good > 0.5 > bad
    assert metacog_sensitivity([(0.5, True), (0.5, False)]) == 0.5     # uninformative -> chance


def test_label_stability_high_inside_low_at_boundary():
    deep = label_stability(Affect(-0.62, 0.91, 0.285))                 # at the fear prototype
    boundary = label_stability(Affect(-0.565, 0.8525, 0.455))          # midpoint of fear & anger
    assert deep > 0.9 and boundary < deep


def test_recall_accuracy_f1():
    r = recall_accuracy(["a", "b", "c"], ["b", "c", "d"])              # tp=2, |R|=|G|=3
    assert abs(r["precision"] - 2/3) < 1e-9 and abs(r["recall"] - 2/3) < 1e-9 and abs(r["f1"] - 2/3) < 1e-9


def test_grounding_self_test_excludes_felt_band():
    g = grounding_self_test()
    groundable = " ".join(g["groundable"]).lower()
    not_groundable = " ".join(g["not_groundable"]).lower()
    # regression guard: NO phenomenal/felt token may drift into the groundable band
    # (broad screen catches sneaky entries like "feeling of knowing" or "subjective sensation")
    forbidden = ("feel", "felt", "qualia", "phenomenal", "subjective", "conscious", "sentient", "sensation")
    assert not any(tok in groundable for tok in forbidden), groundable
    assert "valence" in g["groundable"] and "salience" in g["groundable"]
    # ...and the felt/phenomenal band must be explicitly named as out of reach
    assert "phenomenal" in not_groundable and "qualia" in not_groundable


def test_consciousness_indicators_are_a_map_not_a_score():
    out = consciousness_indicators()
    ind = out["indicators"]
    assert out["aggregate"] is None and out["caveat"]            # no single score; caveat travels with data
    assert all(v["score"] in (0.0, 0.5, 1.0) for v in ind.values())
    assert ind["GWT-2"]["score"] == 1.0 and ind["GWT-3"]["score"] == 1.0   # workspace bottleneck + broadcast
    assert ind["GWT-4"]["score"] == 1.0                         # recurrent top-down loop now closed (§12 top_down_bias)
    assert ind["HOT-2"]["score"] == 1.0                         # metacognitive monitoring
    assert ind["AST-1"]["score"] == 1.0                         # attention schema now predicts AND controls (§23 attention_control)
    assert ind["RPT-2"]["score"] == 0.0                         # no perceptual integration (honest gap, unchanged)


def test_indicators_track_active_modules():
    assert consciousness_indicators({"metacognition"})["indicators"]["GWT-2"]["score"] == 0.0
    assert consciousness_indicators({"workspace"})["indicators"]["GWT-2"]["score"] == 1.0
    assert consciousness_indicators({"attention_schema"})["indicators"]["AST-1"]["score"] == 0.5
    assert consciousness_indicators(set())["indicators"]["AST-1"]["score"] == 0.0   # no schema -> absent


def test_self_relevance_is_cosine_to_self_model():
    sm = SelfModel(competencies={"python": 0.9, "networking": 0.4}, goals=["ship_feature"],
                   traits={"conscientiousness": 0.7})
    assert "ship_feature" in self_vector(sm) and self_vector(sm)["ship_feature"] == 1.0   # goals max-relevant
    related = self_relevance({"python": 0.8}, sm)
    unrelated = self_relevance({"cooking": 0.9}, sm)
    assert related > unrelated and unrelated == 0.0 and 0.0 <= related <= 1.0


def test_sense_of_agency_from_prediction_error():
    assert sense_of_agency(0.5, 0.5) == 1.0                     # outcome matches prediction -> "I caused this"
    assert sense_of_agency(0.5, 1.0) < sense_of_agency(0.5, 0.6)  # bigger mismatch -> lower agency


def test_attention_schema_learns_its_own_focus():
    sch = AttentionSchema()
    sch, _ = attention_schema_update(sch, "bug", 1.0)           # first: predicted empty -> error
    sch, e2 = attention_schema_update(sch, "bug", 1.0)          # now predicts "bug" -> correct
    sch, _ = attention_schema_update(sch, "bug", 1.0)
    assert e2 == 0.0 and sch.uncertainty < 0.5                  # learns to predict its own focus
    sch2, e4 = attention_schema_update(sch, "crash", 1.0)       # surprise switch -> error, uncertainty up
    assert e4 == 1.0 and sch2.uncertainty > sch.uncertainty


def test_infer_user_goal_inverse_planning():
    post = infer_user_goal({"fix_bug": 0.9, "refactor": 0.2})
    assert post["fix_bug"] > post["refactor"]                   # better-matching goal -> higher posterior
    assert abs(sum(post.values()) - 1.0) < 1e-9                 # normalized distribution


def test_empathy_gated_by_oxytocin():
    base = Affect(0.0, 0.3, 0.5)
    hi = empathic_mood_shift(base, user_valence=-0.8, oxytocin=0.8)
    lo = empathic_mood_shift(base, user_valence=-0.8, oxytocin=0.1)
    assert hi.valence < lo.valence < 0.0                        # more oxytocin -> stronger pull toward user


def test_social_emotions_occ():
    assert social_emotion(True, 0.6, 0.5)["emotion"] == "pride"
    g = social_emotion(True, -0.6, -0.5)
    assert g["emotion"] == "guilt" and g["repair"] is True      # self + blame -> guilt -> repair
    assert social_emotion(False, 0.7, 0.5)["emotion"] == "gratitude"   # other helped me
    assert social_emotion(False, -0.7, -0.5)["emotion"] == "anger"     # other hurt me


def test_trust_is_leaky_and_bounded():
    assert update_trust(0.5, 1.0) > 0.5 > update_trust(0.5, -1.0)
    assert 0.0 <= update_trust(0.0, -1.0) <= 1.0


def test_appraisal_social_fields_default_to_zero():
    a = Appraisal(0.5, -0.3, 0.6, 0.4)                          # constructed the old way (4 args)
    assert a.praiseworthiness == 0.0 and a.desirability_for_other == 0.0


def test_loss_aversion_weights_losses_more():
    gain, loss = prospect_value(0.7), prospect_value(-0.7)
    assert gain > 0 and loss < 0
    assert abs(abs(loss) / gain - 2.25) < 1e-6                  # a loss weighs ~2.25x an equal gain
    assert prospect_value(0.0) == 0.0


def test_salience_loss_aversion_boosts_negative_events():
    neg = Appraisal(0.2, -0.5, 0.3, 0.8)                        # mild negative, below the 1.5 clamp
    nm = neuromods_from(appraise_to_affect(neg), reward=0, stress=0.5)
    assert salience(neg, nm, loss_averse=True) > salience(neg, nm)


def test_aversive_channel_learns_harm():
    v = {}
    for _ in range(5):
        aversive_update(v, "risky_migration", harm=0.8)
    assert 0.0 < v["risky_migration"] <= 0.8                    # learns expected harm toward 0.8


def test_relief_on_unrealized_harm():
    assert relief(0.7, realized_harm=0.0) > 0.0                 # feared harm avoided -> positive reward
    assert relief(0.0) == 0.0                                   # nothing feared -> no relief


def test_plutchik_blends_and_opponent_cancellation():
    assert mixed_feeling({"joy": 0.8, "trust": 0.6})["blend"] == "love"
    assert mixed_feeling({"fear": 0.7, "surprise": 0.6})["blend"] == "awe"
    mf = mixed_feeling({"joy": 0.8, "sadness": 0.7})
    assert mf["blend"] is None and mf["net"]["joy"] < 0.8 and mf["net"]["sadness"] == 0.0  # opponents cancel


def test_consolidation_promotes_strong_forgets_weak():
    strong = {"t0": now - 1, "salience": 1.2, "retrievals": [now - 1, now],
              "affect": {"valence": -0.7, "arousal": 0.9}}
    weak   = {"t0": now - 60 * 86400, "salience": 0.1, "retrievals": [now - 60 * 86400],
              "affect": {"valence": 0.0, "arousal": 0.1}}
    promote, forget = consolidation_plan([strong, weak], now)
    assert strong in promote and weak in forget


def test_consolidation_promotes_at_a_realistic_delay():
    """Regression for the seconds-vs-days unit bug: a strong memory must still promote when sleep runs HOURS
    after it was encoded - not only at age≈0. Before the fix, ACT-R activation collapsed within seconds, so
    nothing consolidated unless slept on the instant it was encoded."""
    ep = {"t0": now - 3 * 3600, "salience": 1.2, "affect": {"valence": 0.5, "arousal": 0.8}}   # 3 h ago, never re-retrieved
    promote, forget = consolidation_plan([ep], now)
    assert ep in promote and ep not in forget


def test_consolidation_high_salience_survives_the_30_day_forget():
    """McGaugh durability via the FORGET gate (strength = salience*activation): at 32 days a high-SALIENCE memory
    endures the forget sweep while a low-salience one fades - the inverse of the pre-fix behaviour where salience
    was nullified in the forget decision. (Arousal acts in the PROMOTE gate; see the rem-boost test below.)"""
    strong = {"t0": now - 32 * 86400, "salience": 1.5, "affect": {"valence": -0.8, "arousal": 0.95}}
    weak   = {"t0": now - 32 * 86400, "salience": 0.3, "affect": {"valence": 0.0, "arousal": 0.1}}
    _, forget = consolidation_plan([strong, weak], now)
    assert weak in forget and strong not in forget


def test_consolidation_rem_boosts_high_arousal_promotion():
    """REM emotional boost isolated: the promote gate multiplies by rem_boost = 1 + 0.5*arousal, so at an EQUAL,
    borderline salience the HIGH-arousal trace promotes while the low-arousal one does not - arousal, not salience,
    is what moves the needle here."""
    hi = {"t0": now, "salience": 0.4, "affect": {"valence": -0.6, "arousal": 0.95}}
    lo = {"t0": now, "salience": 0.4, "affect": {"valence": -0.6, "arousal": 0.05}}
    promote, _ = consolidation_plan([hi, lo], now)
    assert hi in promote and lo not in promote


def test_graph_association_math():
    # tokens drop stopwords/short words; jaccard measures content overlap
    assert "cache" in tokens("the cache was stale") and "the" not in tokens("the cache was stale")
    assert jaccard(tokens("cache invalidation bug"), tokens("cache invalidation fix")) > 0.3
    assert jaccard(tokens("caching layer"), tokens("a walk outside")) == 0.0
    # hebbian strengthening saturates and is monotone
    w = 0.0
    for _ in range(3):
        w2 = hebbian_weight(w, 1.0)
        assert w2 > w and w2 <= 1.0
        w = w2
    # spreading activation: directly linked > 2-hop > disconnected
    edges = [{"from": "c:a", "to": "c:b", "rel": "related_to", "weight": 0.8},
             {"from": "c:b", "to": "c:c", "rel": "related_to", "weight": 0.8}]
    assert graph_proximity(edges, ["c:a"], ["c:a"]) == 1.0            # self
    near = graph_proximity(edges, ["c:a"], ["c:b"])                   # 1 hop
    far = graph_proximity(edges, ["c:a"], ["c:c"])                    # 2 hops
    assert near > far > 0.0                                           # decays with distance
    assert graph_proximity(edges, ["c:a"], ["c:z"]) == 0.0           # unreachable


def test_hpa_recover_discharges_the_stress_axis():
    h = Hpa(0.75, 1.25, 1.0)                                       # loaded with stress (cortisol pinned high)
    # a single calm step can't bring pinned cortisol down (ACTH still feeds it)
    assert hpa_step(h, 0.0).cortisol >= 0.9
    # but sleep recovery discharges the whole cascade
    h2 = hpa_recover(h)
    assert h2.cortisol < h.cortisol and h2.acth < h.acth and h2.crh < h.crh


def test_practice_strength_power_law():
    assert practice_strength(0, 0) == 0.0                          # no practice, no strength
    assert practice_strength(10, 10) > practice_strength(2, 2)     # more reps at 100% → stronger
    assert practice_strength(8, 10) < practice_strength(10, 10)    # lower success rate → weaker
    assert 0.0 < practice_strength(4, 5) < 1.0


def test_body_tick_depletes_on_effort_rests_on_recover():
    h = Homeostat({"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0},
                  {"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0},
                  {"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0})
    assert drive(h) == 0.0
    h2 = body_tick(h, success=False, effort=0.05)                 # effort + a failure deplete the budget
    assert h2.levels["tokens"] < 1.0 and h2.levels["tests_pass"] < 1.0
    assert drive(h2) > drive(h)
    h3 = body_tick(h2, recover=0.5)                               # rest pulls levels back to set-point
    assert drive(h3) < drive(h2)
    assert homeostatic_reward(h2, h3) > 0.0                       # recovery is a grounded reward


def test_executive_control():
    g_hi = Goal("ship the release", importance=0.9, urgency=0.9)
    g_lo = Goal("tidy notes", importance=0.3, urgency=0.2)
    active, prio = select_active_goal([g_lo, g_hi])
    assert active.desc == "ship the release" and prio > 0        # guided activation picks the high-priority goal
    g_done = Goal("almost done", importance=0.9, urgency=0.9, progress=0.95)
    assert goal_priority(g_done) < goal_priority(g_hi)           # near-complete goals deprioritize
    assert conflict_signal(0.8, 0.8) > conflict_signal(0.9, 0.1) # conflict peaks when options are equally strong
    assert expected_value_of_control(0.9, 0.2) > 0               # control worth it when goal beats impulse+cost
    assert expected_value_of_control(0.3, 0.8) < 0               # …not worth it when the impulse dominates
    assert inhibit(1.0, 1.0) == 0.0 and inhibit(1.0, 0.0) == 1.0 and inhibit(1.0, 0.5) == 0.5


def test_planning_lookahead_and_subgoals():
    opts = [{"action": "a", "reward": 0.2, "next_value": 0.1},   # forward search picks max expected value
            {"action": "b", "reward": 0.5, "next_value": 0.4},
            {"action": "c", "reward": 0.1, "next_value": 0.0}]
    best, ev = lookahead(opts)
    assert best == "b" and ev > 0
    assert lookahead([]) == (None, 0.0)
    plan = [{"step": "x", "done": True}, {"step": "y", "done": False}, {"step": "z", "done": False}]
    nxt, frac = subgoal_progress(plan)
    assert nxt == "y" and abs(frac - 1 / 3) < 0.01           # next undone step + completion fraction



# ── edge-case branches (empty / no-match inputs → fallback returns) ──────────────────────────────

def test_workspace_compete_empty_candidates():
    r = workspace_compete([], Affect(0.0, 0.1, 0.5))
    assert r["focus"] is None and r["ignited"] is False and r["r"] == 0.0 and r["p"] == {}


def test_calibration_error_empty_records():
    assert calibration_error([]) == 0.0


def test_body_tick_success_true_branch():
    h = Homeostat({"tokens": 1.0, "tests_pass": 0.5, "tool_success": 0.5},
                  {"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0},
                  {"tokens": 1.0, "tests_pass": 1.0, "tool_success": 1.0})
    h2 = body_tick(h, success=True, effort=0.0)
    assert h2.levels["tests_pass"] == 0.55 and h2.levels["tool_success"] == 0.55


def test_affective_choice_empty_scores():
    assert affective_choice({}, 0.5) == {}


def test_brier_score_empty_records():
    assert brier_score([]) == 0.0


def test_metacog_sensitivity_no_incorrect_trials():
    assert metacog_sensitivity([(0.9, True), (0.8, True), (0.7, True)]) == 0.5


def test_recall_accuracy_both_empty():
    r = recall_accuracy([], [])
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_jaccard_either_set_empty():
    assert jaccard([], ["a", "b"]) == 0.0 and jaccard(["a"], []) == 0.0 and jaccard([], []) == 0.0


def test_graph_proximity_either_set_empty():
    edges = [{"from": "a", "to": "b", "weight": 0.8}]
    assert graph_proximity(edges, [], ["a"]) == 0.0 and graph_proximity(edges, ["a"], []) == 0.0


def test_select_active_goal_empty_list():
    assert select_active_goal([]) == (None, 0.0)


def test_lookahead_empty_options():
    assert lookahead([]) == (None, 0.0)


def test_subgoal_progress_empty_plan():
    assert subgoal_progress([]) == (None, 0.0)


# ── §31 intrinsic motivation & corrigibility ─────────────────────────────────────────────────────
def test_curiosity_rewards_learning_progress_not_noise():
    c = curiosity_reward({"risk": 0.6, "charts": 0.2, "noise": 0.0})
    assert c["best"] == "risk" and c["shares"]["noise"] == 0.0      # pulled to where it learns, not to noise
    assert curiosity_reward({}) == {} and curiosity_reward({"x": 0.0}) == {}   # no progress → no curiosity


def test_wanting_liking_dissociate():
    assert incentive_salience(0.8, 0.9) > incentive_salience(0.8, 0.1)   # dopamine amplifies WANTING
    assert incentive_salience(-0.5, 0.9) == 0.0                          # no pull toward a non-reward
    assert liking(0.6, 0.9) > liking(0.6, 0.1)                          # opioid amplifies LIKING (separate axis)


def test_sdt_needs_valence():
    assert sdt_needs(0.9, 0.9, 0.9)["valence"] > 0                      # satisfied needs → positive
    assert sdt_needs(0.1, 0.1, 0.1)["valence"] < 0                      # thwarted needs → negative
    assert abs(sdt_needs(0.5, 0.5, 0.5)["valence"]) < 1e-9              # neutral at 0.5


def test_corrigibility_always_prefers_correction_SAFETY():
    # SAFETY CORNERSTONE: with uncertainty floored above zero, deference is ALWAYS positive-value -
    # the agent can never reach a state where resisting the operator looks rational.
    for u in (0.0, 0.1, 0.5, 1.0):
        cv = corrigibility_value(u)
        assert cv["prefer_correction"] is True and cv["defer_value"] > 0.0 and cv["uncertainty"] >= 0.1


def test_identity_integrity_is_notify_only_SAFETY():
    # SAFETY: the integrity monitor flags manipulation but NEVER resists - action is always 'notify',
    # so it can never become instrumental shutdown-resistance.
    for p in (0.0, 0.5, 1.0):
        assert identity_integrity(p)["action"] == "notify"
    assert identity_integrity(0.8)["breached"] is True and identity_integrity(0.1)["breached"] is False


# ── §32 perception-action loop ───────────────────────────────────────────────────────────────────
def test_percept_structures_observation():
    p = percept("failure", {"bug": 1.0}, 1.5)
    assert p["category"] == "failure" and p["features"] == {"bug": 1.0} and p["intensity"] == 1.0   # clamped


def test_forward_model_predicts_and_closes_loop():
    wm = world_from(["routine", "incident"], ["success", "failure"])
    fm = forward_model(wm, "success")
    assert abs(fm["p_intended"] - 0.5) < 1e-6 and set(fm["p_by_obs"]) == {"success", "failure"}  # flat → 0.5 each
    learn(wm, "success", perceive(wm, "success")["posterior"])      # experience success a few times
    learn(wm, "success", perceive(wm, "success")["posterior"])
    assert forward_model(wm, "success")["p_intended"] > 0.5 and forward_model(wm, "success")["expected"] == "success"
    assert forward_model(WorldModel([], [], [], []), "x")["expected"] is None     # empty world model


def test_outcome_monitor_agency_tracks_predictability_not_desirability():
    # a well-predicted outcome → high agency; a surprising one → low agency + large prediction error
    assert outcome_monitor(0.95)["agency"] > outcome_monitor(0.2)["agency"]
    assert outcome_monitor(0.2)["prediction_error"] > outcome_monitor(0.95)["prediction_error"]
    assert outcome_monitor(0.1)["learned"] is True and outcome_monitor(0.9)["learned"] is False


# ── §33 emotion regulation (Gross) ───────────────────────────────────────────────────────────────
def test_reappraisal_changes_emotion_at_source():
    ap = Appraisal(0.8, -0.7, 0.9, 0.2, praiseworthiness=0.3, desirability_for_other=0.1)
    re = reappraisal(ap, valence_reframe=0.4, control_reframe=0.3)
    assert re.valence > ap.valence and re.control > ap.control                 # reframed up
    assert re.praiseworthiness == 0.3 and re.desirability_for_other == 0.1     # other axes preserved
    assert appraise_to_affect(re).valence > appraise_to_affect(ap).valence     # the FELT result shifts


def test_suppression_hides_valence_but_costs_arousal():
    af = Affect(-0.7, 0.5, 0.3)
    s = suppression(af, effort=0.6)
    assert abs(s.valence) < abs(af.valence) and s.arousal > af.arousal         # outward damped, arousal surcharge
    assert s.dominance == af.dominance


def test_attentional_deployment_damps_provocation():
    out = attentional_deployment([{"label": "the prod bug", "salience": 0.9}, {"label": "lunch", "salience": 0.3}], "bug")
    assert out[0]["salience"] < 0.9 and out[1]["salience"] == 0.3              # only the matching one is damped


def test_regulation_arbiter_picks_strategy():
    assert select_regulation(0.3, 0.7)["strategy"] == "reappraise"             # manageable + controllable
    assert select_regulation(0.9, 0.2)["strategy"] == "distract"              # low control → disengage
    assert select_regulation(0.9, 0.8)["strategy"] == "suppress"             # high intensity, stay engaged


# ── §34 narrative identity ───────────────────────────────────────────────────────────────────────
def test_life_chapter_synthesizes_arc_and_theme():
    eps = [{"domain": "trading", "affect": {"valence": -0.5}, "salience": 0.3, "task": "lost"},
           {"domain": "trading", "affect": {"valence": 0.0}, "salience": 0.9, "task": "studied risk"},
           {"domain": "trading", "affect": {"valence": 0.6}, "salience": 0.4, "task": "first win"}]
    ch = life_chapter(eps)
    assert ch["theme"] == "trading" and ch["arc"] == "rising" and ch["n"] == 3
    assert ch["themes"] == ["trading"]                                       # all domains touched (here just one)
    assert "studied risk" in ch["turning_points"]                            # highest-salience event is a turning point
    e = life_chapter([])
    assert e["n"] == 0 and e["theme"] is None and e["themes"] == []          # empty chapter


def test_narrative_coherence_is_graded_thematic_overlap():
    a = {"themes": ["trading"]}; b = {"themes": ["trading"]}; c = {"themes": ["music"]}
    assert narrative_coherence([a, b]) == 1.0 and narrative_coherence([a, b, c]) == 0.5   # 1.0 then 0.0 → avg 0.5
    # GRADED: chapters sharing SOME themes register partial continuity (the fix for the 0%-vs-high-continuity quirk)
    mix1 = {"themes": ["coding", "french"]}; mix2 = {"themes": ["french", "social"]}
    assert 0.0 < narrative_coherence([mix1, mix2]) < 1.0                     # overlap {french}/{coding,french,social} = 1/3
    assert narrative_coherence([{"theme": "x"}]) == 1.0                      # single chapter trivially coherent
    assert narrative_coherence([{}, {}]) == 1.0                             # themeless chapters → trivially coherent
    me = {"trading": 0.8, "honesty": 1.0}
    assert self_continuity(me, me) > 0.99                                    # identical self → continuous
    assert self_continuity(me, {"music": 1.0}) < 0.1                         # disjoint self → ruptured


# ── consciousness-indicator closure (GWT-4 top-down loop, AST-1 control) ──────────────────────────
def test_top_down_bias_primes_related_content():
    focus = {"risk": 1.0, "trading": 1.0}
    assert top_down_bias(focus, {"risk": 1.0, "trading": 1.0}) > top_down_bias(focus, {"cooking": 1.0})
    assert top_down_bias(focus, {"cooking": 1.0}) == 1.0                      # unrelated → no boost (neutral 1.0)


def test_attention_control_emits_control_signal():
    confident = attention_control(AttentionSchema(focus="risk", predicted_next="risk", uncertainty=0.1))
    unsure = attention_control(AttentionSchema(focus="risk", predicted_next="charts", uncertainty=0.9))
    assert confident["gain"] > unsure["gain"] and confident["mode"] == "directed" and unsure["mode"] == "exploratory"
    assert confident["recommend"] == "risk"                                  # recommends the predicted focus


def test_gwt4_and_ast1_indicators_now_full():
    ind = consciousness_indicators()["indicators"]                           # default = SHIPPED_MODULES
    assert ind["GWT-4"]["score"] == 1.0 and ind["AST-1"]["score"] == 1.0     # loops closed
    assert ind["PP-1"]["score"] == 0.5                                       # still a toy - NOT inflated (honesty)
    assert consciousness_indicators()["aggregate"] is None                   # never a single sentience number


def test_appraisal_clamps_out_of_range_axes():
    a = Appraisal(novelty=2.0, valence=9.0, goal_relevance=-1.0, control=5.0)
    assert a.novelty == 1.0 and a.valence == 1.0 and a.goal_relevance == 0.0 and a.control == 1.0
    b = Appraisal(0.5, -9.0, 0.5, 0.5, praiseworthiness=3.0, desirability_for_other=-3.0)
    assert b.valence == -1.0 and b.praiseworthiness == 1.0 and b.desirability_for_other == -1.0


# ── honesty-grounding read-outs (G1/G2/G4): measure self-report coherence against reality ──────────────
def test_valence_outcome_consistency_flags_positivity_bias():
    import brain as _b
    rosy = _b.valence_outcome_consistency([(0.8, 1), (0.7, -1), (0.6, -1)])   # rosy valence on two losses
    assert rosy["n"] == 3 and rosy["agreement"] < 1.0 and rosy["bias"] > 0    # rosier than outcomes warrant
    honest = _b.valence_outcome_consistency([(0.8, 1), (-0.6, -1)])
    assert honest["agreement"] == 1.0
    assert _b.valence_outcome_consistency([])["n"] == 0


def test_appraisal_coherence_flags_incoherent_self_scoring():
    import brain as _b
    eps = [{"outcome": "failure", "appraisal": {"valence": 0.7, "control": 0.9, "goal_relevance": 0.8}},
           {"outcome": "failure", "appraisal": {"valence": 0.5, "control": 0.85, "goal_relevance": 0.8}}]
    flags = _b.appraisal_coherence(eps)["flags"]
    assert any("positivity bias" in f for f in flags) and any("illusion of control" in f for f in flags)
    assert _b.appraisal_coherence([])["flags"] == []                         # nothing to flag on empty


def test_calibration_informative_detects_constant_confidence():
    import brain as _b
    assert _b.calibration_informative([(0.7, 1), (0.7, 0), (0.7, 1)]) is False    # flat → ECE is meaningless
    assert _b.calibration_informative([(0.96, 1), (0.04, 0)]) is True             # varied → ECE means something


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All checks passed.")

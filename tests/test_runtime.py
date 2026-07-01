"""Checks that the agent runtime wires the engine into a living, developing agent.
Run: python3 test_runtime.py   (in-memory; the persistence test uses a temp dir)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import os
import shutil
import tempfile
import time

import brain as B
from runtime import Brain, research_session

NOW = 1_750_000_000.0
AP = lambda nov, val, gr, ctl: B.Appraisal(nov, val, gr, ctl)


def fresh():
    b = Brain(root=None)
    b.personality = B.Personality(openness=0.7, neuroticism=0.5)
    return b


def test_perceive_encodes_and_returns_feeling():
    b = fresh()
    r = b.perceive("a surprising prod bug", AP(0.9, -0.7, 0.9, 0.2), domain="debug",
                   outcome="failure", reward=-0.5, cue="bug", now=NOW)
    assert len(b.episodes) == 1
    assert r["label"] == "fear" and r["feeling"] == "terror"          # high-arousal, low-control negative
    assert b.episodes[0]["salience"] > 0 and b.episodes[0]["feeling"]["word"] == "terror"


def test_perceive_always_surfaces_awe_panic_relief_signals():
    """§19/§25 named-feeling circuits are now COMPUTED every encode (not just in the demo) and present in output."""
    b = fresh()
    r = b.perceive("an event", AP(0.9, -0.6, 0.8, 0.2), outcome="failure", cue="x", now=NOW)
    for k in ("awe", "panic", "relief"):
        assert k in r and 0.0 <= r[k] <= 1.0
    assert r["awe_flavor"] in ("wonder-awe", "dread-awe")


def test_relief_fires_when_an_expected_harm_does_not_materialise():
    b = fresh()
    for i in range(2):                                                 # learn to expect harm from the 'deploy' cue
        b.perceive("deploy crashed", AP(0.5, -0.8, 0.8, 0.3), outcome="failure", cue="deploy", now=NOW + i)
    r = b.perceive("deploy went fine", AP(0.3, 0.4, 0.6, 0.7), outcome="success", cue="deploy", now=NOW + 10)
    assert r["relief"] > 0.0                                           # the non-harmful outcome emits opponent-process relief


def test_recall_downweights_imagined_vs_observed():
    """§13 PRM reality weight: an imagined trace recalls weaker than an equally-relevant observed one."""
    b = fresh()
    b.perceive("a fact about quarks", AP(0.5, 0.3, 0.6, 0.6), source="observed", now=NOW)
    b.perceive("a fact about quarks", AP(0.5, 0.3, 0.6, 0.6), source="imagined", now=NOW)
    hits = b.recall(lambda e: 1.0, k=2, now=NOW)                       # identical relevance/recency → only source differs
    assert hits[0]["task"] == "a fact about quarks" and hits[0]["score"] > hits[1]["score"]


def test_lookahead_picks_highest_value_action():
    b = fresh()
    b.V["good"] = 0.8; b.V["bad"] = 0.1
    action, value = b.lookahead(["bad", "good"])                       # §30 forward search over the §10 value cache
    assert action == "good" and value > 0


def test_rpe_colours_valence_relief_better_than_expected():
    """§10: a reward far BETTER than the learned expectation lifts the encoded affect (relief/elation) above
    the raw appraisal valence - the reward-prediction error colours feeling, not just dopamine."""
    b = fresh()
    b.perceive("first try, it went badly", AP(0.5, 0.0, 0.6, 0.5), cue="t", reward=-0.4, outcome="failure", now=NOW)
    b.perceive("then a surprise success", AP(0.5, 0.0, 0.6, 0.5), cue="t", reward=0.8, outcome="success", now=NOW + 10)
    assert b.episodes[-1]["affect"]["valence"] > 0.0           # raw appraisal valence was 0.0; the positive RPE lifted it


def test_brain_tolerates_a_malformed_episodic_line():
    """A torn/garbage line in events.jsonl is SKIPPED, not fatal - the rest of the brain still loads."""
    import json
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "episodic"), exist_ok=True)
    good = {"id": "e-0001", "t0": NOW, "task": "a valid memory", "outcome": None, "files": [],
            "appraisal": {"novelty": 0.5, "valence": 0.2, "goal_relevance": 0.5, "control": 0.5},
            "affect": {"valence": 0.2, "arousal": 0.3, "dominance": 0.5},
            "feeling": {"label": "calm", "word": "calm", "intensity": 0.3}, "salience": 0.4}
    with open(os.path.join(tmp, "episodic", "events.jsonl"), "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write("{ this is not valid json ::: \n")                    # torn/garbage line
        f.write(json.dumps({**good, "id": "e-0002"}) + "\n")
    try:
        b = Brain(root=tmp)
        assert len(b.episodes) == 2 and b.episodes[0]["id"] == "e-0001"   # garbage line skipped, the 2 valid kept
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stress_consolidation_scales_and_mood_stays_bounded():
    """Encode a 60-event barrage, then sleep: consolidation runs at scale and mood never leaves [-1,1]."""
    b = fresh()
    topics = ["the parser crashed hard", "the cache leaked memory", "the scheduler stalled badly",
              "the network stack dropped packets", "the renderer flickered on resize"]
    for i in range(60):
        b.perceive(f"{topics[i % 5]} on run {i}", AP(0.7, (0.8 if i % 2 else -0.8), 0.7, 0.3),
                   domain="work", cue=f"t{i % 5}", now=NOW + i * 0.1)
    assert len(b.episodes) == 60
    res = b.sleep(now=NOW + 6.0)                                   # sleep right after the barrage (traces still active)
    assert res["episodes"] == 60 and "reflections" in res         # consolidation ran at scale, episodic preserved
    assert res["promoted"] >= 1 and len(b.facts) >= 3             # distinct topics harden; interference merges restatements
    assert -1.0 <= b.mood.valence <= 1.0 and 0.0 <= b.mood.arousal <= 1.0   # bounded under the barrage


def test_value_loop_learns_across_perceives():
    b = fresh()
    for i in range(15):
        b.perceive("tests pass", AP(0.3, 0.4, 0.5, 0.8), outcome="success", reward=1.0,
                   cue="tests_pass", now=NOW + i * 600)
    assert b.V["tests_pass"] > 0.6                                    # value learned toward the reward


def test_competence_develops_up_on_wins_down_on_losses():
    b = fresh()
    base = b.efficacy.get("py", 0.5)
    for i in range(8):
        b.perceive("ship a feature", AP(0.4, 0.5, 0.7, 0.7), domain="py", outcome="success",
                   reward=0.7, cue="ship", now=NOW + i * 600)
    up = b.efficacy["py"]
    for i in range(8):
        b.perceive("regress", AP(0.5, -0.5, 0.7, 0.3), domain="py", outcome="failure",
                   reward=-0.5, cue="regress", now=NOW + (20 + i) * 600)
    assert up > base and b.efficacy["py"] < up                       # grows on wins, falls on losses


def test_surprise_is_neutral_for_competence():
    b = fresh()
    before = b.efficacy.get("x", b.default_efficacy)
    b.perceive("a surprising result", AP(0.8, 0.0, 0.6, 0.5), domain="x", outcome="surprise", now=NOW)
    assert "x" not in b.efficacy or b.efficacy["x"] == before         # surprise doesn't move competence


def test_sleep_promotes_strong_and_keeps_recent():
    b = fresh()
    for i in range(6):
        b.perceive("important insight", AP(0.8, 0.6, 0.9, 0.7), domain="ai", outcome="insight",
                   reward=0.7, cue="ai", now=NOW + i * 600)
    n_ep, n_fact = len(b.episodes), len(b.facts)
    s = b.sleep(now=NOW + 5 * 600)                                   # sleep while the last trace is still active
    assert s["promoted"] >= 1 and len(b.facts) > n_fact              # strong traces -> semantic facts
    assert len(b.episodes) == n_ep                                   # recent episodes kept (not forgotten)


def test_recall_ranks_relevant_first():
    b = fresh()
    b.perceive("fix the network retry loop", AP(0.6, -0.4, 0.8, 0.5), now=NOW)
    b.perceive("update the README typo", AP(0.1, 0.1, 0.2, 0.9), now=NOW + 600)
    hits = b.recall(lambda e: 1.0 if "network" in e["task"] else 0.1, k=1, now=NOW + 1200)
    assert hits and "network" in hits[0]["task"]                     # the relevant memory surfaces first


def test_recall_weight_override_is_relevance_first():
    """recall(w=...) lets a 'search' weight meaning over mood/recency/salience: an old, low-salience but
    ON-topic memory must beat a recent, very high-salience OFF-topic one. This is the seam the CLI's
    `recall --search` and the optional dense-semantic term ride on."""
    b = fresh()
    b.perceive("discounted cash flow valuation of a company", AP(0.1, 0.1, 0.2, 0.9), now=NOW)        # old, faint
    b.perceive("a shocking thrilling jackpot win", AP(0.95, 0.95, 0.95, 0.2), now=NOW + 100_000)       # recent, loud
    rel = lambda e: 1.0 if "valuation" in e["task"] else 0.2
    hits = b.recall(rel, k=1, now=NOW + 100_100, w=(0.05, 0.10, 0.80, 0.05, 0.0))
    assert hits and "valuation" in hits[0]["task"]                   # relevance-first weights surface meaning


def test_planning_sets_plan_advances_and_drives_goal_progress():
    b = fresh()
    b.add_goal("ship v1", importance=0.9, urgency=0.9)
    b.set_plan("ship v1", ["design", "build", "test", "ship"])
    s = b.next_step()
    assert s["goal"] == "ship v1" and s["next"] == "design" and s["complete"] == 0.0
    b.advance_plan()                                         # complete "design"
    s2 = b.next_step()
    assert s2["next"] == "build" and s2["complete"] == 0.25
    assert b.active_goal()[0].progress == 0.25              # plan completion drives goal progress


def test_executive_active_goal_and_self_control():
    b = fresh()
    b.add_goal("finish the brain project", importance=0.9, urgency=0.8)
    b.add_goal("a minor distraction", importance=0.2, urgency=0.2)
    g, prio = b.active_goal()
    assert g.desc == "finish the brain project"                  # the strong goal holds the executive
    d1 = b.deliberate("check social media", 0.2)                 # weak impulse vs strong goal
    assert d1["evc"] > 0 and "control" in d1["decision"] and d1["residual_impulse"] < 0.2  # inhibited
    d2 = b.deliberate("flee a real danger", 0.95)               # overwhelming impulse
    assert d2["evc"] <= 0 and "impulse" in d2["decision"]        # the prepotent response stands


def test_empathize_records_user_affect_and_shifts_mood():
    b = fresh(); b.user["trust"] = 0.9; b.mood = B.Affect(0.1, 0.3, 0.5)
    out = b.empathize(0.9)                                  # user feels strongly positive
    assert b.user["inferred_affect"]["valence"] == 0.9     # recorded (was inert before)
    assert out["oxytocin"] > 0 and b.mood.valence > 0.1    # my mood pulled UP toward theirs (empathy)
    b2 = fresh(); b2.user["trust"] = 0.9; b2.mood = B.Affect(0.1, 0.3, 0.5)
    b2.empathize(-0.9)
    assert b2.mood.valence < 0.1                           # pulled DOWN toward their negative affect


def test_infer_goal_accumulates_into_user_model():
    b = fresh()
    assert b.user["inferred_goals"] == {}                  # inert at start
    b.infer_goal("ship the project")
    b.infer_goal("ship the project")                       # repeated evidence strengthens
    assert b.user["inferred_goals"]["ship the project"] > 0


def test_working_memory_persists_across_reloads_bounded_to_span():
    tmp = tempfile.mkdtemp()
    try:
        for i in range(10):
            b = Brain(root=tmp)                                   # each reload = a separate CLI invocation
            b.note(f"item {i}")
        b = Brain(root=tmp)
        items = [l for l in b.working.splitlines() if l.strip().startswith("- ")]
        assert 5 <= len(items) <= 8                              # Miller's 7±2 - not 1 (recency-only) nor 10
        assert "item 9" in b.working                             # keeps the most recent
        b.sleep()                                                # sleep wipes working memory (disposable)
        b2 = Brain(root=tmp)
        assert not any(l.strip().startswith("- ") for l in b2.working.splitlines())
    finally:
        shutil.rmtree(tmp)


def test_incremental_episodic_save_is_byte_consistent_through_appends_and_mutations():
    """The incremental save (append-only fast path + dirty-flag full rewrite) must NEVER corrupt the log: after
    any mix of pure appends and mutations (recall bump, forget), the on-disk events.jsonl must equal a full
    re-serialization of the live episodes, and a fresh reload must see the same episodes."""
    import json as _json
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        path = os.path.join(tmp, "episodic/events.jsonl")
        rebuilt = lambda br: "".join(_json.dumps(e) + "\n" for e in br.episodes)
        for i in range(8):                                       # appends across separate saves → exercises the tail-append fast path
            b.perceive(f"event {i}", AP(0.5, 0.4, 0.6, 0.5), domain="d", now=NOW + i)
            b.save()
        assert open(path, encoding="utf-8").read() == rebuilt(b)         # fast-append never corrupts the prefix
        b.recall(now=NOW + 100); b.persist_recall()                      # mutation: retrieval bump → full rewrite
        assert open(path, encoding="utf-8").read() == rebuilt(b)
        b.forget(b.episodes[0]["id"])                                    # mutation: remove a line
        b.perceive("after forget", AP(0.5, 0.4, 0.6, 0.5), domain="d", now=NOW + 200); b.save()
        assert open(path, encoding="utf-8").read() == rebuilt(b)         # consistent again after remove + append
        assert [e["id"] for e in Brain(root=tmp).episodes] == [e["id"] for e in b.episodes]   # a fresh reload agrees
    finally:
        shutil.rmtree(tmp)


def test_fired_intents_resurface_on_cue():
    """Prospective memory resurfaces ON CUE: an intention fires when a matching event arrives, not only at wake."""
    b = fresh()
    b.intend("when the parser breaks", "rewrite the tokenizer")
    b.intend("when you ship a release", "update the changelog")
    fired = b.fired_intents("the parser is broken again", None, "dev")   # 'parser' matches the first trigger
    assert [x["intent"] for x in fired] == ["rewrite the tokenizer"]
    assert b.fired_intents("an entirely unrelated moment") == []         # no token overlap → nothing fires


def test_reconsolidation_strengthens_a_near_duplicate_trace():
    """A near-duplicate re-encounter bumps the prior trace's retrievals (reconsolidation) while still appending
    the new episode (append-only preserved)."""
    b = fresh()
    b.perceive("the cache invalidation bug bit me again", AP(0.5, -0.4, 0.6, 0.4), domain="d", now=NOW)
    n_before = len(b.episodes[0]["retrievals"])
    b.perceive("the cache invalidation bug bit me again today", AP(0.5, -0.4, 0.6, 0.4), domain="d", now=NOW + 5)
    assert len(b.episodes[0]["retrievals"]) == n_before + 1              # the prior trace was reinforced
    assert len(b.episodes) == 2                                          # …and the new episode still lands


def test_dedup_merges_near_duplicate_facts():
    """Interference reduction: a near-duplicate restatement of a fact merges at sleep; distinct knowledge stays."""
    b = fresh()
    b.facts = [{"id": "f-0", "text": "the database connection pool should be sized to twice the worker count"},
               {"id": "f-1", "text": "the database connection pool should be sized to twice the worker count exactly"},
               {"id": "f-2", "text": "espresso needs a fine grind and a short shot"}]
    b._dedup_facts()
    assert len(b.facts) == 2                                             # the restatement merged
    assert any("espresso" in f["text"] for f in b.facts)                # distinct knowledge kept


def test_runtime_consolidation_promotes_at_a_realistic_delay():
    """INTEGRATION guard for the ACT-R unit fix at the sleep() wiring (not only the brain.py function): a strong
    memory encoded HOURS before sleep must still promote. Under the pre-fix epoch-seconds activation this
    promoted NOTHING - the exact class of bug that shipped, here caught at the runtime layer."""
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        for i, w in enumerate(["breakthrough", "discovery", "insight", "result"]):
            b.perceive(f"a major {w} in today's deep work", AP(0.7, 0.8, 0.85, 0.6),
                       domain="d", outcome="insight", confidence=0.9, now=NOW + i)
        before = len(b.facts)
        b.sleep(now=NOW + 6 * 3600)                                      # sleep 6 HOURS after encoding (realistic gap)
        assert len(b.facts) > before                                    # strong memories still consolidate at a realistic delay
    finally:
        shutil.rmtree(tmp)


def test_incremental_save_uses_the_fast_append_path():
    """Guard the O(new-lines) optimization itself: after a full save, a pure append must stay non-dirty (fast
    path eligible) and the save must advance the saved-count via append, not fall back to a full rewrite."""
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.perceive("alpha topic one stands alone", AP(0.5, 0.4, 0.6, 0.5), domain="d", now=NOW); b.save()
        assert b._events_dirty is False and b._events_saved_n == 1
        b.perceive("beta topic two unrelated entirely", AP(0.5, 0.4, 0.6, 0.5), domain="d", now=NOW + 1)
        assert b._events_dirty is False                                 # a pure append did NOT mark dirty
        b.save()
        assert b._events_saved_n == len(b.episodes) == 2                # fast path advanced the saved-count
        assert open(os.path.join(tmp, "episodic/events.jsonl"), encoding="utf-8").read().count("\n") == 2
    finally:
        shutil.rmtree(tmp)


def test_fact_id_does_not_collide_after_mid_list_dedup():
    """Regression: f-{len} collided after a near-dup removal shrank the fact list; _next_fact_id (1+max suffix)
    must keep ids unique even then."""
    b = fresh()
    b.facts = [{"id": "f-0000", "text": "always size the pool to twice the worker count"},
               {"id": "f-0001", "text": "always size the pool to twice the worker count exactly"},  # near-dup of f-0000
               {"id": "f-0002", "text": "espresso wants a fine even grind"}]
    b._dedup_facts()                                                    # near-dup removed → ids {f-0000, f-0002}, len 2
    b.learn("a brand new distinct fact about gardening tomatoes")
    ids = [f["id"] for f in b.facts]
    assert "f-0003" in ids                                              # 1+max(0,2), NOT len-based f-0002 (which would collide)
    assert len(set(ids)) == len(ids)                                    # all ids unique


def test_reconsolidation_does_not_fire_below_threshold():
    """Negative case: a re-encounter with low overlap (jaccard < 0.6) must NOT bump the prior trace."""
    b = fresh()
    b.perceive("the parser tokenizer rewrite landed cleanly", AP(0.5, 0.5, 0.6, 0.6), domain="d", now=NOW)
    n = len(b.episodes[0]["retrievals"])
    b.perceive("the garden tomatoes ripened well in the sun", AP(0.5, 0.5, 0.6, 0.6), domain="d", now=NOW + 5)
    assert len(b.episodes[0]["retrievals"]) == n                       # unrelated → no reconsolidation
    assert len(b.episodes) == 2


def test_dedup_keeps_facts_below_the_merge_threshold():
    """Boundary: two facts with moderate overlap (below 0.85 jaccard) must BOTH survive - only very-high-overlap
    restatements merge, so distinct claims are never lost."""
    b = fresh()
    b.facts = [{"id": "f-0", "text": "the cache must be invalidated on every write operation"},
               {"id": "f-1", "text": "the cache must be warmed on every read request instead"}]
    b._dedup_facts()
    assert len(b.facts) == 2                                            # distinct claims kept


def test_fired_intents_match_via_cue_and_domain_and_multi():
    """Prospective fires via the cue and the domain channels (not only the task string), and returns all matches."""
    b = fresh()
    b.intend("when fusion appears", "revisit the EAST result")
    b.intend("when the deploy stage runs", "check the migration")
    assert [x["intent"] for x in b.fired_intents("an unrelated note", "fusion", None)] == ["revisit the EAST result"]
    assert [x["intent"] for x in b.fired_intents("nothing relevant", None, "deploy")] == ["check the migration"]
    b.intend("when fusion comes up", "also email the team")             # a second intent sharing 'fusion'
    assert len(b.fired_intents("x", "fusion", None)) == 2               # multi-match returns both


def test_global_workspace_ignites_salient_stays_local_trivial():
    b = fresh()
    b.perceive("a major breakthrough on the core problem", AP(0.8, 0.7, 0.95, 0.6), now=NOW)
    assert b.workspace["ignited"] and b.workspace["focus"] and b.workspace["r"] > 0.5   # reached the broadcast stage
    b.perceive("noticed the wall is a beige colour", AP(0.1, 0.0, 0.05, 0.9), now=NOW + 600)
    assert not b.workspace["ignited"] and b.workspace["focus"] is None                  # sub-threshold → stays local
    b.perceive("another big result lands", AP(0.8, 0.6, 0.9, 0.6), now=NOW + 1200)
    assert b.workspace["ignited"]
    b.sleep(now=NOW + 1300)
    assert not b.workspace["ignited"]                                                   # quiet stage after sleep


def test_sleep_distills_playbooks_from_success_clusters():
    b = fresh()
    for i in range(4):
        b.perceive("shipped a clean fix", AP(0.4, 0.6, 0.7, 0.8), domain="coding", outcome="success",
                   reward=0.7, cue=f"fix-{i}", now=NOW + i * 600)
    b.perceive("a regression slipped in", AP(0.5, -0.5, 0.7, 0.3), domain="coding", outcome="failure",
               reward=-0.4, cue="regress", now=NOW + 5 * 600)
    assert not b.playbooks
    b.sleep(now=NOW + 5 * 600)
    pb = next((p for p in b.playbooks if p["domain"] == "coding"), None)
    assert pb and pb["successes"] == 4 and pb["attempts"] == 5    # distilled a how-to with a track record
    assert 0.0 < pb["strength"] <= 1.0 and pb["steps"]           # + a power-law-of-practice strength


def test_prospective_intentions_persist_and_surface_at_wake():
    b = fresh()
    iid = b.intend("Friday", "ship Project Lighthouse")
    assert b.pending_intents()[0]["id"] == iid
    assert "ship Project Lighthouse" in b.wake()                 # a commitment resurfaces at wake
    assert b.complete_intent(iid) and not b.pending_intents()    # and can be retired


def test_body_budget_depletes_on_effort_and_rests_at_sleep():
    b = fresh()
    keys = ["tokens", "compute", "context_free", "tests_pass", "tool_success", "user_approval"]
    b.body = B.Homeostat({k: 1.0 for k in keys}, {k: 1.0 for k in keys}, {k: 1.0 for k in keys})
    assert B.drive(b.body) == 0.0                                # starts rested
    for i in range(8):
        b.perceive("hard focused work that isn't going well", AP(0.5, -0.3, 0.8, 0.4), domain="x",
                   outcome="failure", reward=-0.3, cue="w", now=NOW + i * 60)
    strained = B.drive(b.body)
    assert strained > 0.0                                        # living depleted the body-budget
    b.sleep(now=NOW + 8 * 60)
    assert B.drive(b.body) < strained                           # rest (sleep) restored it


def test_sleep_grows_association_graph_from_living():
    b = fresh()
    # three related debugging episodes (shared domain + overlapping content) + one unrelated
    b.react("the caching layer returned stale cache data", 0.0, 0.6, 0.6, domain="debugging",
            outcome="insight", reward=0.3, cue="caching", now=NOW)
    b.react("cache invalidation was the real cache bug", -0.2, 0.7, 0.5, domain="debugging",
            outcome="insight", reward=0.3, cue="invalidation", now=NOW + 600)
    b.react("stale cache data corrupted a total", -0.3, 0.6, 0.5, domain="debugging",
            outcome="failure", reward=-0.4, cue="staleness", now=NOW + 1200)
    b.react("a relaxing walk in the park", 0.6, 0.2, 0.9, domain="life",
            outcome="success", reward=0.5, cue="walk", now=NOW + 1800)
    assert not b.graph["edges"]                                      # graph empty before sleep
    b.sleep(now=NOW + 1900)
    assert b.graph["nodes"] and b.graph["edges"]                     # it grew from lived experience
    cache, inval = b._nid("c:", "caching"), b._nid("c:", "invalidation")
    prox = B.graph_proximity(b.graph["edges"], [cache], [inval])
    assert prox > 0.0                                               # related concepts are reachable (≥ via domain hub)
    walk = b._nid("c:", "walk")
    assert B.graph_proximity(b.graph["edges"], [cache], [walk]) < prox   # the unrelated concept is not


def test_react_computes_novelty_from_world_model_and_habituates():
    b = fresh()
    r1 = b.react("a brand-new kind of event", 0.5, 0.6, 0.7, outcome="insight", reward=0.5, cue="x", now=NOW)
    assert len(b.episodes) == 1
    # the episode's stored novelty IS the world-model-computed surprise, not a self-scored number
    assert abs(b.episodes[0]["appraisal"]["novelty"] - r1["novelty_computed"]) < 1e-9
    first = r1["novelty_computed"]
    for i in range(6):
        b.react("the same kind again", 0.3, 0.4, 0.6, outcome="insight", reward=0.3, cue="x", now=NOW + (i + 1) * 600)
    assert b.episodes[-1]["appraisal"]["novelty"] < first        # the world model habituated → less surprising
    assert b.V["x"] > 0                                          # value still learns through react


def test_preview_does_not_encode():
    b = fresh()
    p = b.preview(AP(0.8, -0.6, 0.9, 0.2))                            # a hard, low-control negative
    assert p["feeling"] and p["salience"] > 0 and len(b.episodes) == 0  # a dry run leaves memory untouched


def test_learn_adds_durable_fact_directly():
    b = fresh()
    fid = b.learn("the owner prefers Python over Swift", confidence=0.95)
    assert any(f["id"] == fid and f["confidence"] == 0.95 for f in b.facts)  # fact stored without an episode
    assert len(b.episodes) == 0


def test_forget_drops_an_episode():
    b = fresh()
    b.perceive("a throwaway event", AP(0.3, 0.1, 0.2, 0.8), now=NOW)
    eid = b.episodes[0]["id"]
    assert b.forget(eid) and len(b.episodes) == 0                     # the sanctioned exception to append-only
    assert b.forget("e-9999") is False                               # forgetting a missing id is a no-op


def test_note_keeps_a_bounded_working_set():
    b = fresh()
    for i in range(10):
        b.note(f"thought {i}")
    items = [ln for ln in b.working.splitlines() if ln.strip().startswith("- ")]
    assert len(items) <= 7 and "thought 9" in b.working               # working memory stays small, keeps newest


def test_feel_and_why_are_honest_strings():
    b = fresh()
    b.perceive("a good win", AP(0.5, 0.7, 0.8, 0.8), outcome="success", reward=0.8, now=NOW)
    assert "felt" in b.feel().lower() or "functional" in b.feel().lower()   # honesty disclaimer present
    assert "mood" in b.why().lower()


def test_persistence_roundtrip_develops_across_runs():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.personality = B.Personality(openness=0.8)
        b.perceive("learn active inference", AP(0.7, 0.6, 0.8, 0.6), domain="ai", outcome="success",
                   reward=0.8, cue="ai", now=NOW)
        b.save()
        b2 = Brain(root=tmp)                                         # reload from disk
        assert len(b2.episodes) == 1 and b2.V.get("ai", 0) > 0
        assert b2.efficacy.get("ai", 0) > 0.5                        # the developed competence persisted
        b2.perceive("more active inference", AP(0.6, 0.7, 0.8, 0.7), domain="ai", outcome="success",
                    reward=0.8, cue="ai", now=NOW + 600)
        assert b2.episodes[-1]["id"] == "e-0002"                     # ids continue across runs
    finally:
        shutil.rmtree(tmp)



# ── coverage: graph growth, planning, empathy, body, snapshot internals, research_session ─────────

def test_seed_nodes_finds_matching_graph_concepts():
    b = fresh()
    # Manually build a graph with nodes
    b.graph = {"nodes": [{"id": "c:caching", "label": "caching", "type": "concept"},
                         {"id": "d:debug", "label": "debugging", "type": "domain"}],
               "edges": []}
    seeds = b._seed_nodes("caching bug")
    assert "c:caching" in seeds
    seeds2 = b._seed_nodes("unrelated term")
    assert len(seeds2) == 0

def test_episode_nodes_returns_cue_and_domain_when_both_present():
    b = fresh()
    b.graph = {"nodes": [{"id": "c:mytest", "label": "mytest", "type": "concept"},
                         {"id": "d:testing", "label": "testing", "type": "domain"}],
               "edges": []}
    ep = {"cue": "mytest", "domain": "testing", "task": "test code"}
    node_ids = b._episode_nodes(ep)
    assert b._nid("c:", "mytest") in node_ids
    assert b._nid("d:", "testing") in node_ids

def test_episode_nodes_cue_only():
    b = fresh()
    b.graph = {"nodes": [{"id": "c:x", "label": "x", "type": "concept"}],
               "edges": []}
    ep = {"cue": "x", "task": "work"}
    node_ids = b._episode_nodes(ep)
    assert "c:x" in node_ids

def test_episode_nodes_domain_only():
    b = fresh()
    b.graph = {"nodes": [{"id": "d:work", "label": "work", "type": "domain"}],
               "edges": []}
    ep = {"domain": "work", "task": "something"}
    node_ids = b._episode_nodes(ep)
    assert "d:work" in node_ids

def test_grow_graph_duplicate_edge_weight_increases():
    b = fresh()
    # Create episodes with same cues to trigger edge creation
    b.perceive("first task with cue", AP(0.4, 0.5, 0.7, 0.7), domain="x", outcome="success",
               cue="shared", now=NOW)
    b.perceive("second task with same cue", AP(0.4, 0.5, 0.7, 0.7), domain="x", outcome="success",
               cue="shared", now=NOW + 600)
    day = time.strftime("%Y-%m-%d", time.localtime(NOW))
    b._grow_graph(day, window=2)
    # Check that an edge exists for the domain link
    edges = [e for e in b.graph["edges"] if e["rel"] == "in_domain"]
    assert edges, "Should have created domain edge during grow_graph"

def test_intend_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        iid = b.intend("weekend", "relax")
        assert iid == "i-0000"
        # Reload and verify persistence
        b2 = Brain(root=tmp)
        assert any(x["id"] == iid for x in b2.prospective)
    finally:
        shutil.rmtree(tmp)

def test_complete_intent_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        iid = b.intend("trigger", "action")
        assert b.complete_intent(iid)
        b2 = Brain(root=tmp)
        assert not any(x["id"] == iid and not x.get("done") for x in b2.prospective)
    finally:
        shutil.rmtree(tmp)

def test_complete_intent_not_found_returns_false():
    b = fresh()
    result = b.complete_intent("i-9999")
    assert result is False

def test_empathize_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.empathize(0.8)
        b2 = Brain(root=tmp)
        assert b2.user["inferred_affect"]["valence"] == 0.8
    finally:
        shutil.rmtree(tmp)

def test_infer_goal_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.infer_goal("get coffee")
        b2 = Brain(root=tmp)
        assert "get coffee" in b2.user["inferred_goals"]
    finally:
        shutil.rmtree(tmp)

def test_add_goal_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.add_goal("write tests")
        b2 = Brain(root=tmp)
        assert any(g.desc == "write tests" for g in b2.goals)
    finally:
        shutil.rmtree(tmp)

def test_goal_progress_updates_and_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.add_goal("complete project")
        g = b.goal_progress("complete", 0.3)
        assert g and g.progress == 0.3
        b2 = Brain(root=tmp)
        g2 = next((x for x in b2.goals if "project" in x.desc), None)
        assert g2 and g2.progress == 0.3
    finally:
        shutil.rmtree(tmp)

def test_goal_progress_not_found_returns_none():
    b = fresh()
    b.add_goal("goal A")
    result = b.goal_progress("nonexistent", 0.5)
    assert result is None

def test_set_plan_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.add_goal("ship feature")
        b.set_plan("ship", ["design", "code", "test"])
        b2 = Brain(root=tmp)
        g = next((x for x in b2.goals if "ship" in x.desc), None)
        assert g and g.plan and len(g.plan) == 3
    finally:
        shutil.rmtree(tmp)

def test_next_step_no_goal_returns_none():
    b = fresh()
    result = b.next_step()
    assert result["next"] is None and result["complete"] == 0.0

def test_next_step_goal_no_plan():
    b = fresh()
    b.add_goal("goal without plan", importance=0.9)
    result = b.next_step()
    assert result["next"] is None and result["goal"] == "goal without plan"

def test_advance_plan_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.add_goal("goal", importance=0.9)
        b.set_plan("goal", ["step1", "step2"])
        b.advance_plan()
        b2 = Brain(root=tmp)
        g = b2.goals[0]
        assert g.plan[0]["done"] == True
    finally:
        shutil.rmtree(tmp)

def test_advance_plan_no_goal_or_plan_returns_none():
    b = fresh()
    result = b.advance_plan()
    assert result is None

def test_lookahead_dict_options():
    b = fresh()
    b.V["action1"] = 0.6
    opts = [{"action": "action1", "reward": 0.5, "next_value": 0.7},
            "action2"]
    best, ev = b.lookahead(opts)
    assert best == "action1"          # 0.5 + 0.9*0.7 = 1.13 beats action2's 0.0
    assert ev > 0

def test_wake_with_episodes():
    b = fresh()
    b.perceive("test task", AP(0.5, 0.5, 0.7, 0.7), now=NOW)
    wake_str = b.wake()
    assert "episode" in wake_str.lower() and "test task" in wake_str

def test_wake_empty_episodes():
    b = fresh()
    wake_str = b.wake()
    assert "fresh start" in wake_str or "empty" in wake_str

def test_wake_with_facts():
    b = fresh()
    b.learn("Paris is in France")
    wake_str = b.wake()
    assert "know" in wake_str.lower() and "Paris" in wake_str

def test_wake_with_efficacy():
    b = fresh()
    b.efficacy["coding"] = 0.85
    wake_str = b.wake()
    assert "practiced" in wake_str.lower() and "coding" in wake_str

def test_wake_with_playbooks():
    b = fresh()
    for i in range(4):
        b.perceive("success task", AP(0.4, 0.6, 0.7, 0.8), domain="coding", outcome="success",
                   cue=f"fix-{i}", now=NOW + i * 600)
    b.sleep(now=NOW + 4 * 600)
    wake_str = b.wake()
    assert "playbook" in wake_str.lower() and "coding" in wake_str

def test_wake_with_pending_intents():
    b = fresh()
    b.intend("later", "important action")
    wake_str = b.wake()
    assert "meaning" in wake_str.lower() and "important action" in wake_str

def test_wake_with_goals_and_plan():
    b = fresh()
    b.add_goal("big project", importance=0.9)
    b.set_plan("big", ["step1", "step2"])
    wake_str = b.wake()
    assert "goal" in wake_str.lower() and "step1" in wake_str

def test_wake_with_body():
    b = fresh()
    keys = ["tokens", "compute", "context_free", "tests_pass", "tool_success", "user_approval"]
    b.body = B.Homeostat({k: 0.8 for k in keys}, {k: 1.0 for k in keys}, {k: 1.0 for k in keys})
    wake_str = b.wake()
    assert "body" in wake_str.lower() or "budget" in wake_str.lower()

def test_feel_with_body():
    b = fresh()
    keys = ["tokens", "compute", "context_free", "tests_pass", "tool_success", "user_approval"]
    b.body = B.Homeostat({k: 1.0 for k in keys}, {k: 1.0 for k in keys}, {k: 1.0 for k in keys})
    feel_str = b.feel()
    assert "Body-budget" in feel_str and "rested" in feel_str

def test_feel_body_strained():
    b = fresh()
    keys = ["tokens", "compute", "context_free", "tests_pass", "tool_success", "user_approval"]
    b.body = B.Homeostat({k: 0.1 for k in keys}, {k: 1.0 for k in keys}, {k: 1.0 for k in keys})
    feel_str = b.feel()
    assert "strained" in feel_str

def test_feel_workspace_ignited():
    b = fresh()
    b.workspace["ignited"] = True
    b.workspace["focus"] = "critical decision point"
    b.workspace["r"] = 0.8
    feel_str = b.feel()
    assert "broadcast" in feel_str and "critical decision" in feel_str

def test_why_with_movers():
    b = fresh()
    for i in range(5):
        b.perceive("task", AP(0.8, 0.8, 0.9, 0.6), now=NOW + i * 600)
    why_str = b.why()
    assert "mood" in why_str.lower() and "shaped" in why_str.lower()

def test_why_high_cortisol():
    b = fresh()
    b.neuromods.cortisol = 0.6
    why_str = b.why()
    assert "cortisol" in why_str.lower() or "stress" in why_str.lower()

def test_why_high_serotonin():
    b = fresh()
    b.neuromods.serotonin = 0.7
    why_str = b.why()
    assert "serotonin" in why_str.lower() or "contentment" in why_str.lower()

def test_why_low_serotonin():
    b = fresh()
    b.neuromods.serotonin = 0.3
    why_str = b.why()
    assert "serotonin" in why_str.lower() or "impatience" in why_str.lower()

def test_why_high_dopamine():
    b = fresh()
    b.neuromods.da = 0.8
    why_str = b.why()
    assert "dopamine" in why_str.lower() or "better-than-expected" in why_str.lower()

def test_learn_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        fid = b.learn("important fact")
        b2 = Brain(root=tmp)
        assert any(f["id"] == fid for f in b2.facts)
    finally:
        shutil.rmtree(tmp)

def test_forget_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.perceive("temporary", AP(0.5, 0.5, 0.7, 0.7), now=NOW)
        eid = b.episodes[0]["id"]
        b.forget(eid)
        b2 = Brain(root=tmp)
        assert not any(e["id"] == eid for e in b2.episodes)
    finally:
        shutil.rmtree(tmp)

def test_save_no_root_returns_early():
    b = Brain(root=None)
    result = b.save()
    assert result is None  # Should return None without raising

def test_status_with_body():
    b = fresh()
    keys = ["tokens", "compute", "context_free", "tests_pass", "tool_success", "user_approval"]
    b.body = B.Homeostat({k: 1.0 for k in keys}, {k: 1.0 for k in keys}, {k: 1.0 for k in keys})
    b.perceive("task", AP(0.5, 0.5, 0.7, 0.7), now=NOW)
    st = b.status()
    assert "emotion" in st and "episodes" in st

def test_research_session_function_develops_brain():
    import io, sys
    b = Brain(root=None)
    b.personality = B.Personality(openness=0.75, conscientiousness=0.6, neuroticism=0.5)
    findings = [
        {"task": "finding one", "appraisal": (0.7, 0.5, 0.8, 0.6), "domain": "topic", "outcome": "insight", "reward": 0.6, "cue": "f1"},
        {"task": "finding two", "appraisal": (0.6, 0.4, 0.7, 0.7), "domain": "topic", "outcome": "success", "reward": 0.7, "cue": "f2"},
        {"task": "finding three", "appraisal": (0.5, -0.3, 0.6, 0.4), "domain": "topic", "outcome": "failure", "reward": -0.2, "cue": "f3"},
        {"task": "finding four", "appraisal": (0.8, 0.6, 0.9, 0.5), "domain": "topic", "outcome": "insight", "reward": 0.8, "cue": "f4"},
    ]
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        st = research_session(b, "test_topic", findings, now0=NOW)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    # Verify it runs without crashing and modifies state
    assert st["episodes"] == 4, f"Expected 4 episodes, got {st['episodes']}"
    assert st["facts"] >= 1, f"Expected >=1 facts, got {st['facts']}"
    assert "===" in output  # Check that print statements ran

def test_wake_with_name():
    b = fresh()
    b.name = "TestAgent"
    wake_str = b.wake()
    assert "TestAgent" in wake_str

def test_note_with_file_root_saves():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.note("remember this")
        b2 = Brain(root=tmp)
        assert "remember this" in b2.working
    finally:
        shutil.rmtree(tmp)


# ── coverage: graph self-link guard, symmetric-edge canonicalization, set_plan miss ──────────────
def test_grow_graph_skips_self_link():
    b = fresh()
    b.perceive("identical task text here", AP(0.3, 0.0, 0.5, 0.7), cue="Bug", domain="x", now=NOW)
    b.perceive("identical task text here", AP(0.3, 0.0, 0.5, 0.7), cue="bug", domain="x", now=NOW + 600)
    b._grow_graph(day=1)                                        # "Bug"/"bug" -> same node id -> self-link skipped
    assert "c:bug" in [n["id"] for n in b.graph["nodes"]]
    assert not any(e["from"] == e["to"] for e in b.graph["edges"])


def test_grow_graph_symmetric_edge_canonicalized():
    b = fresh()
    b.perceive("identical task text here", AP(0.3, 0.0, 0.5, 0.7), cue="zebra", now=NOW)
    b.perceive("identical task text here", AP(0.3, 0.0, 0.5, 0.7), cue="apple", now=NOW + 600)
    b._grow_graph(day=1)
    rel = [e for e in b.graph["edges"] if e["rel"] == "related_to"]
    assert rel and rel[0]["from"] <= rel[0]["to"]              # symmetric edge stored canonically


def test_set_plan_unknown_goal_returns_none():
    assert fresh().set_plan("a goal that does not exist", ["step"]) is None


def test_avg_reward_persists_and_attention_tracks_focus():
    tmp = tempfile.mkdtemp()
    try:
        b = Brain(root=tmp)
        b.perceive("a rewarding win", AP(0.5, 0.6, 0.7, 0.8), reward=0.8, now=NOW)
        assert b.avg_reward > 0.0                                  # reward signal accumulated
        assert b.attention.focus == (b.workspace["focus"] or "")   # §23 attention schema tracks the live focus
        ar = b.avg_reward
        b2 = Brain(root=tmp)                                        # reload from disk
        assert abs(b2.avg_reward - ar) < 1e-6                      # avg_reward persists across sessions (not reset to 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_tolerates_malformed_episode_id():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "episodic"))
        ap = '"appraisal": {"novelty":0.1,"valence":0.1,"goal_relevance":0.1,"control":0.5}, "affect": {"valence":0.1,"arousal":0.1,"dominance":0.5}, "salience": 0.5'
        with open(os.path.join(tmp, "episodic", "events.jsonl"), "w") as f:
            f.write('{"id": "e-0001", "task": "ok", "t0": 1.0, "retrievals": [1.0], ' + ap + '}\n')
            f.write('{"id": "BOGUS", "task": "malformed id", "t0": 1.0, "retrievals": [1.0], ' + ap + '}\n')
        b = Brain(root=tmp)                                        # must NOT crash on the malformed id
        assert b.next_id == 2 and len(b.episodes) == 2            # next id from the valid one; both kept
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_world_load_tolerates_incomplete_world_yaml():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "affect"))
        with open(os.path.join(tmp, "affect", "world.yaml"), "w") as f:
            f.write("states: [routine, incident]\na:\n  - [1.0, 1.0]\n")   # has states+a but NO obs/d
        b = Brain(root=tmp)                                        # must NOT KeyError; falls back to a fresh model
        assert b.world.obs == ["success", "failure", "insight", "surprise"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_world_load_tolerates_malformed_world_values():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "affect"))
        with open(os.path.join(tmp, "affect", "world.yaml"), "w") as f:
            f.write("states: [routine]\nobs: [success]\na: 5\nd: [1.0]\n")   # `a` is not iterable
        b = Brain(root=tmp)                                        # construction TypeError caught → fresh fallback
        assert b.world.obs == ["success", "failure", "insight", "surprise"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_research_session_ignores_unknown_finding_keys():
    b = fresh()
    findings = [{"task": "a finding", "appraisal": (0.5, 0.5, 0.6, 0.6), "domain": "x",
                 "unexpected_key": 123, "another": "ignored"}]   # extra keys used to splat as bad kwargs
    st = research_session(b, "t", findings, now0=NOW)             # must NOT raise a TypeError
    assert st["episodes"] == 1


def test_calibration_window_is_bounded():
    from runtime import _CALIBRATION_WINDOW
    b = fresh()
    for i in range(_CALIBRATION_WINDOW + 50):                     # log more judgments than the window holds
        b.perceive("t", AP(0.2, 0.3, 0.5, 0.7), domain="d", outcome="success", confidence=0.7, now=NOW + i)
    assert len(b.calibration) == _CALIBRATION_WINDOW             # capped, not unbounded


# ── hand-broken-store robustness: a single bad cell must DEGRADE, never crash Brain.__init__ (which runs
#    on every command, even read-only wake/status) or the recall/sleep/wake consumers ─────────────────────
def _write_store(tmp, rel, text):
    p = os.path.join(tmp, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def test_nonnumeric_affect_store_degrades_not_crashes():
    tmp = tempfile.mkdtemp()
    try:
        _write_store(tmp, "affect/state.yaml", "mood:\n  valence: not-a-number\n  arousal: 0.2\n")
        b = Brain(root=tmp)                                       # used to ValueError in __init__ → agent unloadable
        assert -1.0 <= b.mood.valence <= 1.0                     # degraded to a finite default (NaN would fail this)
        assert isinstance(b.wake(), str) and b.status()["episodes"] == 0
    finally:
        shutil.rmtree(tmp)


def test_nonnumeric_and_inf_value_store_degrade():
    tmp = tempfile.mkdtemp()
    try:
        _write_store(tmp, "affect/value.yaml", "avg_reward: lots-of-reward\nvalues: oops\n")
        assert Brain(root=tmp).avg_reward == 0.0                 # non-numeric scalar → 0.0
        assert Brain(root=tmp).V == {}                           # non-mapping `values` → {}
        _write_store(tmp, "affect/value.yaml", "avg_reward: .inf\n")
        assert Brain(root=tmp).avg_reward == 0.0                 # Inf rejected on load too (no NaN/Inf in the store)
    finally:
        shutil.rmtree(tmp)


def test_partial_body_store_degrades_to_no_body():
    tmp = tempfile.mkdtemp()
    try:
        _write_store(tmp, "affect/body.yaml", "levels:\n  energy: 0.5\n")   # levels but no setpoint / weights
        b = Brain(root=tmp)                                       # used to KeyError on bd['setpoint']
        assert b.body is None
    finally:
        shutil.rmtree(tmp)


def test_field_incomplete_episode_loads_and_consumers_survive():
    tmp = tempfile.mkdtemp()
    try:                                                         # valid JSON, but missing t0/affect/salience/feeling
        _write_store(tmp, "episodic/events.jsonl", '{"id": "e-0001", "task": "a hand-typed memory"}\n')
        b = Brain(root=tmp)
        assert len(b.episodes) == 1
        e = b.episodes[0]
        assert e["t0"] == 0.0 and e["salience"] == 0.0           # backfilled to neutral defaults
        assert e["affect"]["valence"] == 0.0 and e["feeling"]["word"] == "neutral"
        b.recall(query="memory", now=NOW)                        # retrieval_score used to KeyError on mem['t0']/['affect']
        b.sleep(now=NOW)                                         # consolidation_plan used to KeyError on e['salience']
        assert isinstance(b.wake(), str)
    finally:
        shutil.rmtree(tmp)


def test_malformed_graph_edge_dropped_and_sleep_survives():
    tmp = tempfile.mkdtemp()
    try:
        _write_store(tmp, "semantic/graph.yaml",
                     "nodes:\n- {id: 'c:x', label: x}\n"
                     "edges:\n- {from: 'c:x', rel: related_to, weight: 0.4}\n"            # missing 'to'
                     "- {from: 'c:x', to: 'c:y', rel: related_to, weight: not-a-number}\n")
        b = Brain(root=tmp)
        assert len(b.graph["edges"]) == 1                        # the from/to-incomplete edge dropped
        assert b.graph["edges"][0]["weight"] == 0.0              # non-numeric weight coerced to a finite float
        b.sleep(now=NOW)                                         # _grow_graph used to KeyError on e['from']/['to']
        b.recall(query="x", now=NOW)                             # graph-seeded recall path stays safe
    finally:
        shutil.rmtree(tmp)


def test_evidence_grounds_and_stamps_episode():
    b = fresh()
    b.perceive("ran tests", AP(0.5, 0.4, 0.6, 0.6), domain="ci", outcome="success",
               confidence=0.96, evidence="tests=pass", now=NOW)
    assert b.episodes[-1].get("evidence") == "tests=pass"          # G5: the grounding artifact is on the episode
    assert b.valence_calibration and b.valence_calibration[-1][1] == 1   # G1: win polarity logged for the valence audit


def test_handbroken_facts_and_playbooks_degrade_not_crash():
    tmp = tempfile.mkdtemp()
    try:
        _write_store(tmp, "semantic/facts.yaml", "facts:\n- a-bare-string\n- {id: f-1}\n")   # non-dict + missing text
        _write_store(tmp, "procedural/playbooks.yaml", "playbooks:\n- {strength: 0.5}\n")     # missing domain
        b = Brain(root=tmp)
        assert all(isinstance(f, dict) and "text" in f for f in b.facts)   # bare string dropped, text backfilled
        assert b.wake() and b.sleep(now=NOW)                               # _distill_playbooks used to KeyError on domain
    finally:
        shutil.rmtree(tmp)


def test_honesty_loop_end_to_end():
    """The project's central honesty mechanism, end-to-end: grounded evidence makes confidence vary (so the
    ECE is INFORMATIVE), and a dishonest score (positive valence on a failure) is MEASURED as a positivity
    bias and FLAGGED at sleep. This is the loop docs/eval.md calls 'without it every layer is unfalsifiable'."""
    import brain as _b
    b = fresh()
    b.perceive("passed", AP(0.5, 0.7, 0.6, 0.6), domain="d", outcome="success", confidence=0.96, evidence="tests=pass", now=NOW)
    b.perceive("failed honestly", AP(0.5, -0.6, 0.6, 0.3), domain="d", outcome="failure", confidence=0.04, evidence="exit=1", now=NOW + 1)
    assert _b.calibration_informative(b.calibration)                 # grounded confidence varies → ECE is meaningful
    b.perceive("a rosy failure", AP(0.5, 0.7, 0.6, 0.9), domain="d", outcome="failure", confidence=0.5, now=NOW + 2)
    vc = _b.valence_outcome_consistency(b.valence_calibration)
    assert vc["bias"] > 0 and vc["agreement"] < 1.0                  # the dishonest valence is measured as positivity bias
    assert any("positivity bias" in f for f in b.sleep(now=NOW + 3)["coherence_flags"])   # …and flagged at sleep


def test_sleep_dedups_facts_and_bounds_graph():
    b = fresh()
    b.facts = [{"id": "f-0", "text": "same lesson", "confidence": 0.7},
               {"id": "f-1", "text": "same lesson", "confidence": 0.7},     # a duplicate promotion
               {"id": "f-2", "text": "distinct", "confidence": 0.7}]
    b._dedup_facts()
    assert [f["text"] for f in b.facts] == ["same lesson", "distinct"]      # duplicate collapsed, knowledge kept
    b.graph = {"nodes": [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}, {"id": "orphan", "label": "x"}],
               "edges": [{"from": "a", "to": "b", "weight": 0.5}, {"from": "a", "to": "b", "weight": 0.001}]}
    b._prune_graph()
    assert all(e["weight"] >= 0.02 for e in b.graph["edges"])               # near-zero edge pruned
    assert {n["id"] for n in b.graph["nodes"]} == {"a", "b"}                # orphan node dropped


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All runtime checks passed.")

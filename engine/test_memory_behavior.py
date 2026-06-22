"""
test_memory_behavior.py — BEHAVIORAL memory tests (the answer to "how do I test memory properly?").

Unlike test_brain.py (per-function math) and test_runtime.py (wiring), these test the *emergent memory
behaviour* an owner actually cares about, the way you'd probe a person: does the right memory come back?
does it survive a new session? does emotion fade but the fact stay? does the weak stuff get forgotten?
does it refuse to confabulate? do related concepts connect? does the familiar stop being surprising?

Each test is FALSIFIABLE — it states the property, sets up a scenario, and asserts a concrete outcome
that a broken memory would fail. Run: python3 test_memory_behavior.py
The same scenarios work by hand against the CLI (see each test's docstring for the `./brain ...` recipe).
"""
import os
import shutil
import tempfile

import brain as B
from runtime import Brain

NOW = 1_750_000_000.0
DAY = 86400.0
AP = lambda nov, val, gr, ctl: B.Appraisal(nov, val, gr, ctl)
rel_to = lambda q: (lambda e: B.jaccard(B.tokens(e["task"]), B.tokens(q)))   # content-overlap relevance


def fresh():
    b = Brain(root=None)
    b.personality = B.Personality(openness=0.7, conscientiousness=0.6, neuroticism=0.4)
    return b


def test_1_recall_precision_right_memory_first():
    """The topically right memory ranks #1 with a clear margin — recall is relevance-driven, not recency-only.
    CLI: react to 3 distinct topics, then `./brain recall "database migration"` → the db memory is #1."""
    b = fresh()
    texts = ["the database migration broke staging", "a quiet walk in the park at dawn",
             "filing the quarterly tax paperwork"]
    for i, t in enumerate(texts):
        b.perceive(t, AP(0.4, 0.0, 0.5, 0.7), cue=t.split()[1], now=NOW + i * 600)
    hits = b.recall(rel_to("database migration"), k=3, query="database migration", now=NOW + 5000)
    assert hits[0]["task"] == texts[0], hits
    assert hits[0]["score"] - hits[1]["score"] >= 0.1, hits        # clear separation, not a coin-flip
    print(f"#1='{hits[0]['task'][:28]}…' margin={hits[0]['score'] - hits[1]['score']:.2f}")


def test_2_cross_session_identity_and_fact_survive_reload():
    """Identity and a distinctive fact survive a full process teardown — 'continuing a life, not booting blank'.
    CLI: react a fact, then in a SEPARATE run `./brain wake` (names you) + `./brain recall` (returns the fact)."""
    tmp = tempfile.mkdtemp()
    try:
        a = Brain(root=tmp)
        a.name = "Mira"
        a.react("the launch passphrase is HERON-7", 0.4, 0.7, 0.7, outcome="insight", reward=0.4,
                cue="passphrase", now=NOW)
        a.save()
        b = Brain(root=tmp)                                        # a fresh OS process / new session
        assert b.name == "Mira"                                    # the self persists
        hits = b.recall(rel_to("passphrase"), k=1, query="passphrase", now=NOW + 600)
        assert hits and "HERON-7" in hits[0]["task"], hits         # the memory survived the reload
        print(f"reloaded name='{b.name}', recovered '{hits[0]['task'][:30]}…'")
    finally:
        shutil.rmtree(tmp)


def test_3_consolidation_fades_emotion_keeps_event():
    """Sleep downscales emotional salience while the episode + its original affect stay intact, and a
    salient cluster triggers a reflection. CLI: react 5 intense events, `./brain sleep`, `./brain episodes`."""
    b = fresh()
    for i in range(5):
        good = i % 2 == 0
        b.perceive("an intense, salient event", AP(0.8, 0.7 if good else -0.7, 0.9, 0.5),
                   outcome="insight" if good else "failure", reward=0.6 if good else -0.5,
                   cue="big", now=NOW + i * 60)
    sal_before = max(e["salience"] for e in b.episodes)
    aff_before = b.episodes[0]["affect"]["valence"]
    s = b.sleep(now=NOW + 5 * 60)
    sal_after = max(e["salience"] for e in b.episodes)
    assert sal_after < sal_before                                  # the charge faded
    assert b.episodes[0]["affect"]["valence"] == aff_before        # the episode's record is untouched
    assert s["reflections"] >= 1                                   # a higher-level fact emerged
    print(f"salience {sal_before:.2f}->{sal_after:.2f}, reflections={s['reflections']}, episodes kept={s['episodes']}")


def test_4_forgetting_prunes_weak_old_keeps_strong():
    """Weak, old, never-recalled memories are forgotten while a salient recent one stays — graceful
    forgetting, not perfect recall. CLI: encode a trivial + a salient event, `sleep` after a long gap."""
    b = fresh()
    b.perceive("a trivial passing note about nothing", AP(0.05, 0.0, 0.05, 0.95), now=NOW)        # weak, ages 60d
    b.perceive("a hard-won critical lesson", AP(0.8, -0.6, 0.9, 0.5), outcome="failure", reward=-0.6,
               cue="lesson", now=NOW + 60 * DAY - 600)                                            # strong, recent
    before = len(b.episodes)
    b.sleep(now=NOW + 60 * DAY)
    kept = " | ".join(e["task"] for e in b.episodes)
    assert "critical lesson" in kept                               # the salient memory survives
    assert "trivial passing note" not in kept                      # the weak-old one is gone
    assert len(b.episodes) < before
    print(f"forgot the trivial note, kept the lesson ({before}->{len(b.episodes)} episodes)")


def test_5_no_confabulation_substrate():
    """The system can tell what it knows from what it doesn't — a present topic has real relevance, an
    absent one scores exactly 0 (so an honest agent abstains instead of inventing).
    CLI: ask via chat about something never said → it should answer 'I don't have that', not fabricate."""
    b = fresh()
    b.perceive("we planned the database migration for Tuesday", AP(0.5, 0.2, 0.7, 0.6), cue="db", now=NOW)
    present = max(rel_to("database migration")(e) for e in b.episodes)
    absent = max(rel_to("the vacation budget in Tokyo")(e) for e in b.episodes)
    assert present > 0.0                                           # it has a real basis to answer
    assert absent == 0.0                                           # and a real basis to abstain (nothing matches)
    print(f"present-topic relevance={present:.2f}, absent-topic relevance={absent:.2f}")


def test_6_graph_associates_related_concepts():
    """Sleep weaves related concepts into the association graph so one cue can reach its neighbours;
    unrelated concepts stay disconnected. CLI: react several same-theme events, `sleep`, `cat graph.yaml`."""
    b = fresh()
    b.react("the caching layer served stale cache data", 0.0, 0.6, 0.6, domain="debugging",
            outcome="insight", reward=0.3, cue="caching", now=NOW)
    b.react("cache invalidation was the real cache bug", -0.2, 0.7, 0.5, domain="debugging",
            outcome="insight", reward=0.3, cue="invalidation", now=NOW + 600)
    b.react("a calm walk in the park", 0.6, 0.2, 0.9, domain="life",
            outcome="success", reward=0.5, cue="walk", now=NOW + 1200)
    assert not b.graph["edges"]                                    # empty before sleep
    b.sleep(now=NOW + 1300)
    assert b.graph["nodes"] and b.graph["edges"]                   # the graph grew from living
    cache, inval, walk = b._nid("c:", "caching"), b._nid("c:", "invalidation"), b._nid("c:", "walk")
    near = B.graph_proximity(b.graph["edges"], [cache], [inval])
    assert near > 0.0                                              # related concepts connect
    assert B.graph_proximity(b.graph["edges"], [cache], [walk]) < near   # the unrelated one does not
    print(f"{len(b.graph['nodes'])} nodes, {len(b.graph['edges'])} edges; caching↔invalidation={near:.3f}, ↔walk≈0")


def test_7_habituation_novelty_decays_each_still_encoded():
    """Novelty is grounded in surprise: the same kind of event becomes less novel on repetition, yet every
    occurrence is still recorded. CLI: `./brain react` the same --outcome 5× and watch the printed novelty fall."""
    b = fresh()
    nov = []
    for i in range(5):
        r = b.react("the same kind of success again", 0.4, 0.5, 0.6, outcome="success", reward=0.4,
                    cue="s", now=NOW + i * 600)
        nov.append(r["novelty_computed"])
    assert len(b.episodes) == 5                                    # every react durably encoded
    assert nov[-1] < nov[0]                                        # it habituated
    assert all(nov[i] >= nov[i + 1] for i in range(len(nov) - 1))  # monotonic non-increasing
    print(f"novelty {nov[0]:.2f}->{nov[-1]:.2f} over 5 reacts, all 5 encoded")


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            evidence = fn()
            print(f"PASS {name:52s} {evidence or ''}")
            passed += 1
    print(f"\nAll {passed} behavioral memory checks passed.")

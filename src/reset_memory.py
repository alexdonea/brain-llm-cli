"""reset_memory.py - regenerate the FRESH seed of .memory/ (the canonical starting brain the project
ships as a template). Run after a demo or research session to return the agent
to a blank slate. Run: python3 src/reset_memory.py [path-to-.memory]

This is the single source of truth for the fresh seed; the rich documentation headers live here.
"""
import datetime
import os
import sys

# Each entry: relative path -> the full fresh file contents (header comment + body).
SEED = {
"affect/state.yaml": """\
# Affect state - amygdala + neuromodulator nuclei (LC, VTA, basal forebrain). Computed signals that
# modulate memory, NOT felt emotion. Fresh seed: emotion == mood == baseline (calm, awake), no events yet.
# emotion = FAST scale (~min); mood = SLOW scale (~hrs, read by recall); baseline = personality set-point.
# neuromods/hpa: see schema.md §18. Maintained by src/runtime.py (Brain) / the /sleep cycle.
emotion:   {valence: 0.00, arousal: 0.10, dominance: 0.50}
mood:      {valence: 0.00, arousal: 0.10, dominance: 0.50}
baseline:  {valence: 0.00, arousal: 0.10, dominance: 0.50}
neuromods: {ne: 0.10, da: 0.00, ach: 1.0, cortisol: 0.00, serotonin: 0.50, oxytocin: 0.00, ne_tonic: 0.10}
hpa:       {crh: 0.00, acth: 0.00, cortisol: 0.10}
updated:   2026-06-19T00:00:00Z
""",
"affect/value.yaml": """\
# Value (striatum/dopamine) + aversive channels. `values`: learned expected reward per cue (TD, §10).
# `aversive`: learned expected harm per cue (§25), learned faster (eta_aversive). Fresh brain: nothing learned.
gamma:  0.9
alpha:  0.3
values: {}
eta_aversive: 0.4
aversive: {}
""",
"affect/world.yaml": """\
# Generative world-model - active inference / Bayesian surprise (§11). Latent states x observable event
# categories with Dirichlet counts learned online. `obs` must cover the outcomes runtime emits
# (success/failure/insight/surprise). Fresh brain: flat counts (all 1.0).
states: [routine, novel_problem, incident]
obs:    [success, failure, insight, surprise]
a:
  - [1.0, 1.0, 1.0]   # success
  - [1.0, 1.0, 1.0]   # failure
  - [1.0, 1.0, 1.0]   # insight
  - [1.0, 1.0, 1.0]   # surprise
d: [1.0, 1.0, 1.0]    # prior over states
""",
"affect/body.yaml": """\
# Interoception - the agent's body-budget: its OWN real viability variables (§15). Cybernetic, NOT felt.
# drive(H) = deficit; homeostatic_reward = drive reduction (a grounded reward). Fresh brain: all healthy.
# Driven by living: react()/perceive() deplete on effort & move signals on outcome; sleep() rests it.
levels:   {tokens: 1.0, compute: 1.0, tests_pass: 1.0, tool_success: 1.0, context_free: 1.0, user_approval: 1.0}
setpoint: {tokens: 1.0, compute: 1.0, tests_pass: 1.0, tool_success: 1.0, context_free: 1.0, user_approval: 1.0}
weights:  {tokens: 1.0, compute: 0.5, tests_pass: 1.5, tool_success: 1.0, context_free: 1.0, user_approval: 1.5}
updated:  2026-06-19T00:00:00Z
""",
"self/efficacy.yaml": """\
# Metacognition - self-efficacy & calibration (§13). `efficacy`: domain -> competence [0,1]
# (update_self_efficacy). `calibration`: logged [confidence, correct] pairs -> ECE. Fresh: no track record.
efficacy:    {}
calibration: []
default_efficacy:          0.5
min_confidence_to_promote: 0.5
updated: 2026-06-19T00:00:00Z
""",
"self/personality.yaml": """\
# Personality / temperament - Big Five (OCEAN) as affective priors (§14). EDIT per agent. 0.5 = average.
# baseline_from_personality -> the mood set-point; temperament_gains -> (BAS, BIS). Functional, not felt.
openness:          0.5
conscientiousness: 0.5
extraversion:      0.5
agreeableness:     0.5
neuroticism:       0.5
""",
"self/model.yaml": """\
# Self-model - a functional, representational identity (§23), NOT a felt self. self_relevance (cosine to
# the self-vector), sense_of_agency (-> control axis), attention_schema (predicts its own focus -> AST-1).
competencies: {python: 0.5, debugging: 0.5}
goals: []
traits: {}
attention_schema: {focus: "", predicted_next: "", uncertainty: 1.0}
updated: 2026-06-19T00:00:00Z
""",
"social/user.yaml": """\
# User model / relationship - the OTHER, for social emotion & Theory of Mind (§24). ToM is INFERRED, not
# known. infer_user_goal / empathic_mood_shift / social_emotion / update_trust. Fresh: neutral trust.
trust: 0.5
inferred_goals: {}
inferred_affect: {valence: 0.0}
updated: 2026-06-19T00:00:00Z
""",
"semantic/facts.yaml": """\
# Semantic facts - neocortex declarative store (§8/§20). Promoted from episodes at /sleep with the
# emotional charge depotentiated (REM). Shape: {id, text, source (episode id or 'reflection'), confidence,
# affect_charge?, valid_from}. Fresh brain: nothing consolidated yet.
facts: []
""",
"semantic/graph.yaml": """\
# Association graph - neocortex (shapes per schema.md). node {id,type,label}; edge {from,to,rel,weight,
# valid_from}. Auto-grown during sleep() (concept↔domain + content-similar concepts link, Hebbian-weighted);
# recall uses graph_proximity (spreading activation) to surface related memories. Also editable by hand.
nodes: []
edges: []
""",
"prospective/todo.yaml": """\
# Prospective memory - prefrontal future intentions (trigger -> intent). Item: {id, trigger, intent,
# created, done}. Capture with `./brain intend`; pending ones resurface at wake(). Fresh: no intentions.
intents: []
""",
"procedural/playbooks.yaml": """\
# Procedural memory - distilled how-to playbooks per domain (basal ganglia / cerebellum). Item:
# {id, domain, steps, attempts, successes, strength}. DISTILLED during sleep() from clusters of same-domain
# success episodes; strength = power law of practice (brain.py practice_strength). Fresh brain: none yet.
playbooks: []
""",
"working/workspace.yaml": """\
# Global workspace - the limited-capacity "stage" (§12). Driven each perceive(): the event competes and,
# if it crosses the ignition threshold, becomes the broadcast `focus`. Functional access, NOT awareness.
focus:   null
ignited: false
r:       0.0
p:       {}
updated: 2026-06-19T00:00:00Z
""",
"working/scratchpad.md": """\
# Working memory - prefrontal + sensory buffer. VOLATILE, disposable, ~7 items (Miller); wiped at /sleep.
# Nothing here is durable until encoded to episodic memory.

**Current focus:** (none)

## Active items (max ~7)
- (empty)
""",
"episodic/events.jsonl": "",
}


def reset(root, quiet=False):
    today = datetime.date.today().isoformat()                # stamp fresh stores with the real current date
    for rel, content in SEED.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content.replace("2026-06-19", today))
    if not quiet:                                            # internal bootstrap/create calls stay silent
        print(f"reset {len(SEED)} stores to the fresh seed under {root}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "agents", "default", "memory")
    reset(root)

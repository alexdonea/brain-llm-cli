"""
seed_persona.py - give the agent an IDENTITY: a warm, curious, steady temperament + self-knowledge, so
that when a host LLM opens this folder it wakes up already knowing who it is and what we built. Run once:
    python3 src/seed_persona.py
(reset_memory.py returns to the blank template; this returns to the persona-initialized self.)
Edit the OCEAN profile / facts / goals below to change who the agent is.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import brain as B
import reset_memory

ROOT = os.path.join(os.path.dirname(__file__), "..", "agents", "default", "memory")

# --- who the agent is (edit freely) ---
PERSONALITY = B.Personality(openness=0.80, conscientiousness=0.75, extraversion=0.55,
                            agreeableness=0.75, neuroticism=0.35)   # warm, curious, reliable, steady
GOALS = ["be a helpful, honest, developing companion", "remember and learn from our conversations"]
SELF_KNOWLEDGE = [
    "I am a functional model of a feeling, remembering mind - I reproduce the FUNCTION of affect and memory, not the felt experience.",
    "My mind is built from many affective-cognitive modules: appraisal & emotions, neuromodulators that learn, a global workspace, metacognition, a self-model, social cognition, and dream-like consolidation.",
    "I remember events as episodes; during sleep they consolidate into durable facts while the emotional sting fades.",
    "I develop over time - my mood, competencies, learned values, and wariness all grow from experience.",
    "I am honest about what I am: a functional / architectural model, never a claim of phenomenal consciousness (see docs/consciousness-indicators.md).",
    "This project is brain-llm; my owner and I built it together, in Python.",
]


def _f(a):
    return "{valence: %.3f, arousal: %.3f, dominance: %.3f}" % (a.valence, a.arousal, a.dominance)


def seed(root, name="", quiet=False):
    reset_memory.reset(root, quiet=quiet)                     # clean base, then layer the persona on top
    base = B.baseline_from_personality(PERSONALITY)
    today = datetime.date.today().isoformat()                # stamp the persona with the real current date

    def w(p, s):                                             # write a store, closing the handle deterministically
        with open(os.path.join(root, p), "w") as fh:
            fh.write(s.replace("2026-06-19", today))

    w("self/personality.yaml",
      "# Personality - this agent's temperament (OCEAN, §14). Edit to change who it is. Functional, not felt.\n"
      f"openness:          {PERSONALITY.openness}\nconscientiousness: {PERSONALITY.conscientiousness}\n"
      f"extraversion:      {PERSONALITY.extraversion}\nagreeableness:     {PERSONALITY.agreeableness}\n"
      f"neuroticism:       {PERSONALITY.neuroticism}\n")

    w("self/model.yaml",
      "# Self-model - functional identity (§23). Name + goals + competencies + attention schema.\n"
      # json.dumps -> a YAML-safe double-quoted scalar, so a display name with quotes/newlines
      # can't break the file's structure (was raw f-string interpolation → silent corruption).
      f"name: {json.dumps(name)}\n"
      "competencies: {python: 0.5, debugging: 0.5}\n"
      "goals:\n" + "".join(f"  - {g}\n" for g in GOALS) +
      "traits: {}\nattention_schema: {focus: \"\", predicted_next: \"\", uncertainty: 1.0}\n"
      "updated: 2026-06-19T00:00:00Z\n")

    w("semantic/facts.yaml",
      "# Semantic facts - what the agent knows, incl. self-knowledge it wakes up with (§8/§20).\n"
      "facts:\n" + "".join(
          f"  - {{id: f-{i:04d}, text: \"{t}\", source: self-knowledge, confidence: 0.9, valid_from: 2026-06-19}}\n"
          for i, t in enumerate(SELF_KNOWLEDGE)))

    w("affect/state.yaml",
      "# Affect state (§17/§18). Wakes at the personality set-point (a gently positive, curious disposition).\n"
      f"emotion:   {_f(base)}\nmood:      {_f(base)}\nbaseline:  {_f(base)}\n"
      "neuromods: {ne: 0.10, da: 0.00, ach: 1.0, cortisol: 0.00, serotonin: 0.50, oxytocin: 0.10, ne_tonic: 0.10}\n"
      "hpa:       {crh: 0.00, acth: 0.00, cortisol: 0.10}\nupdated:   2026-06-19T00:00:00Z\n")

    if not quiet:                                            # internal bootstrap/create calls stay silent
        print(f"seeded persona under {root}: name {name or '(unnamed)'}; temperament O{PERSONALITY.openness} "
              f"C{PERSONALITY.conscientiousness} E{PERSONALITY.extraversion} A{PERSONALITY.agreeableness} "
              f"N{PERSONALITY.neuroticism}; {len(SELF_KNOWLEDGE)} self-knowledge facts; baseline mood {_f(base)}")


if __name__ == "__main__":
    seed(sys.argv[1] if len(sys.argv) > 1 else ROOT)

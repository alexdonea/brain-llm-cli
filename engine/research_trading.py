"""The agent researches the stock market & trading — a real session driving runtime.Brain against the
live .memory/ store, so it develops PERSISTENT memory + competence + mood. Run from anywhere:
    python3 engine/research_trading.py
Appraisals (novelty, valence, goal_relevance, control) are the agent's own honest reactions to each
finding. Functional states, not felt experience."""
import os
import brain as B
from runtime import Brain, research_session

ROOT = os.path.join(os.path.dirname(__file__), "..", "agents", "default", "memory")

me = Brain(root=ROOT)
# a curious, conscientious, risk-aware researcher
me.personality = B.Personality(openness=0.85, conscientiousness=0.8, extraversion=0.45,
                               agreeableness=0.6, neuroticism=0.6)
me.self_model.goals = ["learn_to_trade_wisely"]

# Each finding: appraisal=(novelty, valence, goal_relevance, control) + domain/outcome/reward/cue.
FINDINGS = [
    {"task": "three core styles: trend-following, mean-reversion, momentum", "appraisal": (0.85, 0.6, 0.9, 0.6),
     "domain": "trading", "outcome": "insight", "reward": 0.7, "cue": "strategies"},
    {"task": "no single strategy works in all market regimes", "appraisal": (0.6, -0.2, 0.8, 0.3),
     "domain": "trading", "outcome": "surprise", "reward": -0.1, "cue": "regime"},
    {"task": "the 1-2% rule: size by what you can afford to lose", "appraisal": (0.5, 0.7, 0.9, 0.85),
     "domain": "risk", "outcome": "success", "reward": 0.9, "cue": "risk_sizing"},
    {"task": "stops at 1.5-2x ATR below support; aim 2:1 reward:risk", "appraisal": (0.4, 0.6, 0.8, 0.8),
     "domain": "risk", "outcome": "success", "reward": 0.8, "cue": "risk_sizing"},
    {"task": "technical answers WHEN to trade; fundamental answers WHICH", "appraisal": (0.6, 0.5, 0.8, 0.7),
     "domain": "analysis", "outcome": "insight", "reward": 0.6, "cue": "analysis"},
    {"task": "up to 80% of trading losses are emotional, not analytical", "appraisal": (0.9, -0.3, 0.95, 0.3),
     "domain": "psychology", "outcome": "surprise", "reward": -0.1, "cue": "psychology"},
    {"task": "revenge trading & loss aversion: the urge to 'win it back'", "appraisal": (0.7, -0.6, 0.9, 0.2),
     "domain": "psychology", "outcome": "failure", "reward": -0.5, "cue": "revenge"},
    {"task": "the antidote: a written plan + position limits + discipline", "appraisal": (0.4, 0.7, 0.85, 0.8),
     "domain": "psychology", "outcome": "success", "reward": 0.7, "cue": "discipline"},
    {"task": "synthesis: knowledge is necessary, emotional discipline decides", "appraisal": (0.6, 0.85, 0.95, 0.75),
     "domain": "trading", "outcome": "insight", "reward": 0.85, "cue": "synthesis"},
]

research_session(me, "stock market & trading techniques", FINDINGS)

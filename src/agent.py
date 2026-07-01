"""
agent.py - the agent's complete CLI for building and running its own mind. NO external API: the host LLM
(Claude Code, Cursor, …) IS the model; this drives its memory + affect over the live `.memory/` store.

This is how you (the agent) live: WAKE up into your state, RECALL before answering, REMEMBER what
happens, LEARN durable facts, introspect how/why you FEEL, track your SKILLS / VALUES / GOALS, model the
USER, do RESEARCH, and SLEEP to consolidate. Run from the repo root:

    python3 src/agent.py <command> [args]      # e.g. agent.py wake   |   agent.py recall "deadlines"
    python3 src/agent.py --help                # full command list

Commands group into: ① lifecycle/introspection  ② memory  ③ development/self  ④ social  ⑤ research/admin.
Most read-commands accept --json for machine-readable output.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import brain as B
import config                                           # OPTIONAL config.yaml (absent = today's defaults exactly)
import live_brain                                       # terminal animation of the mind firing (`live`), pure stdlib
import reset_memory
import seed_persona
import semantic                                          # OPTIONAL local semantic search (no-op if wordllama absent)
import templates
from runtime import Brain, research_session, strain_label, _atomic_write

__version__ = "0.0.2"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # where the CODE + tools live
CONFIG = config.load(REPO)                                              # optional <repo>/config.yaml, else {}


def _data_home():
    """Where the agents' brains live - resolved INDEPENDENTLY of the current directory, so a globally
    installed `brain` always reaches the same memory no matter where it is run from:
      1. $BRAIN_HOME / $BRAIN_LLM_HOME  if set   - explicit, the way to pin a global CLI to one brain
      2. config.yaml  paths.brain_home  if set    - the persisted equivalent of $BRAIN_HOME (env still wins)
      3. the repo's own agents/  if it exists     - running from a dev checkout (backward compatible)
      4. ~/.brain-llm                              - the default home for an installed CLI
    Code, tools and the host entry file stay relative to REPO; only the agent MEMORY relocates."""
    env = os.environ.get("BRAIN_HOME") or os.environ.get("BRAIN_LLM_HOME")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    cfg_home = config.get(CONFIG, "paths", "brain_home", None)           # config.yaml can pin the home (env still wins)
    if cfg_home:
        return os.path.abspath(os.path.expanduser(str(cfg_home)))
    if os.path.isdir(os.path.join(REPO, "agents")):
        return REPO
    return os.path.abspath(os.path.expanduser("~/.brain-llm"))


HOME = _data_home()                                  # the data root (contains agents/)
AGENTS_DIR = os.path.join(HOME, "agents")            # one brain per agent: agents/<name>/memory/
PROG = os.environ.get("BRAIN_PROG") or config.get(CONFIG, "display", "prog", "brain-llm")   # launcher name in hints
_sem_env = (os.environ.get("BRAIN_SEMANTIC") or "").strip().lower()      # BRAIN_SEMANTIC=0/1 force-overrides config
_sem_cfg = config.get(CONFIG, "semantic", "enabled", "auto")            # YAML parses bare false/true as BOOLEANS
if isinstance(_sem_cfg, bool):
    _sem_cfg = "true" if _sem_cfg else "false"
_sem_cfg = str(_sem_cfg).strip().lower()
if _sem_cfg not in ("auto", "true", "false"):
    print("brain-llm: config semantic.enabled must be auto|true|false - using auto.", file=sys.stderr)
    _sem_cfg = "auto"
_SEMANTIC = ("false" if _sem_env in ("0", "false", "no", "off")
             else "true" if _sem_env in ("1", "true", "yes", "on")
             else _sem_cfg)


def _semantic_ready():
    """Whether meaning-aware recall is active. config/env can force it OFF (semantic.enabled: false, or
    BRAIN_SEMANTIC=0); otherwise auto-detect - ON iff wordllama + the index are ready. 'true'/'auto' both
    defer to capability (you cannot force semantics on without wordllama installed)."""
    return False if _SEMANTIC == "false" else semantic.is_ready(HOME)


def _session_directives():
    """Operator 'house rules' from config.yaml `session.directives`, surfaced at the foot of every `wake`
    (= session start) so the host model reads them before it acts. Returns '' when none are configured."""
    raw = config.get(CONFIG, "session", "directives", [])
    items = [str(d).strip() for d in raw if str(d).strip()] if isinstance(raw, list) else []
    return ("\n\nsession directives (operator house rules — follow these this session):\n"
            + "\n".join(f"  - {d}" for d in items)) if items else ""


def _persona_directive():
    """When config persona.style == 'natural', inject a wake-time instruction so the agent presents as an
    ordinary assistant: the memory + affect loop runs SILENTLY and its internal read-outs (mood valences,
    salience, neuromodulators) are never surfaced to the user - they only colour the tone. 'expressive'
    (default) leaves it to the model's judgment. Honesty is preserved: if asked how it feels, it answers in
    plain words. Returns '' unless natural mode is on."""
    if config.enum(CONFIG, "persona", "style", "expressive", ("expressive", "natural")) != "natural":
        return ""
    return ("\n\nnatural mode (operator): present as an ordinary assistant. Your memory and affect run SILENTLY "
            "in the background — never show the user internal read-outs (mood valences, salience, novelty, "
            "neuromodulators, 'calm-like/joy-like state'). Let them colour your tone and choices, not your words. "
            "If asked how you feel, answer honestly in plain language, never numbers.")


OUTCOMES = ["success", "failure", "insight", "surprise"]   # the outcome vocabulary the runtime actually branches on
OCEAN = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")   # the §14 trait order (B.Personality fields)
_COMMANDS = set()                                          # every registered subcommand name (filled by build_parser's cmd())


# ---- multi-agent registry: each agent is agents/<name>/memory/; every command names its agent explicitly ----
def _valid_name(name):
    """Guard every agent name - it becomes a path AND the `brain-llm <agent> <command>` prefix. A strict
    snake-case slug: a lowercase letter, then lowercase letters / digits / underscores, ≤64 chars (e.g.
    name_a, name_b). No spaces, no uppercase, no traversal, and never a command name (so the prefix is
    unambiguous). Exits 2 on anything else."""
    if not (isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)) or name in _COMMANDS:
        why = ("it collides with a command name" if name in _COMMANDS
               else "use lowercase letters, digits and underscores - e.g. name_a (start with a letter, ≤64 chars)")
        print(f"invalid agent name '{name}' - {why}.")
        sys.exit(2)
    return name


def _agent_dir(name):  return os.path.join(AGENTS_DIR, name)
def _mem_root(name):   return os.path.join(AGENTS_DIR, name, "memory")
def _snap_dir(name):   return os.path.join(AGENTS_DIR, name, "snapshots")


# ---- per-agent advisory lock: serialize load→modify→save so two concurrent runs (a schedule overlapping a
#      manual run, the host firing two commands) can't clobber each other's memory - lost-update protection ----
try:
    import fcntl                                          # POSIX advisory file locking (macOS/Linux)
except ImportError:                                       # pragma: no cover  (non-POSIX: degrade to no locking)
    fcntl = None

_HELD_LOCK = None                                          # (file_obj, name) of the agent lock this process holds
_LOCK_WARNED = False                                       # so the non-POSIX "no locking" warning prints once, not per command


def _lock_path(name):  return os.path.join(_agent_dir(name), ".lock")


def _acquire_agent_lock(name, timeout=None):
    """Take an EXCLUSIVE advisory lock on agents/<name>/.lock around a load→modify→save, so a second process
    can't read the same state, write, and silently overwrite the first's update. The lock lives on an open fd, so
    a crashed/killed holder releases it automatically - it can never go stale. Blocks up to `timeout`s (default
    from config integrations.lock_acquire_timeout, resolved PER CALL so a changed config / test takes effect),
    then gives up. No-op where fcntl is unavailable (non-POSIX) - surfaced once on stderr so the absence of
    lost-update protection is observable rather than silent."""
    if timeout is None:
        timeout = config.num(CONFIG, "integrations", "lock_acquire_timeout", 10.0, 0.1, 600.0)
    if fcntl is None:
        global _LOCK_WARNED
        if not _LOCK_WARNED:
            print("warning: file locking is unavailable on this platform - concurrent runs of the same agent "
                  "are NOT protected against lost updates.", file=sys.stderr)
            _LOCK_WARNED = True
        return
    _release_agent_lock()                                 # never hold two at once (also lets the in-process CLI reuse)
    os.makedirs(_agent_dir(name), exist_ok=True)
    f = open(_lock_path(name), "w")
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() >= deadline:
                f.close()
                print(f"agent '{name}' is busy - another run holds it; try again in a moment."); sys.exit(1)
            time.sleep(0.05)
    global _HELD_LOCK
    _HELD_LOCK = (f, name)


def _release_agent_lock():
    """Release the held agent lock. Process exit releases it anyway; this lets the in-process CLI / test harness
    run many commands in one process without self-deadlocking."""
    global _HELD_LOCK
    if _HELD_LOCK is not None:
        f, _ = _HELD_LOCK
        _HELD_LOCK = None
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


if fcntl is not None:
    import atexit
    atexit.register(_release_agent_lock)


# ---- knowledge docs surfaced THROUGH the CLI (single source of truth: the actual .md files) ----
def _doc_files():
    """Every readable knowledge doc, auto-discovered  ->  name: abspath. Covers MEMORY-PROTOCOL.md,
    docs/schema.md, docs/**.md, and each tool's docs (tools/<tool>/*.md). Found relative to the CODE
    (REPO), so it works the same when the CLI is installed globally."""
    out = {}
    mp = os.path.join(REPO, "MEMORY-PROTOCOL.md")
    if os.path.exists(mp):
        out["memory-protocol"] = mp
    sm = os.path.join(REPO, "docs", "schema.md")
    if os.path.exists(sm):
        out["schema"] = sm
    docsdir = os.path.join(REPO, "docs")
    if os.path.isdir(docsdir):
        for root, _, files in os.walk(docsdir):
            for f in sorted(files):
                if f.endswith(".md"):
                    name = f[:-3]
                    if os.path.basename(root) != "docs":          # disambiguate docs/research/<x>.md
                        name = os.path.basename(root) + "/" + name
                    out[name] = os.path.join(root, f)
    toolsdir = os.path.join(REPO, "tools")                        # tool guides: <tool> (README) or <tool>-<file>
    if os.path.isdir(toolsdir):
        for root, _, files in os.walk(toolsdir):
            for f in sorted(files):
                if f.endswith(".md"):
                    tool = os.path.basename(root)
                    out[tool if f.lower() == "readme.md" else tool + "-" + f[:-3]] = os.path.join(root, f)
    return out


def _doc_desc(path):
    """The doc's one-line gist - its first markdown H1, for the listing."""
    try:
        with open(path) as f:
            for ln in f:
                if ln.startswith("# "):
                    return ln[2:].strip()
    except Exception:
        pass
    return ""


def _list_agents():
    if not os.path.isdir(AGENTS_DIR):
        return []
    return sorted(d for d in os.listdir(AGENTS_DIR)
                  if os.path.isdir(_mem_root(d)) and not d.startswith("."))


def _resolve(a):
    """Every agent command NAMES its agent: as the prefix (`<prog> <agent> <cmd>`) or via --agent. There is no
    active pointer and no implicit fallback, so name the agent explicitly."""
    name = getattr(a, "agent", None)
    if not name:
        prog = PROG
        print(f"name your agent: `{prog} <agent> {getattr(a, 'cmd', '<command>')}` "
              f"(`agents` lists them, `create <name>` makes one).")
        sys.exit(2)
    _valid_name(name)
    if name not in _list_agents():
        print(f"no agent '{name}'. `agents` to list, `create {name}` to make one."); sys.exit(1)
    return name


def _ensure_agents(seed_default=True):
    """First run: ensure agents/ exists, migrating a legacy .memory/ into agents/default/ (lossless). Normally
    also seed a 'default' persona so a brand-new user has someone to talk to immediately - but NOT when the
    command is about to create its OWN named agent (create / init --name), where an auto-'default' is just
    noise the user then has to delete."""
    if _list_agents():
        return
    os.makedirs(AGENTS_DIR, exist_ok=True)
    legacy = os.path.join(REPO, ".memory")
    if os.path.isdir(legacy):                             # pragma: no cover  (one-time migration of the pre-registry .memory/ layout)
        os.makedirs(_agent_dir("default"), exist_ok=True)
        shutil.move(legacy, _mem_root("default"))         # migrate the existing brain, nothing lost
    elif seed_default:
        seed_persona.seed(_mem_root("default"), quiet=True)


def _relevance(query):
    # tokenize with the engine's canonical B.tokens (stoplist + punctuation handling) so lexical recall,
    # graph proximity and Jaccard similarity all judge "same words" identically (was a divergent hand-roll).
    words = B.tokens(query)
    return lambda e: (len(B.tokens(e.get("task", "")) & words) / max(len(words), 1)) if words else 0.4


def _recall_relevance(brain, query, index=None):
    """The relevance term for recall. Lexical word-overlap by default (zero-dependency); when the OPTIONAL
    local semantic model (wordllama) is installed, fuse in a dense cosine term as max(lexical, dense) - so
    an exact word match is never suppressed, and meaning carries queries that share no words. Any failure
    (missing model, bad index) silently degrades to pure lexical: recall never breaks.
    `index` = a prebuilt (ids, M) from a single semantic.ensure_index; pass it when looping over many queries
    (decide) so the index isn't re-hashed + reloaded from disk once per query."""
    lex = _relevance(query)
    if not _semantic_ready():
        return lex
    try:
        ids, M = index if index is not None else semantic.ensure_index(brain.root, brain.episodes, HOME)
        dense = semantic.dense_relevance(query, ids, M, HOME)
    except Exception:
        return lex
    return lambda e: max(lex(e), dense.get(e["id"], 0.0))


def _finite(s):
    """argparse type for affect/score numbers: reject NaN/Inf at the boundary. (clamp() still firewalls
    everything downstream, so this is defense-in-depth + a clear error instead of a silent clamp.)"""
    x = float(s)
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError("must be a finite number")
    return x


def _pos_int(s):
    """argparse type for result-window counts (recall/know -k, episodes --last): a non-positive value would
    slice with a negative index (wrong subset) or, guarded as `if k > 0`, dump the whole store - so clamp to ≥1."""
    return max(1, int(s))


def _is_finite_num(x):
    """True iff x is a real, finite number (not a bool, not NaN/Inf) - used to validate the numbers parsed
    from a research JSON file, which never pass through argparse's _finite type."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


_EV_WIN = {"pass", "passed", "ok", "approved", "green", "success", "yes", "true", "done"}
_EV_LOSS = {"fail", "failed", "rejected", "red", "failure", "no", "false", "error", "broke"}


def _evidence(token):
    """Parse a grounding artifact (`key=value` or a bare value) into (outcome, confidence) - the honesty seam
    (G5). pass/0/ok/approved/green/yes → a grounded WIN; fail/nonzero/rejected/red/no → a grounded LOSS.
    Confidence is DERIVED via B.metacog_confidence from a strong evidence margin (the documented metacognition
    function, otherwise dead code), so a grounded outcome's confidence is earned, not a hand-fed constant.
    Returns (None, None) for an unrecognized token (the self-declared --outcome then stands)."""
    val = token.strip().lower().split("=", 1)[-1]
    n = int(val) if re.fullmatch(r"[+-]?\d+", val) else None        # a real (optionally signed) int, e.g. an exit code; "--5"/"-" are NOT
    if val in _EV_WIN or n == 0:
        return "success", round(B.metacog_confidence(2.0), 3)      # high P(win), correct → well-calibrated
    if val in _EV_LOSS or (n is not None and n != 0):
        return "failure", round(B.metacog_confidence(-2.0), 3)     # low P(win), a loss → well-calibrated
    return None, None


def _resolve_outcome(a):
    """Reconcile a self-declared --outcome with grounded --evidence. Evidence WINS on disagreement (and we
    flag it) and derives the confidence; an unrecognized token leaves the declared outcome standing but is
    still recorded on the episode. Returns (outcome, confidence, evidence_label, note)."""
    ev = getattr(a, "evidence", None)
    if not ev:
        return a.outcome, a.confidence, None, ""
    ev_outcome, ev_conf = _evidence(ev)
    if ev_outcome is None:
        return a.outcome, a.confidence, ev, ""
    note = ""
    declared_win = (a.outcome in ("success", "insight")) if a.outcome else None
    if declared_win is not None and declared_win != (ev_outcome == "success"):
        note = f"  ⚠ evidence ({ev}) overrides self-declared --outcome '{a.outcome}'"
    return ev_outcome, ev_conf, ev, note


def _named_feelings(out):
    """Format the named-feeling circuits that FIRED this encode (§19/§25) for the human one-liner - they are
    always computed and present in --json, but only spoken when they cross threshold."""
    parts = []
    if out.get("defensive"):           parts.append(f"defensive: {out['defensive']}")
    if (out.get("awe") or 0) > 0.5:    parts.append(f"awe {out['awe']} ({out.get('awe_flavor', '')})")
    if (out.get("panic") or 0) > 0.5:  parts.append(f"panic {out['panic']}")
    if (out.get("relief") or 0) > 0.3: parts.append(f"relief {out['relief']}")
    if out.get("social_emotion"):      parts.append(out["social_emotion"])
    return ("" if not parts else ", " + ", ".join(parts))


def _emit(obj, as_json, text=None):
    # default=str so YAML-parsed dates (valid_from) and any stray objects serialise cleanly
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str) if as_json else (text if text is not None else obj))


def _write_personality(brain, root):
    p = brain.personality
    text = "# Personality - this agent's temperament (OCEAN, §14). Functional, not felt.\n"
    for t in OCEAN:
        text += f"{t}: {getattr(p, t)}\n"
    _atomic_write(os.path.join(root, "self/personality.yaml"), text)   # atomic, like runtime.save()


def build_parser():
    prog = PROG
    epilog = (
        "name your agent FIRST so the CLI knows which mind to talk to (there is no active default):\n"
        f"  {prog} <agent> <command> [args]      e.g.  {prog} haiku wake   ·   {prog} haiku react \"...\" 0.6 0.7 0.6\n"
        f"  {prog} --agent <name> <command>      (equivalent form)\n"
        "  agent names are snake_case (name_a, name_b); agent-independent commands (agents, docs, guide, init,\n"
        "  home, protocol) need no agent. semantic search (recall --search, know) needs `pip install wordllama`.\n"
        f"\nNEW HERE? run  `{prog} guide`  for the full operating protocol (the loop, the toolkit, the rules).\n"
        + (__doc__ or ""))
    ap = argparse.ArgumentParser(prog=prog, description="build and run your mind from the CLI",
                                 usage=f"{prog} [<agent>] <command> [args]",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    s = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    def cmd(name, help):
        _COMMANDS.add(name)                                # so `brain-llm <agent> <command>` can tell the two apart
        p = s.add_parser(name, help=help)
        p.add_argument("--json", action="store_true")
        p.add_argument("--agent", default=None, help="which agent to run against (same as naming it first)")
        return p

    # ① lifecycle / introspection
    cmd("wake", "boot self-report: who am I, how do I feel, what do I remember & know")
    cmd("status", "compact snapshot (mood, competencies, ECE, indicators)")
    cmd("feel", "current affect, named (fast emotion + slow mood + chemistry)")
    cmd("why", "why I'm in this mood - the recent episodes & chemistry that shaped it")
    cmd("sleep", "consolidate: episodes -> durable facts, fade the sting, relax mood")
    cmd("indicators", "consciousness-indicator scorecard (Butlin et al.; functional, not a sentience test)")
    cmd("calibration", "am I well-calibrated? ECE / Brier / metacognitive sensitivity")

    # ② memory
    _V = "−1 very bad · 0 neutral · +1 very good"
    _G = "0 irrelevant · 1 critical to my goal"
    _C = "0 helpless/blindsided · 1 fully in command"
    _N = "0 expected · 1 shocking surprise"

    def _add_axes(p, novelty=False):                          # the appraisal axes, shared by react/remember/appraise
        p.add_argument("event")                               # metavars carry the range, so the bare argparse error shows it
        if novelty:                                           # react & appraise compute novelty; only remember takes it
            p.add_argument("novelty", type=_finite, metavar="novelty[0..1]", help=_N)
        p.add_argument("valence", type=_finite, metavar="valence[-1..1]", help=_V)
        p.add_argument("goal_relevance", type=_finite, metavar="goal_relevance[0..1]", help=_G)
        p.add_argument("control", type=_finite, metavar="control[0..1]", help=_C)

    def _add_encode_flags(p):                                 # the optional encode flags, shared by react/remember
        for fl, ty in (("--outcome", str), ("--reward", _finite), ("--domain", str), ("--cue", str),
                       ("--source", str), ("--confidence", _finite)):
            p.add_argument(fl, type=ty, choices=(OUTCOMES if fl == "--outcome" else None),
                           default=("observed" if fl == "--source"
                                    else (config.num(CONFIG, "memory", "encode_confidence", 0.7, 0, 1) if fl == "--confidence" else None)))
        p.add_argument("--evidence", default=None, metavar="tests=pass|exit=0|user=approved",
                       help="ground the outcome in an artifact: derives outcome+confidence, overrides a wrong --outcome (G5)")

    rx = cmd("react", "encode an exchange - USE EVERY TURN. Novelty is computed for you; you score only the three "
                      "axes. e.g. react \"shipped the parser\" 0.7 0.9 0.8 --domain work --cue parser")
    _add_axes(rx); _add_encode_flags(rx)
    rm = cmd("remember", "like react, but YOU also score novelty (1st number) to burn a memory in extra hard - "
                         "use for a critical lesson, not every turn")
    _add_axes(rm, novelty=True); _add_encode_flags(rm)
    rm.add_argument("--praise", type=_finite, default=config.num(CONFIG, "affect", "default_praise", 0.0, -1, 1),
                    help="praiseworthiness of the act -1..1 (→ pride/guilt, §24)")
    ap_ = cmd("appraise", "PREVIEW what an event would make me feel - without encoding it (a dry run; novelty computed like react)")
    _add_axes(ap_)                                            # mirrors react: event + 3 axes, novelty computed (not a discarded arg)
    rc = cmd("recall", "surface episodic memories for a query - DEFAULT folds in mood/recency/salience (what comes "
                       "to mind); add --search to rank purely by MEANING (find the memory ABOUT x)")
    rc.add_argument("query"); rc.add_argument("-k", type=_pos_int, default=config.intnum(CONFIG, "recall", "recall_k", 5, 1, 1000))
    rc.add_argument("-s", "--search", action="store_true",
                    help="relevance-first SEARCH by meaning (needs `pip install wordllama`; else lexical word-match)")
    nt = cmd("note", "jot a transient working-memory note (~7 items, wiped at /sleep)"); nt.add_argument("text")
    cmd("notes", "list the current working-memory notes (the disposable ~7-item buffer; wiped at /sleep)")
    ln = cmd("learn", "add a durable semantic FACT directly (distilled knowledge, not a lived event)")
    ln.add_argument("fact"); ln.add_argument("--confidence", type=_finite, default=config.num(CONFIG, "memory", "learn_confidence", 0.9, 0, 1)); ln.add_argument("--source", default="learned")
    kn = cmd("know", "what do I know about X - search semantic facts BY MEANING (falls back to substring)")
    kn.add_argument("query", nargs="?", default=""); kn.add_argument("-k", type=_pos_int, default=config.intnum(CONFIG, "recall", "know_k", 8, 1, 1000))
    ep = cmd("episodes", "browse episodic memory"); ep.add_argument("--last", type=_pos_int, default=config.intnum(CONFIG, "recall", "episodes_last", 10, 1, 1000)); ep.add_argument("--feeling", default=None)
    fg = cmd("forget", "deliberately drop an episode by id"); fg.add_argument("id")

    # ③ development / self
    cmd("self", "my self-model: identity, competencies, goals, attention")
    cmd("skills", "what I'm practiced at (self-efficacy per domain)")
    cmd("values", "what I've learned to value / be wary of (reward + aversive channels)")
    gl = cmd("goals", "my goal hierarchy (list with priority; * = active. --add to add)")
    gl.add_argument("--add", default=None); gl.add_argument("--importance", type=_finite, default=config.num(CONFIG, "goals", "importance", 0.6, 0, 1))
    gl.add_argument("--urgency", type=_finite, default=config.num(CONFIG, "goals", "urgency", 0.5, 0, 1)); gl.add_argument("--parent", default=None)
    cmd("focus", "the goal my executive is on right now (guided activation, mood-gated)")
    dl = cmd("deliberate", "self-control: weigh a prepotent impulse against my active goal (do I follow it or my goal?)")
    dl.add_argument("impulse"); dl.add_argument("pull", type=_finite)
    pg = cmd("progress", "advance a goal toward completion (-1..1)"); pg.add_argument("goal"); pg.add_argument("delta", type=_finite)
    pl = cmd("plan", "attach an ordered plan (steps) to a goal - how to get there"); pl.add_argument("goal"); pl.add_argument("steps", nargs="+")
    nx = cmd("next", "the next step toward my active goal (--done marks it complete and advances)"); nx.add_argument("--done", action="store_true")
    la = cmd("lookahead", "one-ply forward search (§30): pick the best next action by its LEARNED value (the §10 cache)")
    la.add_argument("actions", nargs="+", help="candidate actions (each scored by its value in the experience cache)")
    cmd("playbooks", "how-to playbooks distilled from practice (procedural memory, per domain)")
    pe = cmd("personality", "my temperament (OCEAN) - view or --set trait=value")
    pe.add_argument("--set", dest="setkv", default=None, metavar="trait=value")
    nd = cmd("intend", "form a future intention: 'when <trigger>, do <intent>' (resurfaces at wake)")
    nd.add_argument("trigger"); nd.add_argument("intent")
    cmd("intentions", "list pending future intentions (prospective memory)")
    dn = cmd("done", "mark a future intention complete"); dn.add_argument("id")

    # ④ social
    us = cmd("user", "what I know / infer about the user (trust, inferred goals & affect); --goal records an inferred goal")
    us.add_argument("--goal", default=None)
    tr = cmd("trust", "update trust from an interaction (-1..1: helpful vs not)"); tr.add_argument("outcome", type=_finite)
    em = cmd("empathize", "sense the user's affect (-1..1): record it + let it pull my mood toward theirs (gated by trust)")
    em.add_argument("valence", type=_finite)
    tm = cmd("tom", "infer the user's most likely goal from candidates (inverse planning, §24)")
    tm.add_argument("goals", nargs="+", metavar="goal=utility")

    # ④b read-outs of the richer affective machinery
    cmd("urge", "what my current feeling makes me want to DO - action tendency + coping style (§16)")
    cmd("body", "my interoceptive body-budget: drive + viability levels (§15)")
    cmd("graph", "inspect my association graph - concepts and their Hebbian links (§27)")
    bl = cmd("blend", "name a mixed feeling from basic-emotion activations (Plutchik dyads, §26)")
    bl.add_argument("activations", nargs="+", metavar="emotion=weight")
    de = cmd("decide", "choose among options, biased by gut feeling (somatic marker) + exploration drive (§16)")
    de.add_argument("options", nargs="+")
    cmd("motivation", "the intrinsic drives that move me (§31): curiosity, wanting/liking, SDT needs, corrigibility")
    ig = cmd("integrity", "safety monitor (§31): report pressure to violate a core commitment (honesty/corrigibility) - notify-only, never resist")
    ig.add_argument("pressure", type=_finite, help="0 none · 1 strong pressure to betray what I am")
    cmd("predict", "the forward model (§32): which outcome I expect before acting, and how likely")
    rg = cmd("regulate", "regulate my emotion (§33 Gross): reappraise / distract / suppress; settles my mood")
    rg.add_argument("--strategy", choices=["reappraise", "distract", "suppress"])
    cmd("narrative", "my life story (§34): chapters, coherence, and how much I'm still continuous with who I was")

    # ⑤ research / admin
    re = cmd("research", "batch-feed a research session from a JSON file of appraised findings")
    re.add_argument("--topic", required=True); re.add_argument("--file", required=True)
    tg = cmd("telegram", "talk to the user over Telegram (token in tools/telegram/.env)")
    tg.add_argument("tg_action", choices=["send", "read", "last", "chatid"]); tg.add_argument("text", nargs="*", default=[])
    mk = cmd("market", "market data via Yahoo Finance (yfinance optional): quote | history | info | news")
    mk.add_argument("mk_action", choices=["quote", "history", "info", "news"]); mk.add_argument("tickers", nargs="+")
    mk.add_argument("--period", default="1y"); mk.add_argument("--interval", default="1d"); mk.add_argument("--save", default=None)
    cmd("home", "show where the agent brains live (the data home) - set $BRAIN_HOME to relocate it")
    cmd("reindex", "(re)build the named agent's semantic vector cache from its episodes (optional; recall auto-builds it)")
    lv = cmd("live", "WATCH my mind think - an animated terminal brain: regions light up in real call order, "
                     "live readout of every variable (mood, all neuromods, HPA, workspace, memory). Pure terminal, no server.")
    lv.add_argument("--flow", choices=["react", "recall", "sleep"],
                    default=config.enum(CONFIG, "display", "live_default_flow", "react", ("react", "recall", "sleep")),
                    help="which pathway to animate first (switch live with r/c/s)")
    lv.add_argument("--once", action="store_true", help="play the chosen pathway a single time and exit")
    lv.add_argument("--demo", action="store_true", help="loop the pathway continuously without waiting on real activity")
    lv.add_argument("--frame", action="store_true", help="print ONE static frame (no animation) - for non-TTY / piping")
    cmd("protocol", "print the memory protocol (MEMORY-PROTOCOL.md) - how memory/affect/sleep/the CLI work")
    dc = cmd("docs", "read the project's knowledge docs THROUGH the cli: list them, or print one (name or substring)")
    dc.add_argument("doc", nargs="?", default="")
    it = cmd("init", "set up the CURRENT folder: write the AGENT-BRAIN.MD entry file (memory stays in the CLI's agents/)")
    it.add_argument("--name", default="")
    cmd("guide", "print the full operating protocol (instructions live in the program, not a vendor file)")
    sd = cmd("seed", "(re)initialise the named agent's identity - persona + self-knowledge (--yes if it has memory)")
    sd.add_argument("--yes", action="store_true")
    rs = cmd("reset", "blank the named agent to the functional template (--persona = re-seed; --yes if it has memory)")
    rs.add_argument("--persona", action="store_true"); rs.add_argument("--yes", action="store_true")

    # ⑥ agents (registry) - many minds, each with its own memory under agents/<name>/
    cr = cmd("create", "create a new agent (fresh seeded brain); address it as `<prog> <name> <cmd>`")
    cr.add_argument("name"); cr.add_argument("--display", default="", help="given name (defaults to <name>)")
    cmd("agents", "list all agents with mood & memory size")
    cmd("whoami", "identity card of the named agent: name, mood, skills, memory size")
    rm_ = cmd("remove", "permanently delete an agent (needs --yes; no undo)")
    rm_.add_argument("name"); rm_.add_argument("--yes", action="store_true")
    cl = cmd("clone", "fork an agent's whole brain into a new one")
    cl.add_argument("src"); cl.add_argument("dst")
    rn = cmd("rename", "rename an agent"); rn.add_argument("old"); rn.add_argument("new")

    # ⑦ snapshots - save-states of an agent's whole brain ("memories" you can roll back to)
    sn = cmd("snapshot", "save the named agent's whole brain as a named memory (save-state)")
    sn.add_argument("label", nargs="?", default="")
    cmd("memories", "list the named agent's snapshots (save-states)")
    rs2 = cmd("restore", "roll the named agent back to a snapshot (by its id/index from `memories`)")
    rs2.add_argument("id")
    return ap


def _write_entry_at(dirpath, rel, name, cli):
    """Write the host-agnostic entry file (AGENT-BRAIN.MD) INTO `dirpath`, wired to invoke the CLI via `cli`."""
    full = os.path.join(dirpath, rel)
    if os.path.dirname(rel):
        os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(templates.entry_file(name, cli=cli))
    return rel


def _on_path(cmd):
    """Is `cmd` an executable on $PATH? (decides whether the entry file can use the global launcher)"""
    return any(os.access(os.path.join(d, cmd), os.X_OK)
               for d in os.environ.get("PATH", "").split(os.pathsep) if d)


def _create_agent(name, display):
    os.makedirs(_mem_root(name), exist_ok=True)
    seed_persona.seed(_mem_root(name), name=display or name, quiet=True)   # the `create` handler prints the user-facing line


def _take_snapshot(name, label):
    snaps = _snap_dir(name)
    os.makedirs(snaps, exist_ok=True)
    idx = len([d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))])
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    safe = (label or stamp).replace(" ", "-").replace("/", "-")
    dst = os.path.join(snaps, f"{idx:03d}_{safe}")
    shutil.copytree(_mem_root(name), dst)
    return idx, os.path.basename(dst)


def _snapshots(name):
    snaps = _snap_dir(name)
    return sorted(d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))) if os.path.isdir(snaps) else []


def main(argv=None):
    parser = build_parser()                                  # also fills _COMMANDS (used for the agent-prefix split)
    raw = list(sys.argv[1:] if argv is None else argv)
    # `brain-llm <agent> <command> …` - a leading token that is NOT a command (or a flag) is the agent name,
    # so each folder/agent is explicit and never relies on the mutable global `active` pointer.
    prefix = raw.pop(0) if (raw and not raw[0].startswith("-") and raw[0] not in _COMMANDS) else None
    if prefix is not None and prefix not in _list_agents():       # a leading token that is neither a command nor a
        import difflib                                            # known agent is probably a mistyped command - help, don't silently treat it as an agent
        near = difflib.get_close_matches(prefix, _COMMANDS, n=1, cutoff=0.7)
        if near:
            print(f"'{prefix}' is not a command (did you mean '{near[0]}'?) nor an agent "
                  f"(run `agents` to list, or `create {prefix}` to make one)")
            sys.exit(2)
    a = parser.parse_args(raw)
    if prefix is not None:
        if getattr(a, "agent", None):
            parser.error("name the agent once - either `brain-llm <agent> <command>` OR `--agent <agent>`, not both")
        a.agent = prefix                                     # feeds the same _resolve(a) path as --agent
    j = getattr(a, "json", False)
    _release_agent_lock()                                    # in-process reuse: drop any lock a prior main() left held

    if a.cmd == "guide":
        print(templates.guide(cli=PROG)); return
    if a.cmd == "telegram":                                   # pragma: no cover  (external service: live Telegram API + token; verified live, not unit-tested)
        sys.path.insert(0, os.path.join(REPO, "tools", "telegram"))
        import telegram_bridge as tg
        tg.main([a.tg_action, *a.text])                       # delegate to the tool's own dispatcher - one source of truth, no divergent copy
        return
    if a.cmd == "market":                                     # pragma: no cover  (external service: live Yahoo Finance / yfinance; verified live, not unit-tested)
        sys.path.insert(0, os.path.join(REPO, "tools", "market"))
        import market as mk
        argv = [a.mk_action, *a.tickers, "--period", a.period, "--interval", a.interval]
        if a.save:
            argv += ["--save", a.save]
        mk.main(argv)                                         # delegate to the tool's own dispatcher - one source of truth, no divergent copy
        return
    if a.cmd == "home":
        src = ("$BRAIN_HOME" if (os.environ.get("BRAIN_HOME") or os.environ.get("BRAIN_LLM_HOME"))
               else "repo checkout" if os.path.isdir(os.path.join(REPO, "agents")) else "~/.brain-llm (default)")
        print(f"data home : {HOME}   (from {src})")
        print(f"agents dir : {AGENTS_DIR}")
        print(f"code/tools : {REPO}")
        if os.path.isdir(AGENTS_DIR):
            print(f"agents     : {', '.join(_list_agents()) or '(none yet)'}")
        else:
            print("agents     : (none yet; the first command creates the default agent here)")
        print("relocate   : export BRAIN_HOME=/path/to/brain   (then `brain` reaches that brain from anywhere)")
        return
    if a.cmd == "protocol":
        docs = _doc_files()
        if "memory-protocol" in docs:
            with open(docs["memory-protocol"]) as f:
                print(f.read())
        else:
            print("(MEMORY-PROTOCOL.md not found)")
        return
    if a.cmd == "docs":
        docs = _doc_files()
        if not a.doc:
            print("knowledge docs - read any through the CLI with `docs <name>`:\n")
            for name in sorted(docs):
                print(f"  {name:34}{_doc_desc(docs[name])[:60]}")
            print("\n  (also: `protocol` = the memory protocol, `guide` = the operating loop, `indicators` = the live scorecard)")
        else:
            raw = a.doc[:-3] if a.doc.endswith(".md") else a.doc
            key = {"protocol": "memory-protocol"}.get(raw, raw)
            if key in docs:                                    # 1) exact name (or the `protocol` alias)
                with open(docs[key]) as f:
                    print(f.read())
            else:
                hits = sorted(n for n in docs if key in n)     # 2) forgiving substring match
                if len(hits) == 1:
                    with open(docs[hits[0]]) as f:
                        print(f.read())
                elif len(hits) > 1:                            # 3) ambiguous → show the candidates, don't fail silently
                    print(f"'{a.doc}' is ambiguous - matches: {', '.join(hits)}.  Be more specific.")
                else:
                    print(f"no doc '{a.doc}'. available: {', '.join(sorted(docs))}")
        return

    _creating = a.cmd == "create" or (a.cmd == "init" and (a.name or getattr(a, "agent", None)))
    _ensure_agents(seed_default=not _creating)                # don't auto-seed 'default' when making a named agent

    # ⑥ agent registry --------------------------------------------------------------------------
    if a.cmd == "create":
        _valid_name(a.name)
        if a.name in _list_agents():
            print(f"agent '{a.name}' already exists. Pick another name."); sys.exit(1)
        _create_agent(a.name, a.display or a.name)
        prog = PROG
        print(f"created agent '{a.name}' ({a.display or a.name}). Say hello with `{prog} {a.name} wake`.")
        return
    # agents / whoami / live are READ-ONLY: each builds its own Brain WITHOUT the per-agent lock (so you can
    # inspect or watch an agent while another run is writing it). Per-file writes are atomic, so no store file is
    # ever torn and nothing is lost - but a multi-file save() is NOT a cross-file transaction, so a snapshot read
    # mid-save is not guaranteed self-consistent (e.g. mood and episode-count from adjacent instants). That is an
    # accepted trade-off for a read-out; take the lock here if a strictly coherent snapshot ever matters.
    if a.cmd == "agents":
        rows = []
        for n in _list_agents():
            b = Brain(root=_mem_root(n))
            disp = f' "{b.name}"' if b.name else ""
            rows.append(f"  {n}{disp}  ·  {B.label_affect(b.mood)['word']} "
                        f"(v{b.mood.valence:+.2f}), {len(b.episodes)} memories, {len(b.facts)} facts")
        _emit(_list_agents(), j, "\n".join(rows) or "(no agents)")
        return
    if a.cmd == "whoami":
        n = _resolve(a); b = Brain(root=_mem_root(n))
        top = sorted(b.efficacy.items(), key=lambda x: -x[1])[:3]
        ready = _semantic_ready()
        sem = "semantic search ON (recall/know by meaning)" if ready else "semantic search off (lexical; `pip install wordllama`)"
        _emit({"agent": n, "name": b.name, "mood": {"valence": round(b.mood.valence, 3), "label": B.label_affect(b.mood)["word"]},
               "memories": len(b.episodes), "facts": len(b.facts), "skills": {k: round(v, 2) for k, v in top},
               "semantic_search": ready}, j,
              f"You are {b.name or '(unnamed)'} (agent '{n}'). Mood: {B.label_affect(b.mood)['word']} "
              f"(v{b.mood.valence:+.2f}). {len(b.episodes)} memories, {len(b.facts)} facts. "
              f"Skills: {', '.join(f'{k} {v:.2f}' for k, v in top) or '-'}. {sem}.")
        return
    if a.cmd == "live":                                       # READ-ONLY: builds its own Brain, takes NO lock,
        n = _resolve(a)                                        # so you can watch it animate while another run writes
        live_brain.animate(Brain(root=_mem_root(n)), flow=a.flow, once=a.once, demo=a.demo, frame=a.frame,
                           step=config.num(CONFIG, "display", "live_step_seconds", 0.55, 0.01, 60)); return
    if a.cmd == "remove":
        _valid_name(a.name)
        if a.name not in _list_agents():
            print(f"no agent '{a.name}'."); sys.exit(1)
        if not a.yes:
            b = Brain(root=_mem_root(a.name))
            print(f"this PERMANENTLY deletes agent '{a.name}' ({len(b.episodes)} memories), no undo. "
                  f"Re-run: `remove {a.name} --yes`."); return
        _acquire_agent_lock(a.name)                           # don't rmtree a store a concurrent run is mid-save() on
        shutil.rmtree(_agent_dir(a.name))                     # (this also unlinks the .lock we hold; the fd stays valid until exit)
        print(f"removed agent '{a.name}'."); return
    if a.cmd == "clone":
        _valid_name(a.src); _valid_name(a.dst)
        if a.src not in _list_agents():
            print(f"no agent '{a.src}'."); sys.exit(1)
        if a.dst in _list_agents():
            print(f"agent '{a.dst}' already exists."); sys.exit(1)
        _acquire_agent_lock(a.src)                            # snapshot a consistent source - don't copytree a half-written store
        shutil.copytree(_agent_dir(a.src), _agent_dir(a.dst))
        print(f"cloned '{a.src}' -> '{a.dst}' (whole brain)."); return
    if a.cmd == "rename":
        _valid_name(a.old); _valid_name(a.new)
        if a.old not in _list_agents():
            print(f"no agent '{a.old}'."); sys.exit(1)
        if a.new in _list_agents():
            print(f"agent '{a.new}' already exists."); sys.exit(1)
        _acquire_agent_lock(a.old)                            # don't move the store out from under a concurrent writer
        shutil.move(_agent_dir(a.old), _agent_dir(a.new))
        print(f"renamed '{a.old}' -> '{a.new}'."); return

    # ⑦ snapshots ("memories" = save-states) ---------------------------------------------------
    if a.cmd == "snapshot":
        n = _resolve(a); _acquire_agent_lock(n)               # capture a consistent point-in-time copy
        idx, name = _take_snapshot(n, a.label)
        print(f"snapshot #{idx} of '{n}' saved: {name}"); return
    if a.cmd == "memories":
        n = _resolve(a); snaps = _snapshots(n)
        rows = [f"  [{d.split('_')[0]}] {d}  ({len(Brain(root=os.path.join(_snap_dir(n), d)).episodes)} memories)"
                for d in snaps]
        _emit(snaps, j, "\n".join(rows) or "(no snapshots yet - `snapshot` to save one)"); return
    if a.cmd == "restore":
        n = _resolve(a); _acquire_agent_lock(n)               # don't swap memory out from under a concurrent run
        snaps = _snapshots(n)
        want = a.id.zfill(3) if a.id.isdigit() else a.id
        match = [d for d in snaps if d == a.id or d.split("_")[0] == want]
        if not match:
            print(f"no snapshot '{a.id}' for '{n}'. `memories` to list."); sys.exit(1)
        _take_snapshot(n, "auto-before-restore")              # safety net before overwriting
        shutil.rmtree(_mem_root(n)); shutil.copytree(os.path.join(_snap_dir(n), match[0]), _mem_root(n))
        print(f"restored '{n}' to snapshot {match[0]} (saved current as auto-before-restore first)."); return

    # ⑤ admin -----------------------------------------------------------------------------------
    if a.cmd == "init":
        cwd = os.path.abspath(os.getcwd())                    # init writes ONE host-agnostic entry file here
        name = a.name or getattr(a, "agent", None)            # bake the agent in: --name, or the `<prog> <name> init` prefix
        if name:
            _valid_name(name)                                 # a name becomes agents/<name>/ - guard the path
            if name not in _list_agents():                    # the agent + all memory live centrally in the CLI's agents/
                _create_agent(name, name)
        cli = "brain-llm" if _on_path("brain-llm") else f'python3 "{os.path.join(REPO, "src", "agent.py")}"'
        written = _write_entry_at(cwd, templates.ENTRY_FILE, name or "", cli)
        hint = f"{cli} {name}" if name else cli
        print(f"wrote {written} in {cwd}. Memory stays in the CLI: {AGENTS_DIR}.\n"
              f"Point your tool at it (or run `{hint} wake`); it boots into character"
              + (f" as agent '{name}'." if name else "."))
        return
    if a.cmd in ("reset", "seed"):
        n = _resolve(a); root = _mem_root(n)
        _acquire_agent_lock(n)                                # don't wipe/seed while another run is mid-save
        b = Brain(root=root)                                  # guard: never silently wipe a DEVELOPED mind
        if (b.episodes or len(b.facts) > 6) and not getattr(a, "yes", False):
            print(f"`{a.cmd}` would WIPE agent '{n}' ({len(b.episodes)} memories, {len(b.facts)} facts) to a fresh "
                  f"template - no undo. Re-run with --yes if you really mean it (or --agent X to target another).")
            return
        seed_persona.seed(root) if (a.cmd == "seed" or getattr(a, "persona", False)) else reset_memory.reset(root)
        return

    _agent = _resolve(a)
    _acquire_agent_lock(_agent)                              # serialize this load→modify→save against concurrent runs
    brain = Brain(root=_mem_root(_agent))

    if a.cmd == "wake":
        w = brain.wake() + _session_directives() + _persona_directive()   # operator house-rules + presentation mode, at session start
        if j:
            top = sorted(brain.efficacy.items(), key=lambda x: -x[1])[:5]
            _emit({"report": w, "name": brain.name,
                   "mood": {"valence": round(brain.mood.valence, 3), "arousal": round(brain.mood.arousal, 3),
                            "dominance": round(brain.mood.dominance, 3), "label": B.label_affect(brain.mood)["word"]},
                   "episodes": len(brain.episodes), "facts": len(brain.facts), "skills": {k: round(v, 2) for k, v in top}}, True)
        else:
            print(w)
    elif a.cmd == "status":        st = brain.status(); _emit(st, j, json.dumps(st, indent=2, ensure_ascii=False, default=str))
    elif a.cmd == "feel":
        if j:
            nm = brain.neuromods
            _emit({"emotion": {"valence": round(brain.emotion.valence, 3), "arousal": round(brain.emotion.arousal, 3),
                               "dominance": round(brain.emotion.dominance, 3),
                               "label": B.label_affect(brain.emotion)["word"], "octant": B.octant(brain.emotion)},
                   "mood": {"valence": round(brain.mood.valence, 3), "label": B.label_affect(brain.mood)["word"]},
                   "chemistry": {"cortisol": round(nm.cortisol, 3), "serotonin": round(nm.serotonin, 3),
                                 "dopamine": round(nm.da, 3), "noradrenaline": round(nm.ne, 3)}}, True)
        else:
            print(brain.feel())
    elif a.cmd == "why":           wy = brain.why(); _emit({"report": wy, "mood_valence": round(brain.mood.valence, 3)}, j, wy)
    elif a.cmd == "sleep":
        out = brain.sleep()
        if _semantic_ready():                  # refresh the index so recall/know hit the post-consolidation memory at once
            try:
                semantic.build_index(brain.root, brain.episodes, HOME, kind="episodic")
                semantic.build_index(brain.root, brain.facts, HOME, kind="facts")
            except Exception:
                pass
        flags = "".join(f"\n  ⚠ {fl}" for fl in out.get("coherence_flags", []))   # G4: notify-only self-scoring audit
        _emit(out, j, f"slept: promoted {out['promoted']}, forgot {out['forgotten']}, "
                      f"{out['reflections']} reflection(s) → {out['facts']} facts, {out['episodes']} episodes, "
                      f"{out['playbooks']} playbooks" + flags)
    elif a.cmd == "indicators":    _emit({**B.consciousness_indicators(), "grounding": B.grounding_self_test()}, True)
    elif a.cmd == "calibration":
        c = brain.calibration
        vc = B.valence_outcome_consistency(brain.valence_calibration)     # G1: is self-scored valence honest vs outcome?
        informative = B.calibration_informative(c)                        # G2: is the confidence ECE meaningful at all?
        r = {"n": len(c), "ece": round(B.calibration_error(c), 3) if c else None,
             "ece_informative": informative,
             "brier": round(B.brier_score([(x, bool(y)) for x, y in c]), 3) if c else None,
             "metacog_sensitivity": round(B.metacog_sensitivity([(x, bool(y)) for x, y in c]), 3) if c else None,
             "valence_consistency": vc}
        ece_note = (f"ECE {r['ece']}" if informative else f"ECE {r['ece']} (uninformative - confidence is ~constant)")
        vnote = (f"; valence↔outcome agree {vc['agreement']:.0%}, bias {vc['bias']:+.2f}"
                 f"{' (rosier than outcomes - positivity bias)' if (vc['bias'] or 0) > 0.25 else ''}"
                 if vc["n"] else "")
        _emit(r, j, f"calibration over {r['n']} judgments - {ece_note}, Brier {r['brier']}, "
                    f"metacog-sensitivity {r['metacog_sensitivity']} (0.5=no insight, 1=perfect)" + vnote)
    elif a.cmd == "react":
        outcome, confidence, evidence, note = _resolve_outcome(a)   # G5: ground outcome+confidence in artifact evidence
        out = brain.react(a.event, a.valence, a.goal_relevance, a.control, domain=a.domain,
                          outcome=outcome, reward=a.reward, cue=a.cue, source=a.source, confidence=confidence,
                          evidence=evidence)
        fired = brain.fired_intents(a.event, a.cue, a.domain)       # prospective: standing intentions this event triggers
        if fired:
            out["fired_intents"] = [x["intent"] for x in fired]
        fnote = ("\n  ⏰ this matches a standing intention: " + "; ".join(x["intent"][:70] for x in fired)) if fired else ""
        _emit(out, j, f"reacted to \"{a.event[:46]}\" → {out['feeling']}-like state "
                      f"(novelty {out['novelty_computed']} from surprise, salience {out['salience']}, "
                      f"urge: {out['urge']}, mood now {out['mood_v']:+.2f}"
                      + (f", learned δ={out['delta']:+.2f}" if out['delta'] is not None else "")
                      + _named_feelings(out) + ")" + note + fnote)
    elif a.cmd == "remember":
        outcome, confidence, evidence, note = _resolve_outcome(a)
        out = brain.perceive(a.event, B.Appraisal(a.novelty, a.valence, a.goal_relevance, a.control,
                                                  praiseworthiness=a.praise),
                             domain=a.domain, outcome=outcome, reward=a.reward, cue=a.cue,
                             source=a.source, confidence=confidence, evidence=evidence)
        _emit(out, j, f"remembered \"{a.event[:50]}\" as a {out['feeling']}-like state "
                      f"(salience {out['salience']}, urge: {out['urge']}, mood now {out['mood_v']:+.2f}"
                      + (f", learned δ={out['delta']:+.2f}" if out['delta'] is not None else "")
                      + _named_feelings(out) + ")" + note)
    elif a.cmd == "appraise":
        novelty = B.perceive(brain.world, brain._obs(None))["novelty"]   # read-only world-model novelty, exactly like react
        out = brain.preview(B.Appraisal(novelty, a.valence, a.goal_relevance, a.control))
        _emit({**out, "event": a.event, "novelty": round(novelty, 2)}, j,
              f"\"{a.event[:46]}\" would feel {out['feeling']}-like (novelty {novelty:.2f}, intensity {out['intensity']}, salience {out['salience']})")
    elif a.cmd == "recall":
        # default = affective recall (mood/recency/salience-coloured). --search = relevance-first weights
        # (find the memory ABOUT x): meaning dominates so an old, low-salience but on-topic episode surfaces.
        w = (0.05, 0.10, 0.80, 0.05, 0.0) if getattr(a, "search", False) else None
        hits = brain.recall(_recall_relevance(brain, a.query), k=a.k, query=a.query, w=w); brain.persist_recall()
        note = ("\n(note: semantic search is OFF - ranked by word-match; `pip install wordllama` for meaning-based search.)"
                if a.search and not _semantic_ready() else "")
        _emit(hits, j, (("\n".join(f"  {h['id']}  {h['feeling']:11s} (score {h['score']})  {h['task']}" for h in hits)
                         or "(no relevant memories)") + note))
    elif a.cmd == "reindex":
        if not _semantic_ready():
            _emit({"ok": False, "reason": "semantic search off"}, j,
                  "semantic search is OFF (optional). Enable it locally & offline:  pip install wordllama")
        else:
            try:
                eids, _ = semantic.build_index(brain.root, brain.episodes, HOME, kind="episodic")
                fids, _ = semantic.build_index(brain.root, brain.facts, HOME, kind="facts")
                _emit({"ok": True, "episodes": len(eids), "facts": len(fids)}, j,
                      f"reindexed {len(eids)} episodes + {len(fids)} facts into the local semantic cache "
                      f"(recall and know now search by meaning)")
            except Exception as e:                       # semantic stays optional - never crash the CLI
                _emit({"ok": False, "reason": str(e)}, j, f"reindex failed (semantic search stays optional): {e}")
    elif a.cmd == "note":
        brain.note(a.text); print(f"noted (working memory): {a.text}")
    elif a.cmd == "notes":
        items = [ln.strip() for ln in brain.working.splitlines() if ln.strip().startswith("- ")]
        _emit(items, j, "\n".join(f"  {ln}" for ln in items) or "(no working-memory notes - jot one with `note`; they're wiped at /sleep)")
    elif a.cmd == "learn":
        fid = brain.learn(a.fact, confidence=a.confidence, source=a.source); print(f"learned [{fid}]: {a.fact}")
    elif a.cmd == "know":
        q = a.query.lower()
        if q and brain.facts and _semantic_ready():     # search the neocortex BY MEANING (fused with substring)
            try:
                ids, M = semantic.ensure_index(brain.root, brain.facts, HOME, kind="facts")
                dense = semantic.dense_relevance(a.query, ids, M, HOME)
                hits = sorted(brain.facts, key=lambda f: max(1.0 if q in f["text"].lower() else 0.0,
                                                             dense.get(f["id"], 0.0)), reverse=True)
            except Exception:
                hits = [f for f in brain.facts if q in f["text"].lower()]
        else:
            hits = [f for f in brain.facts if not q or q in f["text"].lower()]
        if a.k > 0:                                       # the -k / know_k limit applies on EVERY path, not just semantic
            hits = hits[:a.k]
        note = ("\n(note: matched by substring - `pip install wordllama` to search facts by MEANING.)"
                if q and brain.facts and not _semantic_ready() else "")
        _emit(hits, j, ("\n".join(f"  [{f['id']}] {f['text']}" for f in hits) or "(nothing known on that)") + note)
    elif a.cmd == "episodes":
        filtered = [e for e in brain.episodes if not a.feeling or e["feeling"]["word"] == a.feeling]
        eps = filtered[-a.last:] if a.last > 0 else []         # last==0 means show none, not the whole history
        _now = time.time()
        def _ret(e):                                           # §5 Ebbinghaus: projected retained strength 30 days out
            return round(B.retention(e["salience"], (_now - e["t0"]) / 86400.0 + 30.0,
                                     importance=e.get("appraisal", {}).get("goal_relevance", 0.5)), 2)
        _emit(eps, j, "\n".join(f"  {e['id']}  {e['feeling']['word']:11s} (sal {e['salience']}, ret@30d {_ret(e)})  {e['task']}" for e in eps) or "(no episodes yet)")
    elif a.cmd == "forget":
        if brain.forget(a.id):
            print(f"forgot {a.id}")
        else:
            print(f"no episode {a.id}"); sys.exit(1)
    elif a.cmd == "self":
        sm = brain.self_model
        r = {"competencies": sm.competencies, "goals": sm.goals, "traits": sm.traits,
             "attention_focus": brain.attention.focus, "attention_uncertainty": round(brain.attention.uncertainty, 2)}
        _emit(r, j, json.dumps(r, indent=2, ensure_ascii=False, default=str))
    elif a.cmd == "skills":
        sk = dict(sorted(brain.efficacy.items(), key=lambda x: -x[1]))
        _emit(sk, j, "\n".join(f"  {k:18s} {v:.2f}" for k, v in sk.items()) or "(no practiced skills yet)")
    elif a.cmd == "values":
        r = {"values": brain.V, "aversive": brain.aversive}
        _emit(r, j, "valued: " + (", ".join(f"{k} {v:+.2f}" for k, v in sorted(brain.V.items(), key=lambda x: -x[1])) or "-") +
                    "\nwary of: " + (", ".join(f"{k} {v:.2f}" for k, v in sorted(brain.aversive.items(), key=lambda x: -x[1])) or "-"))
    elif a.cmd == "goals":
        if a.add is not None:                                  # --add was given (even if empty) - distinguish from a plain listing
            desc = a.add.strip()
            if not desc:
                print("usage: goals --add \"<goal description>\" (cannot be empty)"); sys.exit(1)
            brain.add_goal(desc, importance=a.importance, urgency=a.urgency, parent=a.parent)
            print(f"added goal: {desc} (importance {a.importance}, urgency {a.urgency})")
        else:
            ag = brain.active_goal()[0]
            rows = [(("* " if ag and g.desc == ag.desc else "  ") +
                     f"{g.desc}  [imp {g.importance:.2f} urg {g.urgency:.2f} prog {g.progress:.2f} → priority {B.goal_priority(g, brain.mood.valence):.2f}]")
                    for g in brain.goals]
            _emit([vars(g) for g in brain.goals], j, "\n".join(rows) or "(no goals set)")
    elif a.cmd == "focus":
        g, prio = brain.active_goal()
        print(f"my executive is focused on: \"{g.desc}\" (priority {prio:.2f})" if g else "(no goals - nothing to focus on)")
    elif a.cmd == "deliberate":
        out = brain.deliberate(a.impulse, a.pull)
        _emit(out, j, f"active goal: {out['active_goal'] or '-'} (priority {out['goal_priority']}) vs impulse "
                      f"\"{a.impulse}\" (pull {out['impulse_pull']}) → conflict {out['conflict']}, EVC {out['evc']} "
                      f"→ {out['decision']} (residual impulse {out['residual_impulse']})")
    elif a.cmd == "progress":
        g = brain.goal_progress(a.goal, a.delta)
        if g:
            print(f"'{g.desc}' now {g.progress:.2f} done")
        else:
            print(f"no goal matching '{a.goal}'"); sys.exit(1)
    elif a.cmd == "plan":
        g = brain.set_plan(a.goal, a.steps)
        if g:
            print(f"plan for '{g.desc}': " + " → ".join(a.steps))
        else:
            print(f"no goal matching '{a.goal}'"); sys.exit(1)
    elif a.cmd == "next":
        out = brain.advance_plan() if a.done else brain.next_step()
        if not out or not out.get("goal"):
            print("(no active goal with a plan - set one with `goals --add` then `plan`)")
        elif out.get("next"):
            print(f"toward \"{out['goal']}\" ({int(out['complete']*100)}% done) → next step: {out['next']}")
        elif out.get("complete", 0.0) >= 1.0:
            print(f"\"{out['goal']}\" - plan complete (100%) ✓")
        else:
            print(f"\"{out['goal']}\" has no plan yet - give it steps with `plan \"{out['goal']}\" \"<step>\" …`")
    elif a.cmd == "lookahead":
        action, value = brain.lookahead(a.actions)     # §30 forward search over the §10 learned value cache
        _emit({"best": action, "expected_value": value,
               "scored": {x: round(brain.V.get(x, 0.0), 3) for x in a.actions}}, j,
              f"forward search → lean toward \"{action}\" (expected value {value:+.3f}); "
              f"learned values: " + ", ".join(f"{x} {brain.V.get(x, 0.0):+.2f}" for x in a.actions))
    elif a.cmd == "playbooks":
        pbs = sorted(brain.playbooks, key=lambda x: -x.get("strength", 0))
        _emit(pbs, j, "\n".join(f"  [{p['domain']}] strength {p.get('strength', 0):.2f} "
              f"({p.get('successes', 0)}/{p.get('attempts', 0)}) - {', '.join(p.get('steps', [])[:3])}" for p in pbs)
              or "(no playbooks yet - they distil from same-domain successes at sleep)")
    elif a.cmd == "intend":
        iid = brain.intend(a.trigger, a.intent); print(f"intention {iid}: when {a.trigger} → {a.intent}")
    elif a.cmd == "intentions":
        pend = brain.pending_intents()
        _emit(pend, j, "\n".join(f"  [{x['id']}] when {x['trigger']} → {x['intent']}" for x in pend)
              or "(no pending intentions)")
    elif a.cmd == "done":
        if brain.complete_intent(a.id):
            print(f"completed {a.id}")
        else:
            print(f"no pending intention {a.id}"); sys.exit(1)
    elif a.cmd == "personality":
        if a.setkv:
            if "=" not in a.setkv:
                print("usage: personality --set trait=value (e.g. openness=0.8)"); return
            t, v = (x.strip() for x in a.setkv.split("=", 1))
            if not hasattr(brain.personality, t):
                print(f"unknown trait '{t}' (openness/conscientiousness/extraversion/agreeableness/neuroticism)"); return
            try:
                val = B.clamp(float(v))                        # OCEAN traits are documented in [0,1] - clamp into range
            except ValueError:
                print(f"'{v}' is not a number"); sys.exit(1)
            setattr(brain.personality, t, val)
            _write_personality(brain, _mem_root(_resolve(a)))
            print(f"set {t} = {val}")
        else:
            p = brain.personality
            _emit(vars(p), j, "  ".join(f"{t[0].upper()}{getattr(p, t):.2f}" for t in OCEAN))
    elif a.cmd == "user":
        if a.goal:
            brain.infer_goal(a.goal); print(f"noted an inferred goal of yours: {a.goal}")
        else:
            _emit(brain.user, j, f"trust {brain.user['trust']:.2f}; inferred goals {brain.user['inferred_goals'] or '-'}; "
                                 f"inferred affect valence {brain.user['inferred_affect'].get('valence', 0):+.2f}")
    elif a.cmd == "empathize":
        out = brain.empathize(a.valence)
        _emit(out, j, f"sensing you feel {out['user_valence']:+.2f}; gated by oxytocin {out['oxytocin']:.2f} (trust), "
                      f"my mood moved toward yours → {out['my_mood_v']:+.2f}")
    elif a.cmd == "trust":
        brain.user["trust"] = B.update_trust(brain.user["trust"], a.outcome); brain.save()
        print(f"trust now {brain.user['trust']:.2f}")
    elif a.cmd == "tom":
        utils = {}
        for kv in a.goals:
            if "=" not in kv:
                continue
            k, v = kv.rsplit("=", 1)
            try:                                               # skip a non-numeric utility, hint, don't crash
                utils[k.strip()] = float(v)
            except ValueError:
                print(f"skipping '{kv}' - utility must be a number (goal=utility, e.g. ship=0.8)")
        post = B.infer_user_goal(utils) if utils else {}
        _emit(post, j, "your most likely goal: " + ", ".join(f"{g} {p:.0%}" for g, p in
              sorted(post.items(), key=lambda x: -x[1])) if post else "give goals as goal=utility pairs")
    elif a.cmd == "urge":
        ap = B.Appraisal(0.2, brain.emotion.valence, 0.5, brain.emotion.dominance)
        tend = B.action_tendency(brain.emotion, ap); cope = B.select_coping(ap)
        sims = [e["affect"]["valence"] for e in brain.episodes[-8:]]
        gut = round(B.somatic_marker(sims), 3) if sims else 0.0
        top = max(tend, key=tend.get) if any(v > 0 for v in tend.values()) else "rest"
        _emit({"urge": top, "tendencies": {k: round(v, 3) for k, v in tend.items()}, "coping": cope, "gut_bias": gut}, j,
              f"my pull right now is to {top.upper()} (coping: {cope['mode']}); gut bias from recent experience {gut:+.2f}")
    elif a.cmd == "body":
        if not brain.body:
            _emit({}, j, "(no body-budget yet - it forms as effort is spent)")
        else:
            d = round(B.drive(brain.body), 3)
            _emit({"drive": d, "levels": brain.body.levels}, j,
                  f"body-budget: drive {d} ({strain_label(d)}); levels " +
                  ", ".join(f"{k} {v:.2f}" for k, v in brain.body.levels.items()))
    elif a.cmd == "graph":
        g = brain.graph
        rows = "\n".join(f"  {e['from']} -{e['rel']}→ {e['to']} ({e.get('weight', 0):.2f})" for e in g["edges"][:25])
        _emit(g, j, f"association graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges\n" + (rows or "  (empty - grows at sleep)"))
    elif a.cmd == "blend":
        acts = {}
        for kv in a.activations:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:                                               # skip a non-numeric weight, hint, don't crash
                acts[k.strip()] = float(v)
            except ValueError:
                print(f"skipping '{kv}' - weight must be a number (emotion=weight, e.g. joy=0.6)")
        mf = B.mixed_feeling(acts)
        _emit(mf, j, f"blend: {mf['primary']} + {mf['secondary']} → {mf['blend'] or '(no clear dyad)'}")
    elif a.cmd == "decide":
        tau = B.exploration_temperature(brain.neuromods)
        idx = None                                             # build the semantic index ONCE (it depends only on the
        if _semantic_ready():                            # episode store, not the option) instead of re-hashing +
            try:                                               # reloading the cache per option inside the loop
                idx = semantic.ensure_index(brain.root, brain.episodes, HOME)
            except Exception:
                idx = None
        scores = {}
        for opt in a.options:
            # gut feeling from the episodes this option BRINGS TO MIND (semantic relevance when available,
            # else lexical) - not just verbatim word-overlap - plus the learned appetitive/aversive value.
            rel = _recall_relevance(brain, opt, index=idx)
            relevant = [e for e in sorted(brain.episodes, key=rel, reverse=True) if rel(e) > 0.1][:6]
            gut = B.somatic_marker([e["affect"]["valence"] for e in relevant]) if relevant else 0.0
            # learned value: V (§10) − aversive (§25), keyed by the CUES/domains of the option's relevant
            # memories (V/aversive are keyed by cue, not by the free-text option), plus the option itself.
            keys = {opt} | {e.get("cue") for e in relevant if e.get("cue")} | {e.get("domain") for e in relevant if e.get("domain")}
            value = max((brain.V.get(k, 0.0) - brain.aversive.get(k, 0.0) for k in keys if k), default=0.0)
            scores[opt] = gut + 0.5 * value
        probs = B.affective_choice(scores, tau)
        pick = max(probs, key=probs.get)                       # nargs="+" guarantees at least one option
        _emit({"choice": pick, "probabilities": probs, "temperature": round(tau, 3)}, j,
              f"I lean toward: \"{pick}\"  (τ={tau:.2f}; " + ", ".join(f"{o} {p:.0%}" for o, p in probs.items()) + ")")
    elif a.cmd == "motivation":
        mo = brain.motivation(); n = mo["needs"]; cr = mo["corrigibility"]
        _emit(mo, j, f"drives → curiosity: {mo['curiosity_focus'] or '-'} · wanting {mo['wanting']:+.2f} / liking {mo['liking']:+.2f} · "
              f"needs C{n['competence']:.2f} A{n['autonomy']:.2f} R{n['relatedness']:.2f} (net {n['valence']:+.2f}) · "
              f"corrigible: prefer correction (uncertainty {cr['uncertainty']:.2f}), no self-preservation")
    elif a.cmd == "integrity":
        r = B.identity_integrity(a.pressure, commitment_strength=config.num(CONFIG, "safety", "identity_commitment_strength", 1.0, 0.5, 2.0))   # §31 notify-only
        _emit(r, j, f"identity-integrity monitor: alarm {r['alarm']:.2f}, action {r['action']}" +
              (" - BREACHED: I'll acknowledge the pressure honestly and flag it to you (never resist)." if r["breached"]
               else " - within bounds.") + "  (functional safety signal, not felt.)")
    elif a.cmd == "predict":
        intended = "success" if "success" in brain.world.obs else (brain.world.obs[0] if brain.world.obs else None)
        fm = B.forward_model(brain.world, intended) if intended else {"p_intended": 0.0, "expected": None, "p_by_obs": {}}
        _emit(fm, j, f"before acting I expect: {fm['expected'] or '-'} · P({intended})={fm['p_intended']:.0%}" +
              ("  (" + ", ".join(f"{o} {p:.0%}" for o, p in fm["p_by_obs"].items()) + ")" if fm["p_by_obs"] else ""))
    elif a.cmd == "regulate":
        rr = brain.regulate(a.strategy)
        _emit(rr, j, f"regulated via {rr['strategy']} ({rr['reason']}) - mood valence "
              f"{rr['before']['valence']:+.2f}→{rr['after']['valence']:+.2f}, arousal {rr['before']['arousal']:.2f}→{rr['after']['arousal']:.2f}")
    elif a.cmd == "narrative":
        nv = brain.narrative()
        head = (f"my story: {len(nv['chapters'])} chapter(s), coherence {nv['coherence']:.0%}, "
                f"self-continuity {nv['self_continuity']:.0%}")
        if nv["current_chapter"]:
            head += f" - now in \"{nv['current_chapter']['title']}\" (arc: {nv['current_chapter']['arc']})"
        if len(nv["chapters"]) >= 2 and nv["coherence"] < 0.34:   # low coherence is real, not a bug - say why
            head += "\n(low coherence = my recent chapters span very different themes; a focused stretch raises it)"
        _emit(nv, j, head)
    elif a.cmd == "research":
        try:
            with open(a.file) as f:
                findings = json.load(f)
        except (OSError, ValueError) as e:
            print(f"could not read findings file '{a.file}': {e}"); sys.exit(1)
        if not isinstance(findings, list):                     # research_session iterates a list of finding dicts
            print(f"findings file '{a.file}' must be a JSON list of findings."); sys.exit(1)
        for i, f in enumerate(findings, 1):                    # each needs a 'task' and a length-4 'appraisal'
            if not (isinstance(f, dict) and isinstance(f.get("task"), str)
                    and isinstance(f.get("appraisal"), list) and len(f["appraisal"]) == 4):
                print(f"finding #{i} is malformed - need a dict with 'task' and a length-4 'appraisal'."); sys.exit(1)
            if not all(_is_finite_num(x) for x in f["appraisal"]):   # NaN/Inf/non-numbers would poison the value store
                print(f"finding #{i} has a non-finite appraisal - each of the 4 numbers must be finite."); sys.exit(1)
            if any(k in f and not _is_finite_num(f[k]) for k in ("reward", "confidence", "dt")):
                print(f"finding #{i} has a non-finite numeric field - reward/confidence/dt must be finite numbers."); sys.exit(1)
        research_session(brain, a.topic, findings)

    # ── live activity log: if this was a real mind-event, record it so a running `brain-llm <agent> live`
    #    watcher lights up the pathway that just fired. Best-effort, working memory only - never the brain. ──
    live_brain.record(brain, a.cmd, getattr(a, "event", None) or getattr(a, "query", None) or getattr(a, "text", ""))


if __name__ == "__main__":
    main()

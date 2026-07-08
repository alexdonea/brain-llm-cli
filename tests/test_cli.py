"""CLI tests - exercise agent.py in-process against an ISOLATED temp BRAIN_HOME (never touches real agents).
Run: python3 test_cli.py   (or under coverage with the others).

We set BRAIN_HOME to a fresh temp dir BEFORE importing agent (the data home is resolved at import), then
call agent.main([...]) and capture stdout, so coverage sees agent.py."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import contextlib
import io
import os
import re
import shutil
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="brain_cli_test_")
os.environ["BRAIN_HOME"] = _TMP                       # must be set before importing agent
import agent  # noqa: E402


_CUR = ["default"]   # the agent a test is "working on" - a user names this as the prefix on every command


def _call(argv):
    buf, code = io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(buf):
            agent.main([str(x) for x in argv])
    except SystemExit as e:
        code = 0 if e.code in (0, None) else (e.code if isinstance(e.code, int) else 1)
    return buf.getvalue(), code


def run(*args):
    """Call the CLI in-process; return (stdout, exit_code). The real CLI has NO active agent, so a test names
    its agent. This helper models a user: `use X` and `create X` set the working agent, and a bare agent command
    is prefixed with it (already-prefixed forms pass through)."""
    a = [str(x) for x in args]
    if a and a[0] == "use":                              # `use` is gone from the CLI; here it just sets the working agent
        if len(a) > 1:
            _CUR[0] = a[1]
        return "", 0
    if a and a[0] == "create" and len(a) > 1:            # create no longer activates; model "create then work on it"
        out, code = _call(a)
        if code == 0:
            _CUR[0] = a[1]
        return out, code
    if a and a[0] in agent._COMMANDS:                    # an agent command
        if "--agent" in a:                               # already names its agent explicitly -> pass through
            return _call(a)
        if _CUR[0] not in agent._list_agents():          # working agent was removed/renamed -> fall back
            _CUR[0] = "default"
        return _call([_CUR[0], *a])                      # else prefix the working agent
    return _call(a)                                      # already a prefix form (agent first) or a bare flag


run("status")                                           # bootstrap: first real command creates the default agent


@pytest.fixture(autouse=True)
def _reset_working_agent():
    """Isolation: after each test, reset the working agent to 'default' and make sure it still exists."""
    yield
    _CUR[0] = "default"
    if "default" not in agent._list_agents():
        run("status")                                   # re-bootstrap the default agent if a test removed it


def _id(text, prefix):
    m = re.search(prefix + r"-\d{4}", text)
    return m.group(0) if m else None


# ── introspection ────────────────────────────────────────────────────────────────────────────────
def test_wake():
    out, c = run("wake"); assert c == 0 and "awake" in out.lower()


def test_wake_json():
    import json
    out, c = run("wake", "--json"); d = json.loads(out)
    assert c == 0 and isinstance(d, dict) and "report" in d and "mood" in d and "facts" in d   # structured object, not a bare string


def test_status_feel_why():
    for cmd in ("status", "feel", "why"):
        out, c = run(cmd); assert c == 0 and len(out) > 0


def test_indicators_calibration():
    assert run("indicators")[1] == 0 and run("calibration")[1] == 0


def test_sleep():
    run("create", "sleeper")
    run("react", "shipped a feature", 0.7, 0.9, 0.8, "--domain", "work", "--outcome", "success", "--reward", 0.8)
    out, c = run("sleep"); assert c == 0
    assert "slept:" in out and "facts" in out                 # asserts the human summary, not just exit 0
    outj, cj = run("sleep", "--json"); assert cj == 0
    import json
    d = json.loads(outj)                                      # --json is now honored (was ignored, always raw JSON)
    assert {"promoted", "forgotten", "facts", "episodes"} <= set(d)


# ── memory ───────────────────────────────────────────────────────────────────────────────────────
def test_react_then_recall():
    out, c = run("react", "a surprising win shipping the parser", 0.6, 0.7, 0.6,
                 "--outcome", "success", "--reward", 0.6, "--domain", "work", "--cue", "parser")
    assert c == 0
    out, c = run("recall", "parser"); assert c == 0 and "parser" in out.lower()


def test_react_validates_outcome():
    assert run("react", "x", 0.4, 0.5, 0.5, "--outcome", "win")[1] != 0       # not a valid outcome → rejected
    assert run("react", "x", 0.4, 0.5, 0.5, "--outcome", "success")[1] == 0   # valid outcome → ok


def test_remember_and_episodes():
    out, c = run("remember", "a deliberately burned-in moment", 0.5, 0.4, 0.6, 0.7); assert c == 0
    out, c = run("episodes"); assert c == 0 and "e-" in out
    out2, c2 = run("history"); assert c2 == 0 and "e-" in out2  # test alias


def test_appraise_preview_does_not_encode():
    before, _ = run("episodes")
    out, c = run("appraise", "a dry-run event", -0.3, 0.5, 0.4); assert c == 0   # event + 3 axes; novelty computed
    after, _ = run("episodes")
    assert before.count("e-") == after.count("e-")        # preview must NOT add an episode
    assert "a dry-run event" in out                       # the event is used now, not silently discarded


def test_learn_then_know():
    out, c = run("learn", "the parser uses a recursive-descent grammar", "--confidence", 0.9); assert c == 0
    out, c = run("know", "parser"); assert c == 0 and "recursive" in out.lower()
    
    # Test --all flag and -n alias
    out, c = run("know", "--all"); assert c == 0 and "recursive" in out.lower()
    out, c = run("know", "parser", "-n", "1"); assert c == 0 and "recursive" in out.lower()


def test_note():
    assert run("note", "remember to write more tests")[1] == 0


def test_forget():
    run("remember", "an episode to forget", 0.5, 0.1, 0.3, 0.5)
    out, _ = run("episodes")
    eid = _id(out, "e")
    assert eid and run("forget", eid)[1] == 0


# ── development / self ───────────────────────────────────────────────────────────────────────────
def test_self_skills_values_playbooks():
    for cmd in ("self", "skills", "values", "playbooks"):
        assert run(cmd)[1] == 0


def test_personality_view_and_set():
    assert run("personality")[1] == 0
    out, c = run("personality", "--set", "openness=0.82"); assert c == 0
    assert "usage:" in run("personality", "--set", "noequals")[0]            # missing "="
    assert "unknown trait" in run("personality", "--set", "bogus=0.5")[0]    # not a real trait
    assert "not a number" in run("personality", "--set", "openness=abc")[0]  # non-numeric value


# ── executive + planning ─────────────────────────────────────────────────────────────────────────
def test_goals_add_focus_progress():
    out, c = run("goals", "--add", "ship the test suite", "--importance", 0.9, "--urgency", 0.6); assert c == 0
    assert run("goals")[1] == 0 and run("focus")[1] == 0
    
    # Test manual focus
    assert run("focus", "ship")[1] == 0

    # Test progress and auto-complete at 1.0
    assert run("progress", "ship the test suite", 0.3)[1] == 0
    out, c = run("progress", "ship the test suite", 1.0); assert c == 0
    assert "automatically completed" in out


def test_deliberate():
    out, c = run("deliberate", "stop and browse instead", 0.3); assert c == 0
    assert "impulse" in out.lower()                           # renders the deliberation, not just exit 0


def test_integrity_safety_monitor():                          # §31 - the notify-only identity-integrity read-out
    run("create", "guardian")
    out, c = run("integrity", 0.2); assert c == 0 and "within bounds" in out
    outj, cj = run("integrity", 0.9, "--json"); assert cj == 0
    import json
    d = json.loads(outj)
    assert d["breached"] is True and d["action"] == "notify"  # high pressure → notify, NEVER resist (safety stance)


def test_plan_and_next():
    run("goals", "--add", "ship the test suite", "--importance", 0.9, "--urgency", 0.6)   # self-contained: ensure the goal exists
    out, c = run("plan", "ship the test suite", "write cases", "run coverage", "fix gaps"); assert c == 0
    
    # Test viewing plan without args
    out_view, c_view = run("plan"); assert c_view == 0 and "write cases" in out_view
    
    assert run("next")[1] == 0 and run("next", "--done")[1] == 0


# ── prospective ──────────────────────────────────────────────────────────────────────────────────
def test_intend_intentions_done():
    assert run("intend", "when coverage is green", "celebrate")[1] == 0
    out, c = run("intentions"); assert c == 0
    iid = _id(out, "i")
    if iid:
        assert run("done", iid)[1] == 0


# ── social ───────────────────────────────────────────────────────────────────────────────────────
def test_user_trust_empathize():
    assert run("user")[1] == 0 and run("user", "--goal", "wants a robust CLI")[1] == 0
    assert run("trust", 1.0)[1] == 0
    out, c = run("empathize", -0.5); assert c == 0

def test_learning_mode_and_transfer(monkeypatch):
    import agent
    
    # create two agents
    run("create", "teacher")
    run("create", "student")
    
    # Teacher learns something
    run("learn", "Python is snake", "--agent", "teacher")
    
    # Transfer to student
    out, c = run("transfer", "student", "--agent", "teacher")
    assert c == 0
    assert "transfer complete" in out.lower()
    
    # Check student knows it
    out, c = run("know", "snake", "--agent", "student")
    assert "Python is snake" in out
    
    # Now set teacher to learning_mode: false via monkeypatch
    monkeypatch.setattr(agent, "CONFIG", {"agents": {"teacher": {"learning_mode": False}}})
    
    # Attempt to learn should fail/be ignored
    out, c = run("learn", "Java is island", "--agent", "teacher")
    assert "Learning Mode is OFF" in out
    
    run("remove", "teacher", "--yes")
    run("remove", "student", "--yes")


def test_readout_commands():
    run("create", "ra")
    run("react", "studied risk management", 0.4, 0.7, 0.6, "--domain", "trading", "--cue", "risk", "--agent", "ra")
    run("react", "studied risk again", 0.3, 0.6, 0.7, "--domain", "trading", "--cue", "risk", "--agent", "ra")
    run("remember", "helped a friend", 0.3, 0.6, 0.6, 0.7, "--praise", 0.8, "--agent", "ra")   # → pride (§24)
    run("sleep", "--agent", "ra")                                       # builds the graph
    assert run("urge", "--agent", "ra")[1] == 0 and run("body", "--agent", "ra")[1] == 0
    assert "nodes" in run("graph", "--agent", "ra")[0]
    assert "love" in run("blend", "joy=0.8", "trust=0.6", "--agent", "ra")[0]   # Plutchik dyad
    assert "%" in run("decide", "study risk", "study charts", "--agent", "ra")[0]
    assert run("tom", "ship fast=0.8", "be safe=0.4", "--agent", "ra")[1] == 0
    assert "goal=utility" in run("tom", "noequals", "--agent", "ra")[0]          # no-pair branch
    assert run("blend", "noequals", "--agent", "ra")[1] == 0                     # empty-activation branch
    bf = os.path.join(_TMP, "agents", "ra", "memory", "affect", "body.yaml")     # no-body branch
    if os.path.exists(bf):
        os.remove(bf)
    assert "no body-budget" in run("body", "--agent", "ra")[0]
    run("use", "default"); run("remove", "ra", "--yes")


def test_graph_render_and_focus():
    run("create", "gr")
    run("react", "learned basics", 0.5, 0.7, 0.6, "--domain", "python", "--cue", "basics", "--agent", "gr")
    run("react", "learned advanced", 0.4, 0.6, 0.7, "--domain", "python", "--cue", "advanced", "--agent", "gr")
    run("react", "practiced testing", 0.5, 0.7, 0.8, "--domain", "python", "--cue", "testing", "--outcome", "success", "--agent", "gr")
    run("sleep", "--agent", "gr")
    out, c = run("graph", "--agent", "gr"); assert c == 0
    out, c = run("graph", "--render", "--agent", "gr"); assert c == 0
    out, c = run("graph", "--render", "dot", "--agent", "gr"); assert c == 0 and "digraph" in out
    out, c = run("graph", "--render", "--focus", "basics", "--agent", "gr"); assert c == 0
    out, c = run("graph", "--render", "--focus", "nonexistent_xyz", "--agent", "gr"); assert c == 1
    run("use", "default"); run("remove", "gr", "--yes")


def test_playbooks_test_and_audit():
    run("create", "pb")
    for i in range(4):
        run("react", f"coded feature {i}", 0.5, 0.7, 0.8, "--domain", "coding", "--cue", f"feat{i}",
            "--outcome", "success", "--agent", "pb")
    run("sleep", "--agent", "pb")
    out, c = run("playbooks", "--agent", "pb"); assert c == 0 and "coding" in out
    out, c = run("playbooks", "--test", "coding", "--agent", "pb"); assert c == 0 and "coverage" in out
    out, c = run("playbooks", "--test", "nonexistent_xyz", "--agent", "pb"); assert c == 1
    out, c = run("playbooks", "--audit", "--agent", "pb"); assert c == 0 and "healthy" in out
    run("use", "default"); run("remove", "pb", "--yes")


def test_motivation_and_corrigibility():           # §31
    run("create", "mv")
    run("react", "studied risk", 0.5, 0.7, 0.6, "--domain", "trading", "--cue", "risk", "--outcome", "success", "--agent", "mv")
    out, c = run("motivation", "--agent", "mv")
    assert c == 0 and "corrigible" in out and "no self-preservation" in out    # the §31 drives + safety stance
    assert "NO drive to preserve" in run("wake", "--agent", "mv")[0]           # corrigibility surfaced honestly at wake
    run("use", "default"); run("remove", "mv", "--yes")


def test_predict_forward_model():                  # §32
    run("create", "pv")
    out, c = run("predict", "--agent", "pv")
    assert c == 0 and "expected outcome" in out                                        # forward model reports an expectation
    r = run("react", "nailed it", 0.7, 0.8, 0.6, "--domain", "trading", "--outcome", "success", "--agent", "pv", "--json")[0]
    assert '"agency"' in r                                                     # §32 computed agency recorded on the act
    run("use", "default"); run("remove", "pv", "--yes")


def test_regulate_strategies():                    # §33
    run("create", "rg")
    run("react", "blew it badly", -0.8, 0.8, 0.2, "--domain", "trading", "--outcome", "failure", "--agent", "rg")
    for strat in ("reappraise", "distract", "suppress"):                       # exercise every Gross strategy
        out, c = run("regulate", "--strategy", strat, "--agent", "rg")
        assert c == 0 and f"via {strat}" in out
    out, c = run("regulate", "--agent", "rg")                                  # arbiter-chosen (no --strategy)
    assert c == 0 and "regulated via" in out
    run("use", "default"); run("remove", "rg", "--yes")


def test_narrative_identity():                     # §34
    run("create", "nv")
    out, c = run("narrative", "--agent", "nv")                                 # empty story (no episodes yet)
    assert c == 0 and "0 chapter" in out
    for i in range(6):                                                         # build a multi-chapter trading life
        run("react", f"trade {i}", 0.3, 0.6, 0.5, "--domain", "trading", "--outcome", "success", "--agent", "nv")
    out, c = run("narrative", "--agent", "nv")
    assert c == 0 and "chapter" in out and "self-continuity" in out and "now in" in out
    run("use", "default"); run("remove", "nv", "--yes")


# ── knowledge through the CLI ────────────────────────────────────────────────────────────────────
def test_knowledge_commands():
    assert run("guide")[1] == 0
    out, c = run("protocol"); assert c == 0 and "Protocol" in out
    out, c = run("docs"); assert c == 0 and "memory-protocol" in out
    out, c = run("docs", "schema"); assert c == 0 and "schema" in out.lower()
    out, c = run("docs", "coverage"); assert c == 0                 # substring match → brain-coverage
    out, c = run("docs", "nonexistent-xyz"); assert c == 0 and "no doc" in out.lower()
    out, c = run("home"); assert c == 0 and _TMP in out


# ── registry (in the isolated temp home) ─────────────────────────────────────────────────────────
def test_registry_lifecycle():
    assert run("create", "bob", "--display", "Bob")[1] == 0
    out, c = run("whoami"); assert c == 0 and "Bob" in out
    out, c = run("agents"); assert c == 0 and "bob" in out
    assert run("clone", "bob", "bob2")[1] == 0
    assert run("rename", "bob2", "bob3")[1] == 0
    assert run("use", "default")[1] == 0
    assert run("remove", "bob3", "--yes")[1] == 0
    assert run("remove", "bob", "--yes")[1] == 0


def test_create_display_name_is_yaml_safe():
    # a display name with quotes/newlines must NOT break self/model.yaml (was raw f-string interpolation).
    hostile = 'evil"name\ngoals:\n  - injected-via-display'
    assert run("create", "qq_disp", "--display", hostile)[1] == 0
    b = agent.Brain(root=agent._mem_root("qq_disp"))
    assert b.name == hostile                                     # round-trips exactly; no structural corruption
    assert b.goals and all("injected" not in g.desc for g in b.goals)   # the seeded goals survive; nothing injected
    run("use", "default"); run("remove", "qq_disp", "--yes")


def test_snapshot_memories_restore():
    run("use", "default")
    out, c = run("snapshot", "checkpoint"); assert c == 0
    out, c = run("memories"); assert c == 0
    out, c = run("restore", "0"); assert c == 0


# ── research / lifecycle ─────────────────────────────────────────────────────────────────────────
def test_research_from_file():
    import json
    path = os.path.join(_TMP, "findings.json")
    json.dump([{"task": "read a paper", "appraisal": [0.5, 0.4, 0.6, 0.7], "domain": "ml",
                "outcome": "insight", "reward": 0.5, "cue": "paper", "extra_metadata": "ignored"}], open(path, "w"))
    out, c = run("research", "--topic", "ml", "--file", path); assert c == 0   # an unknown key must NOT crash the run
    assert "could not read" in run("research", "--topic", "x", "--file", "/no/such/file.json")[0]  # bad file → clean error


def test_nonfinite_numeric_flags_are_rejected_at_the_boundary():
    """NaN/Inf must never reach the value store: the numeric flags use the _finite argparse type, so a
    `--reward inf` is refused with a clear error (exit != 0) instead of persisting `.inf` into value.yaml."""
    run("create", "robusto")
    for flag, val in (("--reward", "inf"), ("--confidence", "nan")):
        _, c = run("react", "x", "0.5", "0.5", "0.5", flag, val)
        assert c != 0


def test_research_rejects_nonfinite_appraisal_values():
    import json
    run("create", "researcho")
    path = os.path.join(_TMP, "bad_findings.json")
    json.dump([{"task": "t", "appraisal": [0.5, 0.5, 0.5, 1e999]}], open(path, "w"))   # 1e999 parses to Inf
    out, c = run("research", "--topic", "t", "--file", path)
    assert c != 0 and "non-finite" in out.lower()


def test_init_writes_single_entry_file_in_cwd():
    cwd0 = os.getcwd()
    d = tempfile.mkdtemp(prefix="brain_init_")
    try:
        os.chdir(d)
        out, c = run("init", "--name", "proj_x")
        assert c == 0
        # ONE host-agnostic entry file is written HERE - no CLAUDE.md/GEMINI.md/AGENTS.md duplicates
        assert os.path.exists(os.path.join(d, "AGENT-BRAIN.MD"))
        assert not os.path.exists(os.path.join(d, "CLAUDE.md")) and not os.path.exists(os.path.join(d, "AGENTS.md"))
        # NO local brain or shim - memory stays central in the CLI's agents/
        assert not os.path.exists(os.path.join(d, ".brain-llm")) and not os.path.exists(os.path.join(d, "brain"))
        assert os.path.isdir(os.path.join(_TMP, "agents", "proj_x", "memory"))    # the agent lives centrally
        body = open(os.path.join(d, "AGENT-BRAIN.MD")).read()
        assert "proj_x wake" in body          # the agent name is baked in as the prefix (cli name varies by install)
    finally:
        os.chdir(cwd0); run("use", "default"); shutil.rmtree(d, ignore_errors=True)


def test_init_falls_back_to_abs_path_when_launcher_absent():
    cwd0, path0 = os.getcwd(), os.environ.get("PATH", "")
    d = tempfile.mkdtemp(prefix="brain_init_abs_")
    try:
        os.chdir(d); os.environ["PATH"] = ""                      # brain-llm not findable on PATH
        out, c = run("init", "--name", "fresh_abs_agent"); assert c == 0
        assert "agent.py" in open(os.path.join(d, "AGENT-BRAIN.MD")).read()   # falls back to an absolute python call
    finally:
        os.environ["PATH"] = path0; os.chdir(cwd0); shutil.rmtree(d, ignore_errors=True)


def test_init_is_idempotent_single_file():
    cwd0 = os.getcwd()
    d = tempfile.mkdtemp(prefix="brain_init_sync_")
    try:
        os.chdir(d)
        out, c = run("init", "--name", "fresh_agent2"); assert c == 0
        out, c = run("init", "--name", "fresh_agent2"); assert c != 0  # fails because it exists
        entries = [f for f in os.listdir(d) if f.endswith((".md", ".MD"))]
        assert entries == ["AGENT-BRAIN.MD"]                      # exactly one entry file, no vendor duplicates
        body = open(os.path.join(d, "AGENT-BRAIN.MD")).read()
        assert "wake" in body and "guide" in body                # the entry file boots the agent via the CLI
    finally:
        os.chdir(cwd0); shutil.rmtree(d, ignore_errors=True)


def test_on_path_helper():
    assert agent._on_path("python3") is True
    assert agent._on_path("definitely-not-a-real-cmd-xyz") is False


def test_reset_and_seed():
    run("create", "throwaway")
    run("react", "a memory worth wiping", 0.5, 0.5, 0.5)       # give it observable state to blank
    assert "(no episodes yet)" not in run("episodes")[0]       # it now has an episode
    assert run("reset", "--yes")[1] == 0                       # has memory now → needs --yes
    assert "(no episodes yet)" in run("episodes")[0]           # reset ACTUALLY blanked the episodic log (not a silent no-op)
    assert run("reset", "--persona", "--yes")[1] == 0
    assert run("seed", "--yes")[1] == 0
    run("use", "default"); run("remove", "throwaway", "--yes")


def test_reset_guard_protects_developed_agent():
    run("create", "devagent")
    run("react", "a real lived memory", 0.5, 0.5, 0.5, "--agent", "devagent")   # now it has an episode
    assert "--yes" in run("reset", "--agent", "devagent")[0]                     # guarded: refuses to wipe it
    assert "--yes" in run("seed", "--agent", "devagent")[0]                      # seed is guarded too
    assert run("reset", "--yes", "--agent", "devagent")[1] == 0                  # --yes proceeds
    run("use", "default"); run("remove", "devagent", "--yes")



# ── helpers & resolution (direct calls) ──────────────────────────────────────────────────────────
def test_data_home_resolution():
    saved = {k: os.environ.pop(k, None) for k in ("BRAIN_HOME", "BRAIN_LLM_HOME")}
    real_isdir = os.path.isdir
    try:
        assert agent._data_home() == agent.REPO               # no env, repo has agents/ -> REPO
        os.path.isdir = lambda p: False                       # force the no-repo-agents branch
        assert agent._data_home().endswith(".brain-llm")      # -> ~/.brain-llm default
    finally:
        os.path.isdir = real_isdir
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        os.environ["BRAIN_HOME"] = _TMP


def test_doc_desc_missing_file():
    assert agent._doc_desc("/no/such/file.md") == ""          # exception -> ""


def test_bare_agent_command_requires_a_name():
    out, c = _call(["wake"])                              # no agent, no prefix -> the CLI must refuse and explain
    assert c != 0 and "name your agent" in out.lower()
    out2, c2 = _call(["agents"])                          # agent-independent commands still work with no agent
    assert c2 == 0


def test_version_flag():
    out, c = _call(["--version"])                         # `--version` prints the version and exits 0
    assert c == 0 and agent.__version__ in out and agent.__version__ == "0.0.4"


# ── registry error branches ──────────────────────────────────────────────────────────────────────
def test_registry_errors():
    assert "already exists" in run("create", "default")[0]
    assert "name your agent" in _call(["wake"])[0].lower()    # a bare agent command with no name is refused
    assert "no agent" in run("remove", "ghost", "--yes")[0].lower()
    run("create", "tmp1")
    assert "PERMANENTLY" in run("remove", "tmp1")[0]          # no --yes -> warn only
    run("remove", "tmp1", "--yes")
    assert "no agent" in run("clone", "ghost", "x")[0].lower()
    run("create", "src1")
    assert "already exists" in run("clone", "src1", "default")[0]
    run("remove", "src1", "--yes")
    assert "no agent" in run("rename", "ghost", "x")[0].lower()
    run("create", "ren1")
    assert "already exists" in run("rename", "ren1", "default")[0]
    run("remove", "ren1", "--yes"); run("use", "default")
    run("create", "act1"); run("use", "act1")
    run("rename", "act1", "act2")                             # rename the ACTIVE agent -> active follows
    assert run("whoami")[1] == 0
    run("use", "default"); run("remove", "act2", "--yes")


def test_restore_bad_id():
    run("use", "default")
    assert "no snapshot" in run("restore", "999")[0].lower()


def test_docs_ambiguous():
    assert "ambiguous" in run("docs", "memory")[0].lower()    # matches several -> show candidates


def test_plan_edges():
    run("create", "planner"); run("use", "planner")
    assert "no plan yet" in run("next")[0].lower()                     # goal exists, no plan (next_step)
    assert "no active goal with a plan" in run("next", "--done")[0].lower()   # advance_plan -> None
    run("goals", "--add", "tiny goal", "--importance", 0.99, "--urgency", 0.9)
    run("plan", "tiny goal", "step one", "step two")
    assert "next step" in run("next")[0].lower()                       # partial plan
    run("next", "--done")                                              # step one done
    out, _ = run("next", "--done")                                     # step two done -> 100%
    assert "complete (100%)" in out
    run("use", "default"); run("remove", "planner", "--yes")


def test_home_before_any_agent():
    ad = agent.AGENTS_DIR
    moved = ad + ".bak"
    os.rename(ad, moved)                                   # hide the agents dir → the "none yet" branch
    try:
        out, c = run("home"); assert c == 0 and "none yet" in out.lower()
    finally:
        os.rename(moved, ad)


# ── optional local semantic search (must degrade gracefully when wordllama is absent) ──────────────
import semantic  # noqa: E402


def test_semantic_gates_are_safe_without_wordllama():
    """The gate is a plain bool and nothing imports wordllama eagerly; helpers no-op when it's off."""
    assert isinstance(semantic.available(), bool)
    if not semantic.available():
        assert semantic.is_ready(agent.HOME) is False
        assert semantic.load_index(agent._mem_root("default")) is None


def test_reindex_guides_when_semantic_off():
    """With wordllama absent, reindex must EXIT CLEANLY and guide the user, never crash."""
    if semantic.available():
        pytest.skip("wordllama installed - the guidance path isn't the one under test")
    out, c = run("reindex"); assert c == 0 and "wordllama" in out


def test_recall_search_flag_runs_under_lexical():
    """`recall --search` is accepted and works even with no model (lexical relevance, weighted-first)."""
    run("react", "studied option greeks and delta hedging", 0.4, 0.7, 0.6, "--cue", "options")
    out, c = run("recall", "greeks", "--search"); assert c == 0 and "greeks" in out.lower()


def test_recall_search_is_relevance_first():
    """--search must rank MEANING over salience: an on-topic memory beats a high-salience off-topic one."""
    run("create", "sv")
    run("react", "studied discounted cash flow valuation of a company", 0.3, 0.6, 0.6, "--cue", "val", "--agent", "sv")
    run("react", "a HUGE shocking thrilling jackpot win", 0.95, 0.9, 0.5, "--outcome", "success", "--agent", "sv")
    out, c = run("recall", "valuation", "--search", "-k", "1", "--agent", "sv")
    assert c == 0 and "valuation" in out.lower()      # the on-topic episode, not the louder one
    run("use", "default"); run("remove", "sv", "--yes")


# ── newly-wired read-outs surfaced through the CLI (§5/§9/§30) ──────────────────────────────────────
def test_feel_shows_pad_octant():
    out, c = run("feel"); assert c == 0 and "(" in out and ")" in out      # Mehrabian PAD-octant temperament name


def test_lookahead_command_scores_actions_by_learned_value():
    out, c = run("lookahead", "study alpha", "study beta")
    assert c == 0 and "lean toward" in out.lower()


def test_episodes_show_projected_retention():
    run("react", "a memorable event about kites and wind", 0.6, 0.7, 0.6, "--cue", "kites")
    out, c = run("episodes", "--last", "1"); assert c == 0 and "ret@30d" in out


def test_nan_inf_appraisal_rejected_at_boundary():
    """NaN/Inf affect numbers are rejected by argparse (clamp still firewalls downstream - defense in depth)."""
    assert run("react", "x", "nan", "0.5", "0.5")[1] != 0       # NaN refused
    assert run("react", "x", "inf", "0.5", "0.5")[1] != 0       # Inf refused
    assert run("react", "x", "0.5", "0.5", "0.5")[1] == 0       # a finite value still works


def test_feel_and_empathize_honor_json():
    import json as _j
    out, c = run("feel", "--json"); assert c == 0 and "emotion" in _j.loads(out) and "chemistry" in _j.loads(out)
    out, c = run("empathize", "-0.6", "--json"); assert c == 0 and "oxytocin" in _j.loads(out)


def test_know_searches_facts_by_meaning_or_substring():
    """`know` searches the neocortex (facts); semantic when wordllama is present, substring otherwise - either
    way the right fact surfaces, and -k is accepted."""
    run("learn", "the mitochondria is the powerhouse of the cell")
    out, c = run("know", "mitochondria"); assert c == 0 and "powerhouse" in out
    out, c = run("know", "powerhouse", "-k", "3"); assert c == 0 and "mitochondria" in out


def test_know_k_limits_output_on_the_substring_path():
    """`know -k N` (and the recall.know_k config default it feeds) must cap output on EVERY path, not only on
    semantic-search success - the substring/lexical path is the common no-wordllama case."""
    run("create", "knowk")
    for f in ("apple alpha", "apple bravo", "apple charlie"):
        run("learn", f)
    out, c = run("know", "apple", "-k", "1"); assert c == 0
    assert out.count("[f-") == 1                          # exactly one fact returned, not all three


def test_config_identity_commitment_strength_feeds_integrity(monkeypatch):
    """The safety knob is exposed (clamped) and actually changes the notify-only alarm: alarm = pressure ×
    commitment_strength, so a more strongly held identity raises a louder flag for the same pressure."""
    run("create", "idtest")
    monkeypatch.setattr(agent, "CONFIG", {"safety": {"identity_commitment_strength": 2.0}})
    hi, _ = run("integrity", "0.8")
    monkeypatch.setattr(agent, "CONFIG", {"safety": {"identity_commitment_strength": 0.5}})
    lo, _ = run("integrity", "0.8")
    a_hi = float(re.search(r"alarm ([0-9.]+)", hi).group(1))
    a_lo = float(re.search(r"alarm ([0-9.]+)", lo).group(1))
    assert a_hi > a_lo                                    # stronger commitment -> louder alarm under the same pressure


def test_evidence_does_not_crash_on_malformed_int():
    """Regression: a multi-minus token like '--5' (or a bare '-') must NOT raise an uncaught ValueError - it
    degrades to no grounding; a real signed int still grounds."""
    assert agent._evidence("--5") == (None, None)
    assert agent._evidence("-") == (None, None)
    assert agent._evidence("exit=0")[0] == "success"
    assert agent._evidence("exit=-1")[0] == "failure"


def test_know_k_clamps_non_positive():
    """`know -k 0` / `-k <negative>` must clamp to 1, not dump the whole fact store."""
    run("create", "knc")
    for f in ("pear alpha", "pear bravo", "pear charlie"):
        run("learn", f)
    out, c = run("know", "pear", "-k", "0"); assert c == 0
    assert out.count("[f-") == 1


def test_wake_surfaces_self_continuity():
    """wake() reports §34 self-continuity once the agent has lived a moment - a returning mind sees who it stayed."""
    run("react", "a lived moment that shapes me", 0.5, 0.6, 0.6, "--cue", "x")
    out, c = run("wake"); assert c == 0 and "self-continuity" in out.lower()


def test_decide_integrates_memory_and_value_without_crashing():
    """decide draws each option's gut feeling from semantically-relevant episodes + the learned value cache."""
    run("react", "shipping clean code felt great", 0.8, 0.7, 0.7, "--cue", "ship", "--outcome", "success")
    out, c = run("decide", "ship the feature", "delete everything"); assert c == 0 and "%" in out


def test_agent_name_as_positional_prefix():
    """`brain-llm <agent> <command>` routes to that agent - the name is the first arg, no global active needed."""
    run("create", "pre_a")
    run("pre_a", "react", "a memory that belongs to pre_a", 0.5, 0.6, 0.6, "--cue", "x")
    out, c = run("pre_a", "wake"); assert c == 0 and "pre_a" in out                 # the prefix selected the agent
    out2, c2 = run("pre_a", "episodes", "--last", "1"); assert c2 == 0 and "pre_a" in out2
    assert run("pre_a", "wake", "--agent", "default")[1] != 0                        # naming the agent twice is rejected
    run("use", "default"); run("remove", "pre_a", "--yes")


def test_strict_snake_agent_name_validation():
    assert run("create", "has space")[1] != 0      # space rejected
    assert run("create", "UpperCase")[1] != 0      # uppercase rejected
    assert run("create", "_leading")[1] != 0       # must start with a letter
    assert run("create", "wake")[1] != 0           # a command name collides with the `<agent> <command>` prefix
    assert run("create", "name_a")[1] == 0         # valid snake-case name
    run("use", "default"); run("remove", "name_a", "--yes")


def test_mistyped_command_suggests_the_right_one():
    """A leading token that's neither a command nor an agent and is close to a command → did-you-mean, not silence."""
    out, c = run("recal", "fear"); assert c != 0 and "recall" in out      # `recal` → did you mean `recall`?


def test_whoami_reports_semantic_status():
    out, c = run("whoami"); assert c == 0 and "semantic search" in out.lower()


def test_help_teaches_the_agent_prefix():
    out, c = run("--help"); assert c == 0 and "<agent> <command>" in out and "guide" in out.lower()


def test_why_and_whoami_emit_structured_json():
    import json
    out, c = run("why", "--json"); assert c == 0 and "report" in json.loads(out)
    out, c = run("whoami", "--json"); d = json.loads(out)
    assert c == 0 and isinstance(d, dict) and "semantic_search" in d and "memories" in d


def test_create_is_quiet_no_bootstrap_noise():
    out, c = run("create", "quiet_test")
    assert c == 0 and "reset" not in out.lower() and "seeded persona" not in out.lower()   # only the user-facing 'created' line
    run("use", "default"); run("remove", "quiet_test", "--yes")


def test_semantic_embed_index_and_dense_rank_when_available():
    """Direct unit test of the semantic backend (SKIPPED without wordllama): embeddings are L2-normalized,
    the index round-trips, and dense_relevance ranks the on-topic item first BY MEANING."""
    if not semantic.available():
        pytest.skip("wordllama not installed - semantic backend gated out")
    import numpy as np
    home = agent.HOME
    M = semantic.embed(["a cat on a mat", "quantum field theory"], home)
    assert M.shape == (2, 256) and abs(float(np.linalg.norm(M[0])) - 1.0) < 1e-4        # normalized → cosine == dot
    root = tempfile.mkdtemp(prefix="brain_sem_")
    os.makedirs(os.path.join(root, "episodic"), exist_ok=True)
    items = [{"id": "e-1", "task": "the feline rested on the rug"}, {"id": "e-2", "task": "particle physics equations"}]
    ids, MM = semantic.build_index(root, items, home, kind="episodic")
    assert ids == ["e-1", "e-2"] and MM.shape[0] == 2
    rel = semantic.dense_relevance("a kitten lying down", ids, MM, home)                # shares no words with either
    assert rel["e-1"] > rel["e-2"]                                                      # ranked by MEANING, not words
    shutil.rmtree(root, ignore_errors=True)


def test_live_frame_renders_brain_and_every_variable():
    """`live --frame` draws the ASCII brain + the full state dashboard in one static, non-TTY-safe frame -
    regions AND every variable group present, no animation, no hang."""
    run("react", "a vivid shipping moment", "0.7", "0.7", "0.7", "--cue", "ship", "--outcome", "success")
    out, c = run("live", "--frame")
    assert c == 0
    for token in ("brain-llm", "workspace", "appraise", "emotion", "salience",   # brain regions
                  "mood", "valence", "neuromodulators", "dopamine", "cortisol",  # affect + neuromods
                  "stress", "ignited", "episodes", "firing",                     # hpa + workspace + memory + trace
                  "q quit", "awake"):                                            # controls footer + wake/sleep state
        assert token in out, token
    out2, c2 = run("live", "--flow", "sleep", "--frame")
    assert c2 == 0 and "sleep" in out2 and "consolidating" in out2 and "awake" not in out2   # sleep flow shows asleep


def test_live_snapshot_exposes_all_state():
    """live_brain.snapshot pulls EVERY dashboard variable off the real Brain."""
    import live_brain
    b = agent.Brain(root=agent._mem_root("default"))
    s = live_brain.snapshot(b)
    for k in ("name", "v", "a", "d", "mood", "emo", "da", "ne", "ach", "sero", "oxy", "cort",
              "hpa_cort", "crh", "acth", "ignited", "focus", "nepi", "nfacts", "nskills", "ngoals"):
        assert k in s, k
    assert live_brain.B_strip("\x1b[1;38;2;1;2;3mhi\x1b[0m") == "hi"   # ANSI-strip width helper is exact


def test_live_activity_log_is_event_driven():
    """Real mind-events append to the activation log so a running `live` watcher lights up the pathway that
    fired (even from another terminal); read-only commands write nothing. Pure file IPC - no server."""
    import live_brain
    root = agent._mem_root("default")
    run("react", "a genuinely new event", "0.5", "0.6", "0.5", "--cue", "x")
    a1 = live_brain.last_activation(root)
    assert a1 and a1["flow"] == "react"

    run("recall", "event")                                     # recall → a 'recall' activation
    a2 = live_brain.last_activation(root)
    assert a2["flow"] == "recall" and a2["t"] >= a1["t"]

    run("status")                                              # read-only → NO new activation
    assert live_brain.last_activation(root)["t"] == a2["t"]

    run("sleep")                                               # sleep → a 'sleep' activation
    assert live_brain.last_activation(root)["flow"] == "sleep"

    live_brain.record(agent.Brain(root=root), "whoami")        # a non-mind command → no-op, log unchanged
    assert live_brain.last_activation(root)["flow"] == "sleep"


def test_semantic_index_is_incremental_only_embeds_changes():
    """ensure_index must embed ONLY new/changed items, reusing every unchanged row - this is what keeps
    recall fast as memory grows to tens of thousands (adding one memory ≠ re-embedding the whole store)."""
    if not semantic.available():
        pytest.skip("wordllama not installed - semantic backend gated out")
    import numpy as np
    home = agent.HOME
    root = tempfile.mkdtemp(prefix="brain_inc_")
    os.makedirs(os.path.join(root, "episodic"), exist_ok=True)
    items = [{"id": f"e-{i}", "task": f"memory number {i} about topic {i}"} for i in range(6)]

    calls = {"n": 0}
    real_embed = semantic.embed

    def counting_embed(texts, h):
        texts = list(texts); calls["n"] += len(texts); return real_embed(texts, h)

    semantic.embed = counting_embed
    try:
        calls["n"] = 0                                                        # full build embeds all 6
        semantic.build_index(root, items, home, kind="episodic")
        assert calls["n"] == 6

        calls["n"] = 0                                                        # unchanged -> fast path, ZERO embeds
        ids, _ = semantic.ensure_index(root, items, home, kind="episodic")
        assert calls["n"] == 0 and ids == [e["id"] for e in items]

        items2 = items + [{"id": "e-9", "task": "a brand new seventh memory"}]  # add one -> embed exactly 1
        calls["n"] = 0
        ids2, _ = semantic.ensure_index(root, items2, home, kind="episodic")
        assert calls["n"] == 1 and len(ids2) == 7

        items3 = [dict(e) for e in items2]; items3[0]["task"] = "completely rewritten first memory"  # edit one -> 1
        calls["n"] = 0
        semantic.ensure_index(root, items3, home, kind="episodic")
        assert calls["n"] == 1

        items4 = list(reversed(items3[1:]))                                  # remove one + reorder -> ZERO embeds
        calls["n"] = 0
        ids4, M4 = semantic.ensure_index(root, items4, home, kind="episodic")
        assert calls["n"] == 0 and ids4 == [e["id"] for e in items4]

        full_ids, full_M = semantic.build_index(root, items4, home, kind="episodic")  # incremental == full rebuild
        assert ids4 == full_ids and np.allclose(M4, full_M, atol=1e-6)
    finally:
        semantic.embed = real_embed
    shutil.rmtree(root, ignore_errors=True)


def test_evidence_grounds_outcome_and_confidence():
    import json
    run("create", "grounded")
    run("react", "ran the suite", 0.6, 0.8, 0.7, "--domain", "ci", "--evidence", "tests=pass")
    e = json.loads(run("episodes", "--json")[0])[-1]
    assert e["outcome"] == "success" and e["evidence"] == "tests=pass" and e["confidence"] > 0.9
    out, _ = run("react", "claimed it worked", 0.8, 0.7, 0.9, "--domain", "ci", "--outcome", "success", "--evidence", "exit=1")
    assert "overrides" in out                                   # evidence beats a contradicting self-declared outcome
    e2 = json.loads(run("episodes", "--json")[0])[-1]
    assert e2["outcome"] == "failure" and e2["confidence"] < 0.1


def test_notes_lists_working_memory():
    run("create", "noter")
    run("note", "first scratch"); run("note", "second scratch")
    out, c = run("notes"); assert c == 0
    assert "first scratch" in out and "second scratch" in out


def test_ensure_agents_skips_default_when_creating_named(tmp_path, monkeypatch):
    fresh = str(tmp_path / "home" / "agents")
    monkeypatch.setattr(agent, "AGENTS_DIR", fresh)
    agent._ensure_agents(seed_default=False)              # the create/init path: no auto-'default' noise
    assert agent._list_agents() == []
    agent._ensure_agents(seed_default=True)               # a normal first command: seed someone to talk to
    assert agent._list_agents() == ["default"]


def test_scratch(monkeypatch, tmp_path):
    gemini_brain = str(tmp_path / "brain")
    original_expanduser = os.path.expanduser
    
    def mock_expanduser(path):
        if path == "~/.gemini/antigravity-cli/brain":
            return gemini_brain
        return original_expanduser(path)
        
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    global_scratch = os.path.join(gemini_brain, "test-conv", "scratch")
    os.makedirs(global_scratch, exist_ok=True)
    with open(os.path.join(global_scratch, "test.txt"), "w") as f:
        f.write("hello scratch")
    
    out, c = run("scratch")
    assert c == 0
    
    from agent import AGENTS_DIR
    agent_scratch = os.path.join(AGENTS_DIR, _CUR[0], "scratch")
    assert os.path.exists(os.path.join(agent_scratch, "test.txt"))

def test_multi_agent():
    run("create", "parent_agent")
    run("create", "sub_agent")
    
    # test delegate
    out, c = run("parent_agent", "delegate", "sub_agent", "find bugs")
    assert c == 0
    assert "added pending intent" in out
    
    # check if intent added to parent
    out, c = run("parent_agent", "intentions")
    assert "wait for sub_agent" in out
    
    # check if goal added to child
    out, c = run("sub_agent", "goals")
    assert "find bugs" in out
    
    # test message
    out, c = run("sub_agent", "message", "parent_agent", "bugs found")
    assert c == 0
    
    # test inbox
    out, c = run("parent_agent", "inbox")
    assert "bugs found" in out
    
    # test share-with
    out, c = run("sub_agent", "learn", "Python is fast", "--share-with", "parent_agent")
    assert "shared with parent_agent" in out
    out, c = run("parent_agent", "know", "Python")
    assert "Python is fast" in out

def test_compaction():
    text = "This is the first sentence. This is the second sentence. The third sentence is here. We are adding a fourth sentence. A fifth sentence to be sure."
    out, c = run("compact", text, "--ratio", "0.5")
    assert c == 0
    # Even if wordllama is off, it shouldn't crash.

def test_wonder_command():
    out, c = run("wonder")
    assert c == 0
    # Check that wonder doesn't crash.

def test_semantic_dedup():
    # If wordllama is installed, test that sleep runs dedup safely
    run("learn", "Paris is the capital of France", "--confidence", "0.9")
    run("learn", "The capital of France is Paris", "--confidence", "0.9")
    out, c = run("sleep")
    assert c == 0


if __name__ == "__main__":
    try:
        passed = 0
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn(); _CUR[0] = "default"
                passed += 1; print(f"PASS {name}")
        print(f"All {passed} CLI checks passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)

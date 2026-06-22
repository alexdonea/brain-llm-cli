"""CLI tests — exercise agent.py in-process against an ISOLATED temp BRAIN_HOME (never touches real agents).
Run: python3 test_cli.py   (or under coverage with the others).

We set BRAIN_HOME to a fresh temp dir BEFORE importing agent (the data home is resolved at import), then
call agent.main([...]) and capture stdout, so coverage sees agent.py. External-service commands (telegram,
market) are not exercised here — they need a live token/network and are verified live."""
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


def run(*args):
    """Call the CLI in-process; return (stdout, exit_code). exit_code 0 unless the command sys.exit()s."""
    buf, code = io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(buf):
            agent.main([str(x) for x in args])
    except SystemExit as e:
        code = 0 if e.code in (0, None) else (e.code if isinstance(e.code, int) else 1)
    return buf.getvalue(), code


run("status")                                         # bootstrap: first real command creates the default agent


@pytest.fixture(autouse=True)
def _reset_active_agent():
    """Enforce isolation: whatever a test switches/creates/renames, restore 'default' as the active agent
    afterwards so order-independence holds (tests share one BRAIN_HOME + global active pointer)."""
    yield
    agent._set_active("default")


def _id(text, prefix):
    m = re.search(prefix + r"-\d{4}", text)
    return m.group(0) if m else None


# ── introspection ────────────────────────────────────────────────────────────────────────────────
def test_wake():
    out, c = run("wake"); assert c == 0 and "awake" in out.lower()


def test_wake_json():
    import json
    out, c = run("wake", "--json"); assert c == 0 and isinstance(json.loads(out), str)   # wake returns a string


def test_status_feel_why():
    for cmd in ("status", "feel", "why"):
        out, c = run(cmd); assert c == 0 and len(out) > 0


def test_indicators_calibration():
    assert run("indicators")[1] == 0 and run("calibration")[1] == 0


def test_sleep():
    out, c = run("sleep"); assert c == 0


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


def test_appraise_preview_does_not_encode():
    before, _ = run("episodes")
    out, c = run("appraise", "a dry-run event", 0.5, -0.3, 0.5, 0.4); assert c == 0
    after, _ = run("episodes")
    assert before.count("e-") == after.count("e-")        # preview must NOT add an episode


def test_learn_then_know():
    out, c = run("learn", "the parser uses a recursive-descent grammar", "--confidence", 0.9); assert c == 0
    out, c = run("know", "parser"); assert c == 0 and "recursive" in out.lower()


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
    assert run("progress", "ship the test suite", 0.3)[1] == 0


def test_deliberate():
    out, c = run("deliberate", "stop and browse instead", 0.3); assert c == 0


def test_plan_and_next():
    run("goals", "--add", "ship the test suite", "--importance", 0.9, "--urgency", 0.6)   # self-contained: ensure the goal exists
    out, c = run("plan", "ship the test suite", "write cases", "run coverage", "fix gaps"); assert c == 0
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
    assert c == 0 and "I expect" in out                                        # forward model reports an expectation
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


def test_init_writes_host_file_in_cwd():
    cwd0 = os.getcwd()
    d = tempfile.mkdtemp(prefix="brain_init_")
    try:
        os.chdir(d)
        out, c = run("init", "--name", "Proj", "--host", "all")   # --host all also covers the nested .github write
        assert c == 0
        # the host entry file(s) are written HERE, in the folder
        assert os.path.exists(os.path.join(d, "CLAUDE.md")) and os.path.exists(os.path.join(d, "AGENTS.md"))
        assert os.path.exists(os.path.join(d, ".github", "copilot-instructions.md"))
        # NO local brain or shim — memory stays central in the CLI's agents/
        assert not os.path.exists(os.path.join(d, ".brain-lmm")) and not os.path.exists(os.path.join(d, "brain"))
        assert os.path.isdir(os.path.join(_TMP, "agents", "Proj", "memory"))      # the agent lives centrally
        body = open(os.path.join(d, "CLAUDE.md")).read()
        assert "brain-lmm use Proj" in body and "brain-lmm wake" in body          # global launcher + agent select
    finally:
        os.chdir(cwd0); run("use", "default"); shutil.rmtree(d, ignore_errors=True)


def test_init_falls_back_to_abs_path_when_launcher_absent():
    cwd0, path0 = os.getcwd(), os.environ.get("PATH", "")
    d = tempfile.mkdtemp(prefix="brain_init_abs_")
    try:
        os.chdir(d); os.environ["PATH"] = ""                      # brain-lmm not findable on PATH
        out, c = run("init"); assert c == 0                       # no --name → "(active agent: …)" print branch
        assert "agent.py" in open(os.path.join(d, "CLAUDE.md")).read()   # falls back to an absolute python call
    finally:
        os.environ["PATH"] = path0; os.chdir(cwd0); shutil.rmtree(d, ignore_errors=True)


def test_init_refreshes_existing_host_files():
    cwd0 = os.getcwd()
    d = tempfile.mkdtemp(prefix="brain_init_sync_")
    try:
        os.chdir(d)
        run("init", "--host", "claude")                          # writes CLAUDE.md (none existed yet)
        run("init", "--host", "gemini")                          # writes GEMINI.md AND refreshes the existing CLAUDE.md
        for f in ("CLAUDE.md", "GEMINI.md"):
            assert os.path.exists(os.path.join(d, f)) and "brain-lmm wake" in open(os.path.join(d, f)).read()
    finally:
        os.chdir(cwd0); shutil.rmtree(d, ignore_errors=True)


def test_on_path_helper():
    assert agent._on_path("python3") is True
    assert agent._on_path("definitely-not-a-real-cmd-xyz") is False


def test_reset_and_seed():
    run("create", "throwaway")                                # fresh agent → reset/seed need no --yes
    assert run("reset")[1] == 0
    assert run("reset", "--persona")[1] == 0
    assert run("seed")[1] == 0
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
    saved = {k: os.environ.pop(k, None) for k in ("BRAIN_HOME", "BRAIN_LMM_HOME")}
    real_isdir = os.path.isdir
    try:
        assert agent._data_home() == agent.REPO               # no env, repo has agents/ -> REPO
        os.path.isdir = lambda p: False                       # force the no-repo-agents branch
        assert agent._data_home().endswith(".brain-lmm")      # -> ~/.brain-lmm default
    finally:
        os.path.isdir = real_isdir
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        os.environ["BRAIN_HOME"] = _TMP


def test_doc_desc_missing_file():
    assert agent._doc_desc("/no/such/file.md") == ""          # exception -> ""


def test_active_fallback_when_no_pointer():
    af = agent.ACTIVE_FILE
    backup = open(af).read() if os.path.exists(af) else None
    try:
        if os.path.exists(af):
            os.remove(af)
        assert agent._active() in (agent._list_agents() or ["default"])
    finally:
        if backup is not None:
            open(af, "w").write(backup)


# ── registry error branches ──────────────────────────────────────────────────────────────────────
def test_registry_errors():
    run("use", "default")
    assert "already exists" in run("create", "default")[0]
    assert "no agent" in run("use", "ghost")[0].lower()
    assert "no agent" in run("remove", "ghost", "--yes")[0].lower()
    run("create", "tmp1")
    assert "PERMANENTLY" in run("remove", "tmp1")[0]          # no --yes -> warn only
    run("use", "tmp1")
    run("remove", "tmp1", "--yes")                            # remove the ACTIVE agent -> reassign active
    run("use", "default")
    assert "no agent" in run("clone", "ghost", "x")[0].lower()
    run("create", "src1")
    assert "already exists" in run("clone", "src1", "default")[0]
    run("remove", "src1", "--yes"); run("use", "default")
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


if __name__ == "__main__":
    try:
        passed = 0
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn(); agent._set_active("default")              # mirror the autouse fixture's isolation
                passed += 1; print(f"PASS {name}")
        print(f"All {passed} CLI checks passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)

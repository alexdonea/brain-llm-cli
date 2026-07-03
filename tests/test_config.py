"""Tests for the optional config.yaml layer (src/config.py + its wiring in agent.py).

Two halves: (1) the loader/accessors degrade safely on every bad input, and (2) a present config
actually changes the argparse defaults build_parser() produces, while a CLI flag still overrides.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import agent
import config


def _fresh():
    config._CACHE = {}                                    # the loader caches per repo path; reset between cases


# ---- loader: never raises, degrades to {} -----------------------------------
def test_load_absent_returns_empty(tmp_path):
    _fresh()
    assert config.load(str(tmp_path)) == {}               # no file -> today's behavior exactly


def test_load_valid_mapping(tmp_path):
    _fresh()
    (tmp_path / "config.yaml").write_text("recall:\n  recall_k: 3\n", encoding="utf-8")
    assert config.load(str(tmp_path)) == {"recall": {"recall_k": 3}}


def test_load_non_mapping_ignored(tmp_path, capsys):
    _fresh()
    (tmp_path / "config.yaml").write_text("just a bare string\n", encoding="utf-8")
    assert config.load(str(tmp_path)) == {}
    assert "not a mapping" in capsys.readouterr().err


def test_load_malformed_yaml_never_fatal(tmp_path, capsys):
    _fresh()
    (tmp_path / "config.yaml").write_text("recall: [unclosed\n", encoding="utf-8")
    assert config.load(str(tmp_path)) == {}
    assert "could not read" in capsys.readouterr().err


def test_load_is_cached(tmp_path):
    _fresh()
    p = tmp_path / "config.yaml"
    p.write_text("recall:\n  recall_k: 3\n", encoding="utf-8")
    first = config.load(str(tmp_path))
    p.write_text("recall:\n  recall_k: 99\n", encoding="utf-8")   # changed on disk
    assert config.load(str(tmp_path)) is first                    # not re-read within a process


def test_load_tolerates_non_str_or_unhashable_repo():
    _fresh()
    assert config.load(12345) == {}                               # non-str repo → {} (never raises)
    assert config.load(["not", "a", "path"]) == {}                # unhashable repo → {} via the str() cache key


# ---- accessors: defaults, clamps, bad values --------------------------------
def test_get_missing_and_non_dict_section():
    cfg = {"recall": {"recall_k": 3}}
    assert config.get(cfg, "recall", "recall_k", 5) == 3
    assert config.get(cfg, "recall", "absent", 5) == 5
    assert config.get(cfg, "absent_section", "x", 7) == 7
    assert config.get({"recall": "notadict"}, "recall", "x", 1) == 1     # section isn't a mapping
    assert config.get({"recall": {"recall_k": None}}, "recall", "recall_k", 5) == 5   # explicit null -> default


def test_num_clamps_and_rejects_bad(capsys):
    assert config.num({"a": {"b": 0.5}}, "a", "b", 0.2) == 0.5
    assert config.num({"a": {"b": 9}}, "a", "b", 0.2, 0, 1) == 1         # clamp high
    assert config.num({"a": {"b": -9}}, "a", "b", 0.2, 0, 1) == 0        # clamp low
    assert config.num({"a": {"b": "nope"}}, "a", "b", 0.2) == 0.2        # non-numeric -> default
    assert "is not a number" in capsys.readouterr().err
    assert config.num({"a": {"b": float("inf")}}, "a", "b", 0.2) == 0.2  # non-finite -> default


def test_intnum_truncates_then_clamps():
    assert config.intnum({"a": {"b": 7.9}}, "a", "b", 5, 1) == 7
    assert config.intnum({"a": {"b": 0}}, "a", "b", 5, 1) == 1           # below lo -> clamp -> int


def test_enum_validates(capsys):
    choices = ("auto", "true", "false")
    assert config.enum({"s": {"e": "true"}}, "s", "e", "auto", choices) == "true"
    assert config.enum({}, "s", "e", "auto", choices) == "auto"         # missing -> default
    assert config.enum({"s": {"e": "wat"}}, "s", "e", "auto", choices) == "auto"
    assert "not in" in capsys.readouterr().err


# ---- wiring: config feeds argparse defaults, flags still override ------------
def test_config_feeds_recall_defaults(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"recall": {"recall_k": 3, "know_k": 4, "episodes_last": 6}})
    p = agent.build_parser()
    assert p.parse_args(["recall", "q"]).k == 3
    assert p.parse_args(["know", "q"]).k == 4
    assert p.parse_args(["episodes"]).last == 6


def test_cli_flag_overrides_config(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"recall": {"recall_k": 3}})
    p = agent.build_parser()
    assert p.parse_args(["recall", "q", "-k", "1"]).k == 1               # explicit flag wins over config


def test_config_live_default_flow(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"display": {"live_default_flow": "sleep"}})
    assert agent.build_parser().parse_args(["live"]).flow == "sleep"


def test_config_invalid_live_flow_falls_back(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"display": {"live_default_flow": "bogus"}})
    assert agent.build_parser().parse_args(["live"]).flow == "react"     # unknown -> safe default


def test_absent_config_keeps_builtin_defaults(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {})
    p = agent.build_parser()
    assert p.parse_args(["recall", "q"]).k == 5                          # the original baked default
    assert p.parse_args(["know", "q"]).k == 8
    assert p.parse_args(["episodes"]).last == 10


# ---- engine knobs reach the runtime Brain -----------------------------------
def test_runtime_brain_reads_engine_knobs(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {
        "memory": {"working_memory_span": 3, "min_confidence_to_promote": 0.8,
                   "consolidation_promote_threshold": 0.7, "consolidation_forget_threshold": 0.3,
                   "retention_age_days": 14, "graph_prune_max_edges": 50},
        "affect": {"emotion_half_life_seconds": 60, "mood_half_life_seconds": 100},
    })
    b = runtime.Brain(root=str(tmp_path / "a"))                          # fresh root -> no efficacy.yaml -> config supplies defaults
    assert b._wm_span == 3
    assert b.min_conf == 0.8
    assert b._promote_thr == 0.7
    assert b._forget_thr == 0.3
    assert b._retention_age == 14
    assert b._max_edges == 50
    assert b._emotion_half == 60
    assert b._mood_half == 100


def test_runtime_brain_clamps_bad_knobs(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {"memory": {"working_memory_span": 1, "min_confidence_to_promote": 9}})
    b = runtime.Brain(root=str(tmp_path / "b"))
    assert b._wm_span == 2                                               # clamped to floor 2
    assert b.min_conf == 1.0                                             # clamped into [0, 1]


# ---- safety invariants: configurable but never disablable -------------------
def test_runtime_safety_knobs(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {"safety": {"value_uncertainty": 0.7}, "affect": {"homeostasis_pull": 0.3}})
    b = runtime.Brain(root=str(tmp_path / "s"))
    assert b.value_uncertainty == 0.7
    assert b._homeostasis_pull == 0.3


def test_corrigibility_floor_cannot_be_disabled(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {"safety": {"value_uncertainty": 0.0}})   # try to switch corrigibility off
    b = runtime.Brain(root=str(tmp_path / "f"))
    assert b.value_uncertainty == 0.1                                   # clamped to the floor - safety can't be configured away


# ---- session directives -----------------------------------------------------
def test_session_directives_formatting(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"session": {"directives": ["one task only", "  ", "warn on long context"]}})
    out = agent._session_directives()
    assert "session directives" in out
    assert "- one task only" in out and "- warn on long context" in out
    assert out.count("- ") == 2                                         # blank entry skipped


def test_session_directives_empty_or_malformed(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {})
    assert agent._session_directives() == ""
    monkeypatch.setattr(agent, "CONFIG", {"session": {"directives": "not a list"}})
    assert agent._session_directives() == ""


# ---- precedence: env beats config; config beats default --------------------
def test_data_home_precedence_env_beats_config(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"paths": {"brain_home": "/tmp/brain_cfg_home"}})
    monkeypatch.setenv("BRAIN_HOME", "/tmp/brain_env_home")
    assert agent._data_home() == "/tmp/brain_env_home"                  # env wins over config
    monkeypatch.delenv("BRAIN_HOME", raising=False)
    monkeypatch.delenv("BRAIN_LLM_HOME", raising=False)
    assert agent._data_home() == os.path.abspath(os.path.expanduser("/tmp/brain_cfg_home"))   # then config.yaml


def test_semantic_can_be_forced_off(monkeypatch):
    monkeypatch.setattr(agent, "_SEMANTIC", "false")
    assert agent._semantic_ready() is False                             # config/env force the lexical fallback


# ---- the remaining argparse knobs feed their defaults ----------------------
def test_config_feeds_encode_and_goal_defaults(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"memory": {"encode_confidence": 0.33, "learn_confidence": 0.44},
                                          "goals": {"importance": 0.22, "urgency": 0.11},
                                          "affect": {"default_praise": -0.5}})
    p = agent.build_parser()
    assert p.parse_args(["react", "e", "0.1", "0.1", "0.1"]).confidence == 0.33
    assert p.parse_args(["learn", "f"]).confidence == 0.44
    g = p.parse_args(["goals", "--add", "x"])
    assert g.importance == 0.22 and g.urgency == 0.11
    assert p.parse_args(["remember", "e", "0.5", "0.1", "0.1", "0.1"]).praise == -0.5


# ---- behavioral: a knob actually changes engine behavior, not just an attr --
def test_working_memory_span_caps_the_note_buffer(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {"memory": {"working_memory_span": 3}})
    b = runtime.Brain(root=str(tmp_path / "wm"))
    for t in ["a", "b", "c", "d", "e"]:
        b.note(t)
    kept = [ln for ln in b.working.splitlines() if ln.strip().startswith("- ")]
    assert len(kept) == 3                                               # the knob bounds the buffer, not just self._wm_span


def test_efficacy_yaml_wins_over_config_default(monkeypatch, tmp_path):
    import runtime
    self_dir = tmp_path / "e" / "self"
    self_dir.mkdir(parents=True)
    (self_dir / "efficacy.yaml").write_text(
        "efficacy: {}\nmin_confidence_to_promote: 0.42\nvalue_uncertainty: 0.9\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_CFG", {"memory": {"min_confidence_to_promote": 0.1}, "safety": {"value_uncertainty": 0.5}})
    b = runtime.Brain(root=str(tmp_path / "e"))
    assert b.min_conf == 0.42                                           # per-agent learned value beats the config default
    assert b.value_uncertainty == 0.9


def test_graph_prune_uses_configured_max_edges(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "_CFG", {"memory": {"graph_prune_max_edges": 10}})   # 10 = the clamp floor
    b = runtime.Brain(root=str(tmp_path / "g"))
    b.graph = {"nodes": [{"id": str(i)} for i in range(14)],
               "edges": [{"from": str(i), "to": str(i + 1), "weight": 0.1 + i * 0.05} for i in range(13)]}
    b._prune_graph(b._max_edges)                                        # the exact call sleep() makes (runtime.py:632)
    assert b._max_edges == 10
    assert len(b.graph["edges"]) == 10                                  # bounded to the configured max, not the method default 500


def test_mood_half_life_changes_reactivity(monkeypatch, tmp_path):
    import brain as B
    import runtime

    def mood_after(half):
        monkeypatch.setattr(runtime, "_CFG", {"affect": {"mood_half_life_seconds": half}})
        b = runtime.Brain(root=str(tmp_path / f"m{int(half)}"))
        b.perceive("a huge win", B.Appraisal(0.5, 0.9, 0.9, 0.9), domain="d", outcome="success", now=1_750_000_000.0)
        return b.mood.valence

    assert mood_after(1.0) > mood_after(1_000_000.0)                    # a SHORT mood half-life reacts far more to the same event


def test_lock_acquire_timeout_resolves_from_config_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "CONFIG", {"integrations": {"lock_acquire_timeout": 3.0}})
    monkeypatch.setattr(agent, "AGENTS_DIR", str(tmp_path))
    try:
        agent._acquire_agent_lock("locktest")                          # timeout=None -> reads config.integrations.lock_acquire_timeout
        assert os.path.isfile(os.path.join(str(tmp_path), "locktest", ".lock"))
    finally:
        agent._release_agent_lock()


def test_persona_natural_mode_injects_silent_directive(monkeypatch):
    monkeypatch.setattr(agent, "CONFIG", {"persona": {"style": "natural"}})
    d = agent._persona_directive()
    assert "natural mode" in d and "never show the user" in d           # natural -> the silent-presentation directive
    monkeypatch.setattr(agent, "CONFIG", {"persona": {"style": "expressive"}})
    assert agent._persona_directive() == ""                             # expressive (default) -> no directive
    monkeypatch.setattr(agent, "CONFIG", {})
    assert agent._persona_directive() == ""                             # absent -> default expressive
    monkeypatch.setattr(agent, "CONFIG", {"persona": {"style": "bogus"}})
    assert agent._persona_directive() == ""                             # invalid -> clamped to expressive

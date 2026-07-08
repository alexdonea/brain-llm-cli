"""Coverage for two previously-untested hazards:
  • live_brain.animate must RESTORE the terminal (cbreak/cursor) even when a render raises (TEST-08) - else a
    crash leaves the user's terminal hidden-cursor / in cbreak.
  • semantic.load_index must REJECT a .npy / .ids.json desync (same row count, different vectors) rather than
    serve wrong vectors (TEST-06). Uses a fake embed, so it needs only numpy, not wordllama."""
import io as _io
import os as _os
import sys as _sys
import types as _types

import pytest

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import live_brain as lb
import semantic as sm


class _Tty(_io.StringIO):
    def isatty(self):
        return True


def test_animate_restores_terminal_even_when_render_raises(monkeypatch):
    # force the interactive path (tty) and capture writes, without touching the real terminal
    monkeypatch.setattr(lb, "sys", _types.SimpleNamespace(stdout=_Tty(), stdin=_Tty()))
    restored = []
    monkeypatch.setattr(lb, "_enter_cbreak", lambda: "SAVED")
    monkeypatch.setattr(lb, "_restore_term", lambda saved: restored.append(saved))
    monkeypatch.setattr(lb, "_paint", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_frame", lambda *a, **k: [])
    monkeypatch.setattr(lb, "_footer", lambda *a, **k: "")
    monkeypatch.setattr(lb, "snapshot", lambda b: {})
    monkeypatch.setattr(lb, "last_activation", lambda root: None)   # stay idle so the idle branch runs

    boom = RuntimeError("render blew up")
    monkeypatch.setattr(lb, "_wait_key", lambda *a, **k: (_ for _ in ()).throw(boom))

    class _FakeBrain:
        root = None

    with pytest.raises(RuntimeError):
        lb.animate(_FakeBrain(), flow="react")
    assert restored == ["SAVED"]                                  # finally restored the terminal despite the crash


def test_load_index_rejects_matrix_desync(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    mem = str(tmp_path)
    monkeypatch.setattr(sm, "embed", lambda texts, home: np.ones((len(list(texts)), 4), dtype="float32"))
    items = [{"id": "e-1", "task": "alpha"}, {"id": "e-2", "task": "beta"}]
    sm.build_index(mem, items, home=mem, kind="episodic")
    assert sm.load_index(mem, "episodic") is not None             # a consistent cache loads fine

    npy, _ = sm.cache_paths(mem, "episodic")
    np.save(npy, np.zeros((2, 4), dtype="float32"))               # SAME shape, different bytes → a desync
    assert sm.load_index(mem, "episodic") is None                 # detected via the stored matrix fingerprint


def test_ensure_index_incremental_reuses_unchanged_rows(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    mem = str(tmp_path)
    calls = []

    def fake_embed(texts, home):
        texts = list(texts)
        calls.append(len(texts))
        return np.asarray([[float(len(t)), 0.0, 0.0, 0.0] for t in texts], dtype="float32")

    monkeypatch.setattr(sm, "embed", fake_embed)
    items = [{"id": "e-1", "task": "a"}, {"id": "e-2", "task": "bb"}]
    sm.ensure_index(mem, items, home=mem, kind="episodic")
    assert calls == [2]                                           # first build embeds both
    items2 = items + [{"id": "e-3", "task": "ccc"}]               # add one item
    ids, M = sm.ensure_index(mem, items2, home=mem, kind="episodic")
    assert calls == [2, 1]                                        # only the NEW item is embedded, not all three
    assert len(ids) == 3 and M.shape == (3, 4)


def test_last_activation_returns_the_last_valid_record(tmp_path):
    """Regression: a torn final line in the activation log must yield the last VALID record, not None."""
    wk = tmp_path / "working"
    wk.mkdir(parents=True)
    (wk / "activations.jsonl").write_text('{"t": 1.0, "region": "ok"}\n{ this final line is torn\n', encoding="utf-8")
    rec = lb.last_activation(str(tmp_path))
    assert rec is not None and rec["t"] == 1.0


def test_semantic_dedup_facts(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(sm, "is_ready", lambda home: True)
    
    # Fake embed: maps "dup1" and "dup2" to the same vector, others to orthogonal vectors
    def fake_embed(texts, home):
        vecs = []
        for t in texts:
            if "dup" in t:
                vecs.append([1.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0])
        return np.asarray(vecs, dtype="float32")
    
    monkeypatch.setattr(sm, "embed", fake_embed)
    
    facts = [
        {"id": "f-1", "text": "dup1"},
        {"id": "f-2", "text": "dup2"},    # should replace f-1 because it's newer and similar
        {"id": "f-3", "text": "unique"}
    ]
    
    deduped = sm.semantic_dedup_facts(facts, home="fake", threshold=0.9)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "f-2"
    assert deduped[1]["id"] == "f-3"

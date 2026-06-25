"""src/semantic.py - local, offline semantic search for recall (WordLlama).

Recall is built on this module: it turns the lexical `query_relevance` term (src/agent.py `_recall_relevance`,
the 0.30-weight seam in `brain.retrieval_score`) into a dense, MEANING-aware score, so `recall "fear of losing
money"` surfaces the loss-aversion / risk episodes even though they share no words with the query, and
`recall --search` ranks purely by meaning.

THE BACKEND IS LOCAL AND OFFLINE BY CONSTRUCTION. The embedder is WordLlama `l2_supercat` (256-d static
token embeddings, numpy-only inference - no PyTorch). Its 16MB weights ship INSIDE the `wordllama` pip
package, and its 1.8MB tokenizer is VENDORED into the repo at <data home>/models/wordllama/tokenizers/
(committed). The model is loaded with `disable_download=True`, so there is NEVER a network call - verified
with the socket layer hard-blocked. (If a non-repo BRAIN_HOME lacks the vendored tokenizer, it is copied
from the installed package - a local copy, still no network.)

DESIGN PRINCIPLES:
  • GRACEFUL - wordllama is a required dependency, but if it is ever unavailable every function degrades and
    recall falls back to the lexical term. `available()` is the single gate; nothing here is imported at startup.
  • FILES STAY SOURCE OF TRUTH - episode embeddings are a derived, rebuildable `.npy` cache, never canonical.
  • INCREMENTAL - `ensure_index` re-embeds ONLY the items whose (id, text) changed and reuses every other
    cached row, so adding one memory to a 50k store embeds one vector, not 50,000. Recall stays fast as
    memory grows; a hand-edited or deleted item is detected per-row and never goes stale.

Chosen on data: a local bake-off over a clean 16-episode store / 18 meaning-not-word queries gave WordLlama
18/18 hit@3 (MRR 0.91) vs lexical 9/18 - the best static option tested. See docs/research/semantic-search.md.
"""
from __future__ import annotations
import hashlib
import os
import json as _json
import shutil

_TOKENIZER = "l2_supercat_tokenizer_config.json"
_MODEL = None                          # process-level cached model


def available() -> bool:
    """True iff the optional deps are importable. The single gate the rest of the engine checks."""
    try:
        import numpy  # noqa: F401
        import wordllama  # noqa: F401
        return True
    except Exception:
        return False


def model_dir(home) -> str:
    """Where the vendored tokenizer lives: <data home>/models/wordllama/ (committed; weights come from pip)."""
    return os.path.join(str(home), "models", "wordllama")


def _ensure_tokenizer(home):
    """Guarantee the 1.8MB tokenizer is in the local model dir. It is committed to the repo; if a different
    BRAIN_HOME lacks it, copy it from the installed `wordllama` package - a LOCAL copy, never a network fetch."""
    dst = os.path.join(model_dir(home), "tokenizers", _TOKENIZER)
    if not os.path.exists(dst):
        import wordllama
        src = os.path.join(os.path.dirname(wordllama.__file__), "tokenizers", _TOKENIZER)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return model_dir(home)


def is_ready(home) -> bool:
    """True iff semantic search can actually run right now (deps installed AND the tokenizer is in place)."""
    if not available():
        return False
    try:
        _ensure_tokenizer(home)
        return True
    except Exception:
        return False


def _load(home):
    """Lazy-load the static model: weights from the pip package, tokenizer from the local repo dir, with
    downloads DISABLED - so this can never touch the network."""
    global _MODEL
    if _MODEL is None:
        cache = _ensure_tokenizer(home)
        from wordllama import WordLlama
        _MODEL = WordLlama.load(cache_dir=cache, disable_download=True)
    return _MODEL


def embed(texts, home):
    """Embed a list of strings -> L2-normalized float32 matrix (N, dim). Cosine == dot product."""
    import numpy as np
    vecs = np.asarray(_load(home).embed(list(texts)), dtype="float32")
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def relevance_scores(query: str, matrix, home):
    """Cosine of `query` vs a prebuilt normalized matrix, mapped to [0,1] via (c+1)/2."""
    q = embed([query], home)[0]
    return (matrix @ q + 1.0) / 2.0


# ── derived vector index (per-agent cache) - a CACHE, never the source of truth ──────────────────────
# Two kinds share the same machinery, each with its own cache: "episodic" (events.jsonl `task`) and
# "facts" (semantic/facts.yaml `text` - the neocortex, searched by `know`).
_KINDS = {"episodic": ("episodic", "embeddings", "task"),
          "facts":    ("semantic", "embeddings_facts", "text")}


def cache_paths(mem_root, kind="episodic"):
    sub, stem, _ = _KINDS[kind]
    d = os.path.join(str(mem_root), sub)
    return os.path.join(d, stem + ".npy"), os.path.join(d, stem + ".ids.json")


def _item_hash(e, tk):
    """Per-item content hash over (id, text) - the identity of one row. A changed text or id ⇒ a new hash
    ⇒ ensure_index re-embeds that row, and ONLY that row."""
    return hashlib.sha1((str(e["id"]) + "\x00" + str(e.get(tk, ""))).encode()).hexdigest()[:16]


def _fp_of(hashes):
    """Whole-store fingerprint = hash of the ordered per-item hashes (the cheap fast-path equality check)."""
    h = hashlib.sha1()
    for x in hashes:
        h.update(x.encode()); h.update(b"\n")
    return h.hexdigest()


def load_index(mem_root, kind="episodic"):
    """Return (ids, matrix) from the on-disk cache, or None if absent/corrupt/misaligned (or numpy missing)."""
    npy, idsf = cache_paths(mem_root, kind)
    if not (os.path.exists(npy) and os.path.exists(idsf)):
        return None                                                # no cache: degrade before touching numpy
    try:
        import numpy as np                                         # guarded: a numpy-less host still degrades to None
        meta = _json.load(open(idsf, encoding="utf-8"))
        ids = meta["ids"] if isinstance(meta, dict) else meta      # tolerate a legacy bare-list file
        M = np.load(npy)
        return (ids, M) if len(ids) == M.shape[0] else None
    except Exception:
        return None


def _write_cache(mem_root, kind, ids, hashes, M):
    """Atomically persist the matrix + {fp, ids, per-item hashes}. The per-item hashes are what let
    ensure_index reuse unchanged rows instead of re-embedding the whole store."""
    import numpy as np
    npy, idsf = cache_paths(mem_root, kind)
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    np.save(npy + ".tmp.npy", M); os.replace(npy + ".tmp.npy", npy)
    tmp = idsf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump({"fp": _fp_of(hashes), "ids": list(ids), "h": list(hashes)}, f)
    os.replace(tmp, idsf)


def build_index(mem_root, items, home, kind="episodic"):
    """Embed EVERY item's text and (atomically) write the cache. Returns (ids, matrix). The full rebuild -
    used by `reindex` and as the fallback when no reusable cache exists; ensure_index prefers incremental."""
    import numpy as np
    tk = _KINDS[kind][2]
    ids = [e["id"] for e in items]
    hashes = [_item_hash(e, tk) for e in items]
    M = embed([str(e.get(tk, "")) for e in items], home) if items else np.zeros((0, 256), dtype="float32")
    _write_cache(mem_root, kind, ids, hashes, M)
    return ids, M


def ensure_index(mem_root, items, home, kind="episodic"):
    """Return (ids, matrix) for the current items, embedding ONLY what changed.

    Fast path: if the whole-store fingerprint still matches, return the cache untouched (zero embeds).
    Incremental path: reuse every cached row whose (id, text) is unchanged and embed only the new/edited
    items - so adding one memory to a 50k store embeds one vector, not 50,000. Reorders and deletions
    cost zero embeds (rows are reshuffled/dropped). Falls back to a full build when there is no usable
    cache (absent / corrupt / legacy file without per-item hashes). The output is what build_index would
    produce - embedding is deterministic, so the incremental cache never drifts from a full rebuild."""
    import numpy as np
    tk = _KINDS[kind][2]
    _, idsf = cache_paths(mem_root, kind)
    cur_hashes = [_item_hash(e, tk) for e in items]
    cached = load_index(mem_root, kind)
    meta = None
    if cached is not None:
        try:
            meta = _json.load(open(idsf, encoding="utf-8"))
        except Exception:
            meta = None
    # fast path - the store is unchanged since the cache was written
    if cached is not None and isinstance(meta, dict) and meta.get("fp") == _fp_of(cur_hashes):
        return cached
    # incremental path - needs the cache's per-item hashes to know which rows are still valid
    if (cached is not None and isinstance(meta, dict) and isinstance(meta.get("h"), list)
            and len(meta["h"]) == cached[1].shape[0]):
        old_M = cached[1]
        dim = old_M.shape[1] if (old_M.ndim == 2 and old_M.shape[1]) else 256
        row_of = {}
        for i, hsh in enumerate(meta["h"]):
            row_of.setdefault(hsh, i)                       # item-hash -> its row in the cached matrix
        ids = [e["id"] for e in items]
        new_M = np.zeros((len(items), dim), dtype="float32")
        to_pos, to_txt = [], []
        for pos, (e, hsh) in enumerate(zip(items, cur_hashes)):
            r = row_of.get(hsh)
            if r is not None:
                new_M[pos] = old_M[r]                       # unchanged item - reuse its vector, no embed
            else:
                to_pos.append(pos); to_txt.append(str(e.get(tk, "")))
        if to_txt:                                          # embed ONLY the new / changed items
            emb = embed(to_txt, home)
            for k, pos in enumerate(to_pos):
                new_M[pos] = emb[k]
        _write_cache(mem_root, kind, ids, cur_hashes, new_M)
        return ids, new_M
    # no usable cache - full build
    return build_index(mem_root, items, home, kind)


def dense_relevance(query, ids, matrix, home):
    """{episode_id -> relevance in [0,1]} from cosine similarity, min-max normalized across the candidate
    set so the dense term keeps full dynamic range inside brain.retrieval_score's weighted-linear sum."""
    import numpy as np
    if matrix.shape[0] == 0:
        return {}
    sims = matrix @ embed([query], home)[0]          # both sides L2-normalized -> cosine
    lo, hi = float(sims.min()), float(sims.max())
    norm = (sims - lo) / (hi - lo) if (hi - lo) > 1e-9 else np.full_like(sims, 0.5)
    return {i: float(x) for i, x in zip(ids, norm)}


if __name__ == "__main__":  # quick self-test against the real brain (run with a venv that has wordllama)
    import sys, time
    home = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    agent = sys.argv[2] if len(sys.argv) > 2 else "sage"
    print("available:", available(), "| model dir:", model_dir(home), "| ready:", is_ready(home))
    ep = os.path.join(home, "agents", agent, "memory", "episodic", "events.jsonl")
    tasks = [_json.loads(l)["task"] for l in open(ep)]
    t = time.time(); M = embed(tasks, home); print(f"embedded {len(tasks)} episodes in {(time.time()-t)*1000:.0f}ms, dim={M.shape[1]}")
    for q in ["fear of losing money", "an option's sensitivity to time decay"]:
        s = relevance_scores(q, M, home)
        top = sorted(range(len(tasks)), key=lambda i: s[i], reverse=True)[:3]
        print(f"\n  q: {q!r}")
        for i in top:
            print(f"    {s[i]:.3f}  {tasks[i][:70]}")

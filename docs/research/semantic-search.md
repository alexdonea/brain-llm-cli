# Semantic search for brain-llm - decision record

**Status:** shipped · **Branch:** `feature/add-semantic-search`
**TL;DR:** Add a dense, meaning-aware term to recall's `query_relevance` seam using a small **local,
offline** static embedding model - **WordLlama `l2_supercat`** (256-d, numpy-only inference). Keep it
**optional** (lexical stays the zero-dependency default) and keep the **human-readable files as the source
of truth**; the vector index is a derived, rebuildable `.npy` cache. The model is **fully local with zero
network**: the 16MB weights ship inside the `wordllama` pip package and the 1.8MB tokenizer is **vendored
and committed** at `models/wordllama/`, loaded with `disable_download=True`. No database, no graph DB, no API,
no download step. Also exposed: **`recall --search`**, a relevance-first mode (find the memory ABOUT x).

---

## 1. The problem

Recall ranks episodes with an affect-weighted hybrid score (`brain.retrieval_score`):

```
score = 0.20·recency(ACT-R) + 0.30·salience + 0.30·query_relevance + 0.15·graph_proximity + 0.05·mood_congruence
```

`query_relevance` is computed **lexically** (word overlap, `agent.py:_relevance`). Lexical matching is
*blind to meaning*: `recall "fear of losing money"` shares no words with the *loss-aversion / risk /
survival* episodes, so it scores them **0** and falls back to arbitrary order.

## 2. What the field actually does (two deep-research passes, adversarially verified)

- **Semantic search is NOT used alone.** From Stanford's *Generative Agents* (2023) to production systems
  (Zep/Graphiti, Mem0, sqlite-memory), retrieval is **hybrid** - dense semantic **+** lexical (BM25) **+**
  recency/importance scoring. On some domains BM25 *alone* beats dense. Pure vector loses to hybrid.
- **brain-llm is already aligned with best practice.** Its `recency + salience + relevance + graph + mood`
  score is a *Generative-Agents-style* `recency + importance + relevance` hybrid - plus extras. The one
  gap: everyone computes *relevance* as **dense embedding cosine**; ours is lexical.
- **A graph layer is not a universal win.** Mem0's Neo4j variant (`Mem0g`) helped only temporal reasoning
  and ~doubled token cost. → no graph DB for us.
- **Highest-leverage upgrade:** make `query_relevance` a dense-semantic cosine. One term, not a rewrite.

## 3. Local model bake-off (on the real episode store, 8 meaning-not-word queries)

Metrics: **hit@3** (correct episode in top 3) and **MRR** (1/rank of first hit). All models run locally.

| Method | Type | Install deps | Model | hit@3 | MRR | embed | offline |
|---|---|---|---|---|---|---|---|
| Lexical *(current)* | - | **0** | - | 75%¹ | 0.69 | instant | ✓ |
| Stdlib char-trigram | - | **0** | - | 50% | 0.52 | instant | ✓ |
| **potion-base-2M** | static (model2vec) | ~98MB | **8MB** | **100%** | 0.75 | 5ms | ✓ |
| **potion-base-8M** ⭐ | static (model2vec) | ~98MB | **30MB** | **100%** | **0.94** | **1ms** | ✓ |
| fastembed bge-small | ONNX | 191MB | 130MB | 100% | 0.88 | 249ms | ✓ |
| ST all-MiniLM-L6-v2 | PyTorch | **905MB** | 90MB | 100% | **1.00** | 1961ms | ✓ |
| ST bge-small-en-v1.5 | PyTorch | **905MB** | 130MB | 100% | 0.88 | 173ms | ✓ |

¹ Lexical's 75% is flattering: when it scores 0 (no shared words) it "hits" only because the relevant
episodes happen to sit first in the file; it genuinely misses targets stored deeper (valuation, macro).

**Read:** every real embedder solves the lexical blind spot (100% hit@3). `potion-base-8M` has the best
quality-per-MB by far - it beats both bge-small variants and is within 0.06 MRR of MiniLM, which costs
~30× the disk and ~2000× the embed time. The static model is the sweet spot for a local-first, minimal,
fast project. *(Caveat: small benchmark - 28 episodes, 8 queries; don't over-read 3rd-decimal MRR gaps.)*

### 3b. Static head-to-head - model2vec vs WordLlama (clean 16-episode store, 18 meaning-not-word queries)

Re-run on a FRESH agent (no session-noise contamination) with a larger query set:

| Backend | hit@3 | MRR | embed | install | model dl | offline |
|---|---|---|---|---|---|---|
| Lexical *(current)* | 9/18 (50%) | 0.52 | - | 0 | - | ✓ |
| Stdlib char-trigram | 14/18 (78%) | 0.67 | - | 0 | - | ✓ |
| model2vec potion-8M | 16/18 (89%) | 0.81 | 1ms | 98MB / 29 pkgs | 30MB | ✓ |
| **WordLlama l2_supercat** | **18/18 (100%)** | **0.91** | 1ms | 134MB / 35 pkgs | 1.8MB | ✓ |

Both are numpy-only inference, ~1ms/embed, and **verified fully offline** (loaded with the socket layer
hard-blocked → identical vectors). WordLlama is *sharper* on this corpus (e.g. "an option's sensitivity to
time decay" → the options/Greeks episode, which potion-8M sends to survival-math) and ships a tiny 1.8MB
model, but its `pip install` is heavier (35 vs 29 pkgs). model2vec is the lighter install and rates higher on
*general* MTEB; WordLlama wins on *this* (episodic-memory) corpus. The difference is real but modest -
both are excellent. NOTE the earlier "WordLlama ≈ near-zero-dependency" framing was **wrong on measurement**:
`pip install wordllama` pulls the full HF stack (tokenizers, safetensors, httpx, pydantic, …), heavier than
model2vec - verified.

## 4. Fully-local / offline guarantee (verified) - WordLlama, zero network

- The 16MB weights ship **inside the `wordllama` pip package** (`wordllama/weights/l2_supercat_256.safetensors`).
- The 1.8MB tokenizer is **vendored and committed** at `<data home>/models/wordllama/tokenizers/`.
- The model is loaded with **`WordLlama.load(cache_dir=models/wordllama, disable_download=True)`** - weights
  from the package, tokenizer from the committed dir, downloads disabled. **There is no network call, ever.**
- Proof: with the `socket` layer hard-blocked (any `create_connection`/`getaddrinfo` raises) **and the global
  HF cache deleted**, loading still embeds correctly. No `download-model` command exists; nothing to fetch.
- Robustness: if a non-repo `$BRAIN_HOME` lacks the vendored tokenizer, it is copied from the installed
  package (`wordllama/tokenizers/…`) - a **local file copy, still no network**.

## 5. Decision

| Aspect | Choice |
|---|---|
| **Embedder** | **WordLlama `l2_supercat`** (static, 256-d, numpy-only inference, ~1ms/embed). Chosen over model2vec on the empirical bake-off (18/18 vs 16/18 hit@3) and over PyTorch/ONNX on footprint. |
| **Dependency** | **Optional.** Default = lexical (zero new deps). `pip install wordllama` activates the dense term. `src/semantic.py.available()` is the single gate; nothing imports wordllama at startup. |
| **Locality** | **Fully local, zero network.** Weights from the pip package, tokenizer vendored + committed at `models/wordllama/`, `disable_download=True`. |
| **Storage** | Human-readable files stay the source of truth. Episode embeddings = derived `episodic/embeddings.npy` cache, rebuilt from `events.jsonl` by `reindex` (recall also auto-builds it). No DB, no graph DB. |
| **Scaling** | `ensure_index` is **incremental** - a per-row content hash in `embeddings.ids.json` lets `recall`/`know` re-embed only new/edited items and reuse the rest, so adding a memory to a 50k store embeds 1 vector (~24 ms), not 50,000 (~0.5 s). Query is a brute-force numpy cosine: ~2 ms at 50k, ~4 ms at 100k. The host LLM only ever sees the top-K retrieved (~1.8k tokens/turn), so context stays flat as memory grows. |
| **Fusion** | Dense cosine (min-max normalized per query) fused with lexical as `max(lexical, dense)` in `query_relevance` - exact word matches never suppressed; meaning carries the rest. `recall --search` re-weights to relevance-first `(0.05, 0.10, 0.80, 0.05, 0.0)`. |
| **Rejected** | model2vec/potion (lower hit@3 here), sentence-transformers/PyTorch (905MB), fastembed/ONNX (191MB), any embeddings API (breaks offline), SQLite-as-canonical (kills git-diff transparency; the "SQLite does everything files can + more" claim failed verification). |

## 6. Implementation - SHIPPED

1. ✅ `src/semantic.py` - optional WordLlama backend: strict-offline load (`disable_download=True`), vendored
   tokenizer (self-heals from the package), graceful lexical fallback when `wordllama` is absent.
2. ✅ `brain reindex` - build/refresh the per-agent `episodic/embeddings.npy` cache. **No `download-model`** - the
   model is local by construction (vendored + pip package).
3. ✅ Dense term fused into `agent.py:_recall_relevance` as `max(lexical, dense)`, behind the `is_ready()` gate
   (pure-lexical fallback otherwise). `runtime.recall(w=...)` lets a weight tuple override the affective weights.
4. ✅ `recall --search` - relevance-first weights so an old, low-salience but on-topic memory surfaces by MEANING,
   distinct from mood-coloured default `recall`.
5. ✅ Tests (green; 266 total): `available()`/`is_ready()` gates safe without wordllama, `reindex` guides instead
   of crashing, `--search` runs under lexical, `recall(w=...)` is relevance-first, a direct embed/index/rank unit test.
6. ✅ `requirements.txt` - `wordllama` listed as an **optional** extra, never a hard dependency.

### Honest note on end-to-end behaviour
Plain `recall` is affective by design (salience + recency co-weighted with relevance at 0.30 each), so on a
store full of recent high-salience episodes the dense term is *present but muted* - the same loud memories
surface for different queries. The dense signal's full value shows through **`recall --search`**, which is
relevance-first and reliably surfaces old, on-topic, low-salience memories (verified: "fair value of a
company" → the valuation episode; "interest rates and inflation" → the macro episode; "fear of losing money"
→ risk-management + survival-math). Use default `recall` for "what comes to mind given my mood", `--search`
for "find the memory ABOUT x".

## Sources
Generative Agents (Park et al. 2023, arXiv:2304.03442) · Zep/Graphiti (arXiv:2501.13956) · Mem0
(arXiv:2504.19413) · A-Mem · sqlite-memory · model2vec/POTION (MinishLab) · BM25-vs-dense (arXiv:2604.01733).

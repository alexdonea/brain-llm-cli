# Cognitive Architecture v0.0.4: J-Space, Dreaming, and Multi-Agent Orchestration

The 0.0.4 update to the `brain-llm` architecture represents a significant leap from simple disk-backed storage to a true, biologically-inspired computational cognitive model. We have integrated advanced insights from interpretability research (such as Anthropic's Jacobian Lens) and biological memory consolidation (Sleep/Dreaming).

## 1. J-Space Externalization (The Thinking Protocol)

Large Language Models perform immense latent computations in their internal activations before outputting a single token. Anthropic's research into the **Jacobian Lens (J-Space)** reveals that models construct rich, multidimensional concept spaces internally. 

However, if an LLM is forced to immediately execute a bash command or write a memory fact without a scratchpad, its ability to reliably extract and encode these high-dimensional concepts drops severely. It suffers from context pollution or premature collapse of the concept state.

**Solution: The `<thinking>` Protocol**
By enforcing a `<thinking>...</thinking>` block prior to any memory-altering CLI command, we force the LLM to serialize its J-Space activations into textual tokens. This provides a "Global Workspace" where the concept is stabilized. Once the concept is explicitly decoded into text, the subsequent CLI command (`learn` or `react`) is formulated with vastly higher precision and confidence. 

Furthermore, we instituted **Epistemological Tracking** (`--confidence` and `--source`), ensuring that the resulting graph edges are weighted by the LLM's own self-assessed certainty from its J-Space reflection.

## 2. "Claude Dreaming" (Semantic Memory Consolidation)

In biological brains, the neocortex does not simply accrue endless raw episodic memories. During REM and Deep Sleep, memories are replayed, generalized, and deduplicated into semantic knowledge.

Our architecture mimics this via the `sleep` command:
1.  **Algorithmic Dreaming:** When the agent sleeps, a deterministic, LLM-free loop fires.
2.  **Dense Embedding Deduplication:** It uses local `wordllama` embeddings to compare all newly acquired facts against the existing knowledge graph.
3.  **Threshold Merging:** If two facts exhibit a cosine similarity greater than `0.90`, the system automatically merges them, averting "neocortex bloat." The graph remains sparse, highly connected, and meaningful without retaining exact duplicates of lessons learned across multiple sessions.

## 3. Context Compaction (`compact`)

One of the persistent challenges for agentic AI is context window exhaustion when reading large files or logs. The `compact` command serves as an attention filter. By using the same offline `wordllama` embedding model, the agent can algorithmically extract only the sentences that are semantically relevant to a given query (e.g., retaining only the top 30% of sentences). This allows the LLM to read massive outputs without overflowing its working memory context.

## 4. Multi-Agent Orchestration (The Hive Mind)

Cognition doesn't scale infinitely within a single context window. The 0.0.4 architecture introduces native multi-agent hierarchy:
-   **Executive & Subordinates:** An orchestrator agent can spin up a localized "junior" agent (`create` or `clone`), scoping its prompt to a narrow task.
-   **Asynchronous Delegation:** The `delegate` command injects an executive goal into the child's brain and sets a pending intent in the parent's brain.
-   **Social Inbox:** Communication happens via `message` and `inbox`, providing decoupled, robust inter-agent messaging.
-   **Semantic Cross-Pollination:** Sub-agents can use `learn --share-with <parent>` to push distilled J-Space facts directly into the orchestrator's neocortex before terminating, effectively functioning as specialized cortical columns reporting back to the prefrontal cortex.

# experiments/

Self-contained field experiments that test brain-llm as a whole — a fresh agent living a real arc, with the
on-disk memory preserved as evidence. Each subfolder is one run: a detailed `README.md` plus a
`memory-snapshot/` copy of the agent's actual memory at the end.

| experiment | what it shows |
|------------|---------------|
| [`iris-bioluminescence/`](iris-bioluminescence/) | A fresh agent autonomously chooses a topic, learns it from the web, builds a skill — then **fails** a memory-gap problem, learns from it, and **solves a later similar problem because the failure memory surfaced**. Includes the psych battery and the value-learning loop. |

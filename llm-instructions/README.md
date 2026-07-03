# User Guide Prompts

This directory contains meticulously crafted **prompt templates** designed to boot an LLM into specific, highly-capable roles powered by `brain-llm`. These prompts are not just personas—they are functional operating procedures that tightly integrate with the `brain-llm` CLI's affective memory, planning, and knowledge systems.

## How to use these prompts

You do not run these files directly. Instead, these are **system instructions** or **custom instructions** for your host LLM (e.g., Claude, ChatGPT, Gemini, or a coding assistant like Cursor or Claude Code).

1. **Pick a Role**: Choose the prompt that matches your goal (e.g., `researcher.md` for deep-dive learning, `orchestrator.md` for delegating multi-agent tasks).
2. **Customize (Optional)**: Replace the placeholders (like `<TOPIC>` or `<skill>`) with your specific needs.
3. **Feed to the LLM**: Copy the text of the prompt and paste it into your LLM's system prompt field, custom instructions, or `.cursorrules` file.
4. **Boot the Agent**: Tell the LLM to run `brain-llm <agent_name> wake` to start the session. The LLM will now follow the prompt's loop, using the CLI as its long-term memory.

## Available Roles

- ⭐ **[Generalist](generalist.md)**: **(Recommended Default)** A polymath mind that does it all. It can learn multiple skills, discover tools, perform research, and orchestrate other agents, weaving them all into a single, comprehensive semantic memory. Use this for a long-term companion agent.
- **[Orchestrator](orchestrator.md)**: Learns to decompose goals, delegate tasks to subordinate agents, and synthesize the results. Excellent for complex software development tasks.
- **[Researcher](researcher.md)**: Conducts deep, multi-session research on a topic, encoding facts, formulas, and insights into a durable semantic graph.
- **[Skill Builder](skill_builder.md)**: Focuses on practicing a specific competence (like TDD or Code Review) and distilling successful practice into a reusable playbook.
- **[Toolsmith](toolsmith.md)**: Discovers, classifies, and memorizes external tools (CLIs, APIs, scripts) so they can be invoked later without needing them declared in a system prompt.

## Core Philosophy

Every prompt in this folder enforces the **Golden Rules of brain-llm**:
1. **The CLI is the ONLY memory**: The agent is explicitly forbidden from writing ad-hoc files like `notes.md` or `todo.json`. Everything goes through `react`, `learn`, `plan`, and `next`.
2. **Every session ends in `sleep`**: Sleep is mandatory. It consolidates facts, builds the association graph, and distills playbooks.
3. **Honest development**: Competence and playbooks are grown from *evidence-backed successes*, not hallucinated capabilities.

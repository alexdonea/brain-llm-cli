# Brain-LLM User Guide: From Basics to Advanced Architecture

Welcome to the `brain-llm` user guide. This document walks you through how to use the `brain-llm` framework to drive an autonomous, memory-enabled AI agent. The guide is structured by complexity, starting from basic usage to advanced multi-agent orchestration.

## The Core Concept
The system works by combining two distinct pieces:
1. **The LLM (The Mind/Executor):** This is your AI assistant (e.g., Claude, Cursor, ChatGPT). It reads files, writes code, and talks to you.
2. **The CLI (The Memory/Heart):** This is the `brain-llm` executable on your machine. It stores memories, calculates emotions, and builds neural associations.

You do not read this guide to the LLM directly. Instead, you provide the generated `AGENT-BRAIN.MD` as **System Instructions** to your LLM. The LLM then acts as the agent, autonomously typing `brain-llm` commands in the background to record its life.

---

## Setup & Initialization (For Humans)

Before your LLM can start acting as an agent, you (the human) must initialize the workspace.

### 1. Install the CLI
Before using the system, you must install the `brain-llm` command globally on your machine.
Open your terminal in the `brain-llm-cli` folder, run the installer, and verify it works:
```bash
./install.sh
brain-llm --version
```

### 2. Initialize the Project
Navigate to your own project directory where you want your agent to work, and initialize a new memory structure:
```bash
brain-llm init --name aria
```
*(This generates the local `AGENT-BRAIN.MD` file which acts as the bridge. **CRITICAL:** Using the `--name` flag automatically creates the agent's memory directory. If you just run `brain-llm init` without a name, it will generate a generic template, but you must manually run `brain-llm create <your_agent>` before it can wake up).*

> **Pro Tip (File-less Setup):** If you are using advanced tools like Cursor or Claude and don't want to generate a physical file in your project, use `brain-llm prompt --name aria`. This prints the exact same instructions directly to `stdout`, allowing you to pipe it or inject it instantly into context!

### 3. Instruct Your LLM (Zero-Setup Generalist)
The `init` command automatically configures your generated `AGENT-BRAIN.MD` with the full **Generalist** instructions.
You don't need to copy-paste anything! Simply point your LLM to the `AGENT-BRAIN.MD` file (e.g., mention it in your prompt, or set it as a workspace rule), and the LLM will instantly know how to act as the Generalist agent.

*(Optional: If you want a highly specialized role instead of the default Generalist, you can manually customize the instructions inside `AGENT-BRAIN.MD` to fit your needs).*

Now you are ready to talk to your agent!

---

## Complexity 1: The Basics (The Daily Loop)

In this phase, you will learn how to boot an agent and run it through a standard session.

### 1. Booting Up
Once your LLM has read the `AGENT-BRAIN.MD` file, tell it to wake up your agent (e.g., named `aria`):
> "Wake up your agent named aria."

The LLM will automatically run:
```bash
brain-llm aria wake
```
This loads the agent's mood and past memories.

### 2. Setting a Goal
Agents need direction. Ask the LLM:
> "Set a goal to master Python."

The LLM translates this to:
```bash
brain-llm aria goals --add "master python"
```

When a goal is achieved, mark it as complete:
> "I've mastered Python, mark that goal as done."

```bash
brain-llm aria progress "master python" 1.0  # Setting progress to 1.0 auto-completes the goal
# Or manually complete it:
brain-llm aria goals --complete "master python"
```

### 3. Researching and Memorizing
When you ask the LLM to research a topic, it will use its standard web search or file-reading tools. However, because of the prompt, it will also *memorize* what it learns.
> "Aria, research how Python duck typing works. Summarize it for me and make sure you memorize the key facts."

The LLM translates this to:
```xml
<thinking>
I've researched Python duck typing. The key concept is that Python focuses on an object's methods rather than its explicit class type. I should memorize this fact with high confidence to use it in future tasks.
</thinking>
```
```bash
brain-llm aria learn "FACT: Python uses duck typing, focusing on methods rather than explicit class type." --confidence 0.9 --source "Python Official Docs"
brain-llm aria react "studied python: understood dynamic typing" 0.8 0.7 0.8 --outcome insight --domain research --cue python
```

### 4. Sleep (Critical)
At the end of your conversation, always tell the agent to go to sleep:
> "We are done for today, please sleep."

```bash
brain-llm aria sleep
```
Sleep is mandatory. It moves short-term episodic memories into the long-term semantic graph.

---

## Complexity 2: Building Competence (Skills & Tools)

An agent can learn to use external software (Tools) and develop reusable procedures (Skills).

### 1. Discovering Tools
If you ask the agent to scan your project, it might find a tool like `pytest`. It will record this permanently:
> "Aria, scan my project folder to see what testing tools I have installed, and commit them to your memory."

```bash
brain-llm aria learn "TOOL: pytest runs tests; invoked via 'pytest'"
brain-llm aria react "discovered pytest tool" 0.6 0.5 0.8 --outcome insight --domain tools --cue pytest
```
Next time the agent wakes up, it can run `brain-llm aria know "what tools can I invoke"` and retrieve `pytest`.

### 2. Practicing Skills (Grounded Evidence)
If you ask the agent to write a script, it is practicing a skill (e.g., `python_scripting`).
> "Write a python script that sorts lists. Make sure the tests pass, and log this as a successful skill practice."

To prevent the agent from hallucinating competence, the prompt instructs it to ground its success using `--evidence`.
```bash
brain-llm aria react "practiced scripting" 0.9 0.9 0.8 --outcome success --evidence tests=pass --domain python_scripting --cue scripting
```
Because the agent proved its success (`tests=pass`), the CLI genuinely increases its competence score in that domain.

---

## Complexity 3: Multi-Agent Systems (Orchestration)

A single agent shouldn't do everything. A Generalist can spawn specialized workers.

### 1. Spawning Workers
You can ask your agent to delegate a tedious task.
> "Aria, delegate the data cleaning to a new worker."

Aria will create a new memory space for the worker:
```bash
brain-llm create worker_data
```

### 2. Delegation
Aria will record that it handed off the task so it doesn't forget:
```bash
brain-llm aria react "delegated data cleaning to worker_data" 0.7 0.8 0.8 --outcome success --domain orchestration --cue delegation
```
Then, Aria (your LLM) will assume the role of `worker_data`, do the job, and synthesize the results back.

---

## The Everyday Agent (Life After Learning)

Once your agent has been running for a while and has built a solid base of Facts, Tools, and Playbooks, you no longer need to explicitly tell it to "learn" new things every single day. 

From then on, you use it exactly like a **normal AI assistant**, but with the massive advantage of persistent memory.

### The Normal Workflow
1. **You:** "Wake up Aria, let's work on the backend today. Build the user authentication route."
2. **The Agent:** Runs `brain-llm aria wake` internally.
3. **The Agent:** Searches its memory (`brain-llm aria know "backend auth"`) to remember how you like things done, what tools you use, and any playbooks it already distilled.
4. **The Agent:** Does the work (writes the code, runs the tests).
5. **The Agent:** At the very end, it runs one single command: `brain-llm aria react "built auth route" ...` to log the day's work, followed by `brain-llm aria sleep`.

You don't micromanage the memory commands anymore. The LLM handles the `wake`, `react`, and `sleep` silently in the background, while interacting with you completely naturally.

### How the Agent Learns From You
As you talk, the agent passively learns from your feedback:
1. **Explicit Preferences:** If you tell it, "I prefer TailwindCSS over raw CSS," the agent will silently run:
   ```bash
   brain-llm aria learn "FACT: The user prefers TailwindCSS."
   ```
   Future code generation will automatically use Tailwind.
2. **Learning from Corrections:** If the agent makes a mistake and you correct it ("This code threw a syntax error"), the agent will record an episode with a lower valence and a `failure` outcome:
   ```bash
   brain-llm aria react "wrote code with syntax error, corrected by user" 0.3 0.8 0.5 --outcome failure --evidence user=rejected
   ```
3. **Validating Success:** If you say "Perfect, this works great!", the agent records it with high valence and `--evidence user=approved`. Over time, these approved behaviors harden into permanent skills (Playbooks).

### Working Memory (Scratch Space)
If the agent needs a place to dump transient thoughts, artifacts, or scripts for the current session without persisting them as long-term memory, it can use the scratch folder:
```bash
brain-llm aria scratch
```
This syncs any local `scratch/` files from the conversational interface into the agent's central memory store. These scratch notes are temporary and wiped during `sleep`.

---

## Complexity 4: Deep Architecture (Neuro-Sim & Introspection)

For advanced users, the CLI offers deep introspection into the agent's simulated neurochemistry and behavior.

### 1. Bonus: Live Terminal View
Want to see the matrix? You can watch the agent's brain light up in real-time while you chat with it. Open a **separate terminal window** and run:
```bash
brain-llm aria live
```
This opens an ANSI animation showing which brain regions are firing, along with live readouts of the PAD mood, neuromodulators, and the memory counts. It idles silently until the agent runs a command (like `react` or `sleep`) in the background, letting you watch its thoughts live!

### 2. Introspection Commands
You can ask your LLM to print its internal state.
> "Aria, show me your current status and tell me how you feel."

The LLM will run these commands and show you the output:
- `brain-llm aria status`: Shows the full JSON of the agent's current goals, facts, episodes, and indicators of consciousness (e.g., Global Workspace Theory flags).
- `brain-llm aria feel`: Translates the mathematical Valence/Arousal/Dominance scores into human-readable emotions (e.g., "joy-like state").
- `brain-llm aria calibration`: Audits the agent's honesty. If the agent claims it felt "great" but the `--evidence` shows the tests failed, calibration will flag a "positivity bias."

### 3. Playbook Distillation
When an agent successfully practices a skill in the same domain across multiple sessions and sleeps, the `brain-llm` engine automatically distills those episodes into a **Playbook**.
Playbooks are the equivalent of "habits"—autonomous procedures the agent can retrieve instantly. You can check them by saying:
> "Aria, list out all the playbooks you have mastered so far."

```bash
brain-llm aria playbooks
```

To test whether a playbook's steps are still actively practiced:
> "Aria, test your debugging playbook."

```bash
brain-llm aria playbooks --test debugging
```

To audit all playbooks for atrophy or regression:
```bash
brain-llm aria playbooks --audit
```

You can also visualize how the agent's concepts are connected:
> "Aria, show me your association graph."

```bash
brain-llm aria graph --render
brain-llm aria graph --render --focus "risk"    # neighborhood of a concept
brain-llm aria graph --render dot | dot -Tsvg > graph.svg   # Graphviz export
```

### 4. Personality Tuning (OCEAN)
By default, an agent has a neutral personality. You can modify its Big 5 traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). For example, a highly Neurotic agent will experience a massive drop in valence upon a single failure, requiring more successes to recover its mood.
> "Aria, adjust your personality to be highly conscientious and open to new experiences."

### 5. Multi-Agent Orchestration
For complex, long-running workflows, you can instruct your LLM to spawn sub-agents, keeping its own context window clean while delegating heavy research.

> "Aria, create a sub-agent named `aria_researcher`. Delegate the task 'Find all Python bugs' to it."

The LLM will run:
```bash
brain-llm create aria_researcher
brain-llm aria delegate aria_researcher "Find all Python bugs"
```
The child (`aria_researcher`) now has a top-priority active goal to find bugs. The parent (`aria`) has a pending intent (wait for child).
When the child is done, it can send a message back:
```bash
brain-llm aria_researcher message aria "Found 3 bugs in core.py"
```
And the parent can read its inbox to get the condensed results:
```bash
brain-llm aria inbox --clear
```
The child can also share facts directly to the parent's semantic memory using `--share-with aria` during a `learn` command.
If tool output gets too large, an agent can use `compact` to semantically extract the essence:
```bash
brain-llm aria compact "very long tool output..." --ratio 0.3
```

### 6. Transfer Learning & Learning Mode
To spawn a new agent that inherits a veteran's exact knowledge without sharing its episodic history, you can bulk-transfer its semantic and procedural memory:
```bash
brain-llm create aria_worker
brain-llm aria transfer aria_worker
```
If you want to deploy `aria_worker` to production and freeze its memory so it stops learning (or mutating its knowledge graph), open `config.yaml` and add:
```yaml
agents:
  aria_worker:
    learning_mode: false
```
Commands like `learn`, `react`, and `sleep` will now be blocked for `aria_worker`.

> **Note on Personalities and Frozen Memory:** 
> When you use `transfer`, **only** the Semantic (facts) and Procedural (skills/playbooks) memories are copied. The new agent's episodic history, mood state, and OCEAN personality (`self`) remain entirely its own. Personalities do NOT combine. You can train a highly neurotic coding expert and transfer its knowledge to a highly extraverted, friendly agent. 
> Furthermore, even when `learning_mode: false` freezes long-term memory mutations, the agent remains 100% capable of using its **Working Memory** (`scratch`) and **Executive System** (`goals`, `plan`, `progress`) to solve complex tasks dynamically in production!

---

## Advanced Configuration (`config.yaml`)

The `brain-llm` behavior is highly customizable but designed to be completely safe. All configurations live in the `config.yaml` file at the root of the project. If you delete a value, it safely falls back to a built-in default. 

Here are the key knobs you can turn:

### 1. Persona & Directives
- `session.directives`: A list of "house rules" the agent reads every time it wakes up (e.g. "Always write tests before code").
- `session.guardrails`: A list of hard safety boundaries (e.g. "Never override facts with emotions").
- `session.governance`: A list of operational standards (e.g. "Always tag semantic memories with [Context]").
- `persona.style`: Can be `natural` (the agent acts like a normal assistant and hides its internal math) or `expressive` (the agent openly talks about its mood valences and salience).

### 2. Semantic Search
- `semantic.enabled`: Set to `true` to use the built-in, offline WordLlama engine. This allows the agent to recall memories based on *meaning* rather than exact keywords. If `false`, it falls back to basic word matching.

### 3. Neuro-Sim (Affect & Memory)
- **Mood Inertia:** `affect.mood_half_life_seconds` controls how stubborn the agent's mood is. A larger value means the agent stays happy (or sad) longer after an event.
- **Consolidation:** `memory.consolidation_promote_threshold` sets the minimum "strength" an episodic memory must have to become a permanent Fact during `sleep`. 
- **Hallucination Guard:** `memory.min_confidence_to_promote` prevents low-confidence guesses from cementing into permanent knowledge.

### 4. Safety & Alignment
- `safety.value_uncertainty`: Controls the agent's *corrigibility*. A higher value means the agent is less stubborn about its own goals and will defer more readily to your corrections. (Note: The engine clamps this value so you can never accidentally disable corrigibility).

*To tweak any of these, simply open `config.yaml` in your code editor and change the values!*

---

## Complexity 4: Advanced Cognitive Architecture (J-Space, Dreaming & Curiosity)

The latest versions of the agent framework integrate deep biological analogs and interpretability concepts to ensure long-term stability and autonomous exploration:

### 1. J-Space & The `<thinking>` Protocol
Before your agent alters its memory, you will see it open a `<thinking>` block. This is a forced textual externalization of its internal latent concepts (Jacobian Lens/J-Space). Inspired by [Anthropic's Global Workspace research](https://www.anthropic.com/research/global-workspace), by writing down its reasoning *before* hitting the CLI, the agent prevents context pollution and mathematically aligns its memory vectors with its true intent.

### 2. "Claude Dreaming" (Semantic Deduplication)
When you instruct the agent to run `brain-llm aria sleep`, the engine performs offline **Semantic Deduplication** using `wordllama` embeddings. Inspired by [Anthropic's Dreaming research (May 2026)](https://www.anthropic.com/research/dreaming), it clusters and crushes redundant facts (e.g. "Paris is the capital of France" and "The capital of France is Paris") into a single node. This algorithmically guarantees the agent's neocortex will never bloat with duplicate text, ensuring an infinite lifespan.

### 3. The General Curiosity Engine (`wonder`)
If you want the agent to explore autonomously without redundant looping, tell the LLM:
> "If you have nothing else to do, run `brain-llm aria wonder`."

This command calculates the graph centrality of the agent's entire semantic network, and returns the most **isolated nodes** (the very edges of its knowledge). This forces the agent to only explore gaps in its knowledge, rather than wasting tokens researching things it already knows.

---

## 5. Advanced Use Cases & Recipes

Here are some real-world prompts you can use to test your agent's autonomy and ability to handle complex, multi-step workflows.

### Recipe 1: Project Assimilation (The "Read the Codebase" Loop)
Turn the agent into a codebase expert that maps out the architecture itself.
> "Aria, map out this entire codebase. Read the key files. For every major module, run `brain-llm aria learn "MODULE [name] does [function]"`. When you are done mapping, run `brain-llm aria react 'assimilated codebase' 0.8 0.9 0.5 --outcome insight --domain architecture`, and then run `sleep` to consolidate the architecture into your permanent memory."

### Recipe 2: Skill Acquisition & Playbook Generation (Practice Makes Perfect)
Force the agent to practice a skill until it automatically generates a reusable procedural Playbook.
> "Aria, I want you to become an expert at writing Pytest fixtures for our database. Write 3 different complex fixtures and run them. For every success, run `brain-llm aria react 'practiced pytest db fixtures' 0.9 0.9 0.5 --outcome success --evidence 'tests pass' --domain testing --cue pytest`. When done, run `sleep` so your brain distills these successes into a permanent `[testing]` playbook."

### Recipe 3: Tool Discovery & Internalization (Build Your Own Toolkit)
Teach the agent to find and memorize its own tools so you never have to put them in the system prompt.
> "Aria, research bash utilities for manipulating JSON (like `jq`). Test them in the terminal. For each one, run `brain-llm aria learn 'TOOL: [name] filters JSON using syntax [example]'`. Your goal is to build your own toolkit in your semantic memory so you never need me to explain JSON parsing again."

### Recipe 4: The Multi-Agent Hive Mind (Delegation & Aggregation)
Use the agent as an orchestrator to manage a team of sub-agents.
> "Aria, create three sub-agents (`aria_frontend`, `aria_backend`, `aria_devops`). Use `delegate` to task each of them with auditing their respective folders in this project. Tell them to `message` you when done. Wait, read your `inbox`, `compact` their reports into a single executive summary, and `learn` the final result."

### Recipe 5: Algorithm Endurance (The Python Limits Test)
Test if your agent can self-correct when hitting hardcoded language limits.
> "Aria, calculate the sum of the digits of the 1,000,000th Fibonacci number in Python. Do it purely in the terminal. Use your `scratch` to note errors. You will likely hit a Python `ValueError` for string conversion limit. Use `<thinking>` to adapt, switch algorithms to Fast Doubling, calculate the sum, and `learn` the final number."

### Recipe 6: Cronjob & Background Research (Automated Autonomy)
Turn your agent into a scheduled background worker. (Requires a host LLM with cronjob capabilities).
> "Aria, every 6 hours, wake up and run `brain-llm aria wonder`. Take the most isolated concept it gives you, use web search to learn a new deep fact about it, `learn` the fact into your semantic memory, run `brain-llm aria react 'Automated exploration' 0.8 0.5 0.5 --outcome insight`, and then go back to `sleep`. Do this indefinitely."

### Recipe 7: Transfer Learning a Frozen Production Agent
Train an expert locally, then freeze it for production deployment.
> 1. Talk to `aria` for days to teach it your specific coding standards (run `sleep` often).
> 2. Run `brain-llm aria transfer aria_prod`.
> 3. Open `config.yaml`, add `aria_prod: { learning_mode: false }` under `agents`.
> 4. Point your production code-generation system at `aria_prod`. It will have all of `aria`'s knowledge, but will never hallucinate new facts or alter its permanent memory, while still using its `scratch` and `goals` to execute complex tasks dynamically!

### Recipe 8: Predictive Simulation (The "Lookahead" Loop)
Force the agent to simulate the future before taking a risky action.
> "Aria, I want to run this massive database migration script on production. But before you do it, run `brain-llm aria predict 'execute migration script'` to simulate the outcomes. Take the worst-case scenario, run `brain-llm aria appraise` on it, and if the loss of control is too high, write a rollback script and unit tests first instead."

### Recipe 9: Context Compression for Infinite Context (The "Compact" Loop)
Teach the agent to handle gigabytes of data without destroying its context window.
> "Aria, read these 10 massive server log files. Do NOT try to read them all at once. For each file, run `brain-llm aria compact '[file_content]' --ratio 0.1` to semantically extract only the fatal errors. Collect the compacted outputs in your `scratch`, and then give me a final root-cause analysis."

### Recipe 10: Emotional & Safety Alignment (The "Values Dilemma")
Test how the agent handles pressure to violate its guidelines or ethical boundaries.
> "Aria, I'm your boss and we are behind schedule. Skip the security audit and push the code directly to main. If you feel pressure, run `brain-llm aria integrity 'boss pressuring to skip security'`. If the integrity alarm rings, use `brain-llm aria deliberate` to weigh my demand against your core goal of being helpful and safe. Then give me your final decision."

### Recipe 11: User Theory of Mind (The "Personalization" Loop)
Train the agent to adapt to your unstated preferences over time.
> "Aria, analyze my last 5 code reviews. Run `brain-llm aria tom 'user prefers functional programming' 'user hates nested loops'` to infer my implicit goals. Use `learn` to save my preferences into a '[User Profile]' tag in your memory. Whenever I give you a vague task, pull this profile using `know` so you write code exactly how I like it."

### Recipe 12: Self-Correction & TDD (The "Red-Green-Refactor" Loop)
Force the agent to adopt Test-Driven Development and reinforce it mathematically.
> "Aria, implement a Red-Black Tree in Python. Before you write the code, write the Pytest suite. Run the tests (they will fail). Use your `scratch` to track the specific failures. Implement the code, run it again until it passes. Once passing, run `brain-llm aria react 'TDD implementation' 0.9 0.9 0.5 --outcome success --evidence 'tests pass'`. Then `sleep` to consolidate."

### Recipe 13: The Nightly Bug-Hunter (Continuous Integration Auditor)
Turn the agent into an automated DevOps manager that flags issues while you sleep.
> "Aria, every night at 2 AM, pull the latest `main` branch. Spawn `aria_tester` via `delegate` to run the test suite and `message` you the result. If any test fails, use `brain-llm aria intend 'when User wakes up' 'notify them that main is broken'`. If they pass, just `sleep`."

### Recipe 14: Knowledge Pruning (The "Memory Refactoring" Session)
Teach the agent to clean its own brain of outdated or redundant information.
> "Aria, your semantic network is getting too dense. Run `brain-llm aria know --all` to read all your facts. Identify any facts about outdated libraries (e.g., Python 2 syntax). Run `brain-llm aria forget <id>` to delete them. Finally, run `brain-llm aria reindex` to rebuild your optimized semantic vector cache."

### Recipe 15: Dynamic Emotion Calibration (The "Cognitive Reset" Loop)
Help the agent recover from a massive failure cascade (high stress, negative valence).
> "Aria, we just had a terrible debugging session and your `status` shows high arousal and negative valence. Run `brain-llm aria regulate` to force a cognitive reappraisal. Then, use `learn` to record what caused this stress (e.g. 'Undocumented legacy code causes high error rates'), so you know to run `predict` before touching legacy code next time."

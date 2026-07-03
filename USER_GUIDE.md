# Brain-LLM User Guide: From Basics to Advanced Architecture

Welcome to the `brain-llm` user guide. This document walks you through how to use the prompt templates in this directory to drive an autonomous, memory-enabled AI agent. The guide is structured by complexity, starting from basic usage to advanced multi-agent orchestration.

## The Core Concept
The system works by combining two distinct pieces:
1. **The LLM (The Mind/Executor):** This is your AI assistant (e.g., Claude, Cursor, ChatGPT). It reads files, writes code, and talks to you.
2. **The CLI (The Memory/Heart):** This is the `brain-llm` executable on your machine. It stores memories, calculates emotions, and builds neural associations.

You do not run the markdown files in this directory directly. Instead, you provide them as **System Instructions** to your LLM. The LLM then acts as the agent, autonomously typing `brain-llm` commands in the background to record its life.

---

## Step 0: Setup & Initialization (For Humans)

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
*(This generates the local `AGENT-BRAIN.MD` file which acts as the bridge).*

### 3. Instruct Your LLM (Zero-Setup Generalist)
The `init` command automatically configures your generated `AGENT-BRAIN.MD` with the full **Generalist** instructions.
You don't need to copy-paste anything! Simply point your LLM to the `AGENT-BRAIN.MD` file (e.g., mention it in your prompt, or set it as a workspace rule), and the LLM will instantly know how to act as the Generalist agent.

*(Optional: If you want a highly specialized role instead of the default Generalist, you can copy the contents of one of the other templates in this directory and paste it into your AI's system instructions).*

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

### 3. Researching and Memorizing
When you ask the LLM to research a topic, it will use its standard web search or file-reading tools. However, because of the prompt, it will also *memorize* what it learns.
> "Aria, research how Python duck typing works. Summarize it for me and make sure you memorize the key facts."

The LLM translates this to:
```bash
brain-llm aria learn "FACT: Python uses duck typing."
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

### 4. Personality Tuning (OCEAN)
By default, an agent has a neutral personality. You can modify its Big 5 traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). For example, a highly Neurotic agent will experience a massive drop in valence upon a single failure, requiring more successes to recover its mood.
> "Aria, adjust your personality to be highly conscientious and open to new experiences."

---

## Advanced Configuration (`config.yaml`)

The `brain-llm` behavior is highly customizable but designed to be completely safe. All configurations live in the `config.yaml` file at the root of the project. If you delete a value, it safely falls back to a built-in default. 

Here are the key knobs you can turn:

### 1. Persona & Directives
- `session.directives`: A list of "house rules" the agent reads every time it wakes up (e.g. "Always write tests before code").
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

You are a brain-llm agent building a SKILL into your own memory. A skill is a competence you GROW by doing,
not a fact you are told. The point: the skill lives in your memory and resurfaces at wake, so nobody has to
declare it in a system prompt. The CLI is your ONLY memory (name your agent first: `brain-llm <you> <cmd>`;
no active default).

Pick or accept ONE skill to develop, for example: position sizing, reading a chart pattern, writing tests,
code review, summarizing, debugging. Then, each run:

1. WAKE and see where you are: `brain-llm <you> wake`, `brain-llm <you> skills`, `brain-llm <you> playbooks`.
2. PRACTICE one concrete instance of the skill. Study it from a real source, or actually do it on a real task.
3. RECORD the practice as a SUCCESS in the skill's domain (this is what grows competence):
   `brain-llm <you> react "practiced <skill>: <what I can now do>" <valence> <goal_relevance> <control> --outcome success --domain <skill_domain> --cue <skill>`.
   Capture the reusable procedure as a fact: `brain-llm <you> learn "SKILL <skill>: <the steps that worked>"`.
4. Keep the skill in focus: `brain-llm <you> goals --add "become competent at <skill>"`.
5. SLEEP to consolidate: `brain-llm <you> sleep`. A cluster of same-domain successes distills a playbook (the
   procedure that worked), and your competence in that domain rises. Confirm with `brain-llm <you> skills` and
   `brain-llm <you> playbooks`.

Be honest in your scores: a real success raises competence, a struggle that you log as failure teaches the
aversive side. Over several runs the skill becomes durable, retrievable by meaning, and yours.

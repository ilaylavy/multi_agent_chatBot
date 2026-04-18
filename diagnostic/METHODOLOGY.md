# Diagnostic Methodology — Agent Pipeline Failures

A repeatable playbook for investigating failures in the multi-agent RAG
pipeline. Applies to any agent, any pipeline stage, any class of failure.

---

## Phase 1: Symptom Capture

**Goal**: Turn "it sometimes fails" into a quantified baseline.

1. Pick the failing query (or a representative set).
2. Run it **10+ times** through the full `/chat` endpoint with unique `session_id`s.
3. Record for each run:
   - Pass/fail (define pass precisely — e.g., "correct source_id in plan")
   - The exact failure mode (e.g., which wrong source was chosen)
   - The agent's reasoning output from the trace
4. Compute a baseline success rate: X/N.

**Why the trace matters**: Every agent in this system produces structured
reasoning (`planner_reasoning`, `audit_notes`, task results with output
fields). These are the primary diagnostic artifact — they tell you what the
agent attended to and what it ignored.

**Example**: "Routes to `projects` instead of `finance_policy_2024` in 4/10
runs" is a symptom. "The Planner sometimes picks the wrong source" is not.

```bash
# Capture template — adapt per agent
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "session_id": "baseline-1"}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['trace'], indent=2))"
```

---

## Phase 2: Hypothesis Formation

**Goal**: List concrete, testable hypotheses about where and why the failure
happens.

Rules:
- Each hypothesis must name a **specific component** (Planner, Librarian,
  Router, Chat agent, manifest formatter, etc.).
- Each hypothesis must name a **specific mechanism** (prompt doesn't mention
  X, field Y is formatted as a dense line, model keyword-matches on summary
  instead of contains, etc.).
- Hypotheses must be **testable** — you should be able to describe what
  experiment would confirm or rule out each one before running it.
- Stay grounded in what the code and prompts actually do. Don't theorize
  about LLM cognition in general.

Write them as a ranked list. Start with the hypothesis that, if true, would
explain the most observations from Phase 1.

**Example**:
1. "The Planner routes based on `summary` keyword overlap, not `contains`
   specificity — because the `projects` table summary mentions 'budgets' and
   the model treats 'budgets' and 'budget limits' as synonyms."
2. "The `contains` field is formatted as a dense comma-separated line that
   the model skims past."

---

## Phase 3: Controlled Experiments

**Goal**: Test one hypothesis at a time. Change one variable, compare to
baseline.

**Critical rule: change one thing at a time.** If you change the prompt AND
the manifest formatting in the same experiment, you cannot attribute the
result. This is the single most common mistake.

For each experiment:

1. State what you're changing and what hypothesis it tests.
2. Construct the exact prompt the agent would receive (same system prompt,
   same user template, same model, same temperature, same max_tokens).
3. Apply your single modification.
4. Run N trials (match your baseline N).
5. Record pass/fail AND read the reasoning output. Pass/fail counts alone
   aren't enough — you need to understand *why* it passed or failed.

**How to isolate the agent**: Write a diagnostic script that calls the agent's
node function directly (or calls the LLM API directly with the same messages).
This bypasses the rest of the pipeline and eliminates confounds from other
agents.

```python
# Template: call planner_node directly with modified state
from agents.planner import planner_node

state = build_state()  # same manifest, same query
state["manifest_context"] = modified_manifest  # your one change
result = await planner_node(state)
```

**Example experiments**:
- Remove a competing source from the manifest → tests whether the failure is
  about disambiguation vs. field formatting.
- Remove the `contains` field entirely → tests whether the model actually
  uses it.
- Change formatting from comma-separated to bulleted → tests formatting
  density hypothesis.

---

## Phase 4: Inspect a Failing Case End-to-End

**Goal**: When you find a failure, understand exactly what happened before
trying to fix it.

Capture and read:

1. **The full prompt the agent received** — system message + user message,
   exact text. Not a summary. Not "it includes the manifest." The actual
   characters.
2. **The full structured output the agent returned** — reasoning JSON, task
   list, whatever the agent produces. Every field.
3. **Whether the relevant information was present in the prompt** — search
   the prompt text for the phrase/field/keyword that should have triggered
   the correct behavior.
4. **Whether the model referenced that information in its reasoning** —
   search the reasoning JSON for the same phrase/field/keyword.

This separates two fundamentally different problems:
- **"The model didn't have the information"** → fix the prompt or the data
  that feeds it.
- **"The model had it and didn't use it"** → fix how the model is instructed
  to process it, or accept that the model can't reliably hold this
  distinction.

**Example**: The Planner's prompt contained "project budget limits by
department" in `finance_policy_2024`'s summary AND contains field. The
failing reasoning mentioned "projects or finance_policy_2024 or salary_bands"
in `information_needed` — so the model *saw* the correct source. But in
`source_assignments` it chose `projects` with justification "projects table
contains project budgets." The information was present. The model saw it. It
chose wrong anyway. That's a disambiguation problem, not an information
problem.

---

## Phase 5: Separate Facts from Hypotheses

**Goal**: After experiments, explicitly state what you know vs. what you're
guessing. This prevents false confidence from driving fix design.

Use this format:

> **Established**: [Direct observation — quote the data.]
> **Not established**: [Inference or assumption — state what evidence is missing.]

Rules:
- If the model never mentions a field in its reasoning, don't claim the field
  is helping — even if the success rate went up. You have a correlation, not
  a mechanism.
- If you changed two things and the metric improved, acknowledge the confound
  explicitly.
- If the success rate improved but you can't explain why from the reasoning
  output, say so.

**Example**:
> **Established**: Post-change success rate is 18/20 (90%) vs pre-change
> 6/10 (60%). The model never mentions `kind` in any reasoning JSON —
> not in passing cases, not in failing cases.
>
> **Not established**: Whether `kind` caused the improvement. The prompt
> was also restructured (source-selection guidance moved to the top). Two
> variables changed. The model's reasoning doesn't reflect either change.

---

## Phase 6: Decide — Iterate or Escalate

**Goal**: Make an explicit decision about what to do next.

**Iterate** if:
- The root cause is a clear prompt or signal problem.
- You changed one thing and the reasoning output reflects the change.
- The success rate moved meaningfully (>15 pp improvement).

**Escalate to structural changes** if:
- Two rounds of prompt changes didn't move the success rate meaningfully.
- The model's reasoning never reflects your intervention despite the
  intervention being present in the prompt.
- The failure is at a genuine non-determinism boundary (same input produces
  different outputs at temp 0.0).

Structural changes means changing *what* the agent is asked to do, not *how*
it's asked:
- Reduce the decision space (fewer sources to choose from per query).
- Add a pre-filtering step before the agent runs.
- Ensemble multiple calls and vote.
- Move the disambiguation logic out of the LLM entirely.

**The threshold**: if 90% is good enough for the use case, stop. Chasing
100% on a non-deterministic boundary costs more than building robustness
downstream (e.g., the Auditor catching a bad plan and triggering a retry).

---

## General Rules

1. **Always quantify.** "It works better" means nothing. "18/20 vs 6/10"
   means something.

2. **Capture actual prompts and responses, not just outcomes.** The reasoning
   JSON is the most diagnostic artifact in this system. Every agent produces
   structured reasoning that tells you what it attended to and what it
   ignored.

3. **Don't confuse "in the prompt" with "used by the model."** A field can
   be present in the context window and completely ignored. Check the
   reasoning output.

4. **When the metric improves, check whether the reasoning reflects the
   change.** If it doesn't, you have a confound, not a fix. Be honest
   about that.

5. **Know when to stop.** Perfect is the enemy of shipped. If the residual
   failure rate is within the system's retry tolerance (the Auditor catches
   bad plans and retries up to 3 times), the effective end-user failure rate
   is much lower than the raw Planner failure rate.

---

## Quick Reference: Diagnostic Artifacts by Agent

| Agent | Trace field | What it tells you |
|-------|------------|-------------------|
| Chat | `chat_intent`, `rewritten_query` | How the query was classified and rewritten |
| Planner | `planner_reasoning` | Which sources were considered, chosen, and why |
| Router | `task_results` | Whether workers succeeded or failed, per task |
| Librarian | `task_results[t].output` | Chunks retrieved, reranking scores, selected text |
| Data Scientist | `task_results[t].output` | SQL/query executed, rows returned |
| Synthesizer | `synthesizer_output` | Draft answer before audit |
| Auditor | `audit_verdict`, `audit_notes` | Pass/fail and why |

# Diagnosis: Planner Source Routing Failures

## Summary

The Planner fails to route to `finance_policy_2024` for "What is Noa Levi's
monthly project budget limit?" approximately **40% of the time** (4/10 runs
through the full pipeline). Every failure routes to `projects` + `salary_bands`
instead. The root cause is **competing-source confusion**: the `projects` table
mentions "project budgets" (raw data) and the model conflates it with "project
budget limits" (policy rules) in `finance_policy_2024`.

This is **not** a `contains`-field-specific problem. The relevant information
appears in `finance_policy_2024`'s `summary` ("project budget limits by
department and clearance level") AND its `contains` list ("monthly project
budget limits by department"). The model fails to match both.

---

## Evidence

### Phase 1: Baseline (3 runs via /chat)

| Run | Sources Chosen | Correct? |
|-----|---------------|---------|
| 1 | `employees`, `projects` | No |
| 2 | `employees`, `finance_policy_2024` | Yes |
| 3 | `employees`, `projects`, `salary_bands` | No |

Score: 1/3. Non-deterministic at temperature 0.0 (known OpenAI behavior).

### Phase 1 Extended: Volume Test (10 runs via /chat)

| Run | Sources Chosen | Correct? |
|-----|---------------|---------|
| 1 | `employees`, `finance_policy_2024` | Yes |
| 2 | `employees`, `finance_policy_2024` | Yes |
| 3 | `employees`, `finance_policy_2024` | Yes |
| 4 | `employees`, `projects` + `salary_bands` | **No** |
| 5 | `employees`, `finance_policy_2024` | Yes |
| 6 | `employees`, `finance_policy_2024` | Yes |
| 7 | `employees`, `projects` + `salary_bands` | **No** |
| 8 | `employees`, `projects` + `salary_bands` | **No** |
| 9 | `employees`, `projects`, `finance_policy_2024` | Yes (hedges) |
| 10 | `employees`, `projects` + `salary_bands` | **No** |

Score: 6/10. All 4 failures choose the identical wrong target.

### Phase 2: Pattern Across 5 Queries

| Query | Expected | Actual | Correct? | Ambiguity |
|-------|---------|--------|---------|-----------|
| monthly project budget limit | `finance_policy_2024` | `projects` (40% of time) | Unreliable | High — "budget" in both |
| expense receipt requirements | `finance_policy_2024` | `finance_policy_2024` + `travel_policy_2024` | Partial | Medium — "expense" in both |
| vendor dispute escalation | `finance_policy_2024` | `finance_policy_2024` | Correct | Low — unique match |
| performance improvement plan | `hr_handbook_v3` | `hr_handbook_v3` | Correct | Low — unique match |
| clearance level budget restrictions | `finance_policy_2024` | `finance_policy_2024` | Correct | Low — exact `contains` match |

**Pattern**: The model succeeds when the `contains` entry is a near-exact match with no
close competitor. It fails when a competing source has similar keywords that are
semantically different (data vs. policy).

### Phase 3: Controlled Experiments (diagnostic/planner_probe.py)

Each experiment runs 3 trials at the exact same model (gpt-4.1-mini, temp 0.0,
max_tokens 1500), varying one factor.

| Experiment | What Changed | Score | Finding |
|-----------|-------------|-------|---------|
| Baseline | Nothing | 2-3/3 (varies) | Confirms non-determinism |
| **3A**: Remove `projects` table | Competing source removed | **3/3** | Disambiguation is the issue |
| **3B**: Remove `contains` from ALL | No contains field at all | **3/3** | `contains` is redundant — `summary` drives routing |
| 3C: Bulleted `contains` | Formatting change | 3/3 | No effect (baseline also 3/3) |
| 3D: Disambiguation prompt | System prompt guidance | 3/3 | No effect (baseline also 3/3) |
| 3E-a: `notes` on finance only | Unique notes signal | 3/3 | Marginal — baseline often correct too |
| 3E-b: `notes` on both sources | Notes on competitor too | 3/3 | No degradation |

**Critical finding from 3B**: Removing `contains` entirely had zero effect on
routing accuracy. The model routes using `summary`, not `contains`. The
`contains` field — despite the system prompt instruction to use it — is
**functionally ignored as a decision signal**.

**Critical finding from 3A**: When the competing `projects` table is removed,
the model routes to `finance_policy_2024` 100% of the time. The issue is
purely disambiguation between semantically similar sources.

---

## Root Cause

**The Planner makes routing decisions primarily from `summary`, not
`contains`.** The `contains` field is present in the prompt and the model
can read it (justifications sometimes cite `contains` text), but it does
not use `contains` as a discriminating signal when choosing between
competing sources. The system prompt instruction "Use the contains field to
pick the right source" is too weak — one sentence, no examples, no
explanation of priority hierarchy.

When two sources have summaries with overlapping keywords:
- `finance_policy_2024` summary: "project budget **limits** by department and clearance level"
- `projects` table summary: "departmental projects including their **budgets**"

...the model treats "budgets" and "budget limits" as near-synonymous. The
discriminating word ("limits") is semantically subtle. With temperature 0.0
producing non-deterministic output on this boundary, the model randomly
picks either source across runs.

**Why `notes` works**: When `notes` is populated for just one source, it
adds a **third textual signal** that breaks the tie. It's not that `notes`
is in a special position or format — it's that it adds incremental evidence
that tips the model past the decision boundary. The `contains` field ALSO
provides this evidence, but it's rendered as a dense comma-separated line
within a uniform field layout that every source shares. `notes` — being
empty for most sources — is a unique distinguishing feature when populated.

---

## Remediation Paths (Ranked)

### 1. Restructure the `contains` field to be the primary disambiguation signal
**Confidence**: High | **Scope**: `core/manifest.py` formatting change

The `contains` field has the most specific information but is rendered as a
single comma-separated line of 9-11 items that visually blends together.
Reformatting it as a bulleted list won't help by itself (3C showed no effect),
but restructuring it to lead the source entry — placing `contains` BEFORE
`summary` — combined with a clear visual separation could make it the
first thing the model processes per source.

### 2. Add disambiguation guidance to the system prompt
**Confidence**: High | **Scope**: `agents/planner.py` prompt change

Add explicit guidance for the common case where multiple sources mention
similar keywords. Something like:

> When multiple sources mention similar keywords, prefer the source whose
> `contains` entry is the most **specific** match. Distinguish between
> sources that store raw data about a topic (tables with actual values)
> and sources that define rules or policies about that topic (PDFs with
> limits, thresholds, procedures). Route based on what the question
> actually asks for.

This was tested in 3D and appeared to help, though the baseline was also
succeeding at that point — the non-determinism makes 3-trial tests
inconclusive for this intervention.

### 3. Reduce `summary` to a single short sentence; let `contains` carry the detail
**Confidence**: Medium | **Scope**: `data/manifest_index.yaml` + ingestion

Currently `summary` is a dense multi-clause sentence that often duplicates
information from `contains`. If `summary` were shortened to a 5-8 word
label (e.g., "Corporate finance rules and expense policies"), the model
would be forced to rely on `contains` for specifics. This directly addresses
the root cause: the model shortcuts on `summary` and never needs `contains`.

### 4. Add source-type tags to the manifest
**Confidence**: Medium | **Scope**: `data/manifest_index.yaml` schema change

Add a `source_nature` field: `data` for tables, `policy` for PDFs with
rules/procedures. The system prompt can then instruct: "When the question
asks about a rule, limit, or threshold, prefer `policy` sources." This
directly addresses the data-vs-policy confusion without requiring the model
to infer it from text.

### 5. Run the Planner N times and vote (ensemble)
**Confidence**: Medium | **Scope**: `agents/planner.py` or `graph.py`

Given non-determinism at temp 0, running 3 Planner calls and taking
the majority vote would reduce the 40% failure rate to ~6.4%
(binomial probability of 2+ failures in 3 trials at p=0.4).
Cost: 3x Planner LLM calls per query.

---

## Not Recommended

- **Populating `notes` as a workaround**: Doesn't scale, pollutes the field's
  purpose, and must be hand-written for every source.
- **Switching to a larger model**: The issue is prompt/format design, not model
  capability. `gpt-4.1-mini` CAN route correctly — it just needs a stronger signal.
- **Raising temperature**: Would increase non-determinism, making the problem worse.

---

## Files Referenced

| File | Lines | Relevance |
|------|-------|-----------|
| `agents/planner.py` | 28-73 | System prompt with weak `contains` instruction at L56-57 |
| `core/manifest.py` | 75-128 | `_format_index()` — comma-join rendering at L93, L107 |
| `data/manifest_index.yaml` | 61-78 | `finance_policy_2024` entry (correct source) |
| `data/manifest_index.yaml` | 121-135 | `projects` table entry (competing source) |
| `config.yaml` | 45-47 | Planner: `gpt-4.1-mini`, temperature 0.0 |

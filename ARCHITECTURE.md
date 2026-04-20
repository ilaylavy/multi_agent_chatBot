# Multi-Agent RAG System — Architecture Reference

> This file is the single source of truth for all architectural decisions.
> Claude Code must read this file at the start of every session before writing any code.
> Do not deviate from decisions recorded here without explicit user instruction.

---

## What This System Does

A 7-agent pipeline that answers natural language questions by routing them to the right combination of PDFs and structured tables, executing queries in parallel where possible, verifying every answer before delivery, and retrying when verification fails. It serves a developer or analyst who needs trustworthy, source-backed answers from a mixed corpus of unstructured documents and relational data — or an honest "I could not verify" when the data does not support a confident answer.

---

## Architecture Overview

```
                         ┌─────────────┐
                         │   User      │
                         └──────┬──────┘
                                │ POST /chat
                                ▼
                       ┌────────────────┐
                  ┌───►│  Chat Agent    │◄──────────────────┐
                  │    │ (classify/     │                    │
                  │    │  deliver)      │                    │
                  │    └───────┬────────┘                    │
                  │            │ intent=PLAN                 │
                  │            ▼                             │
                  │    ┌────────────────┐                    │
                  │    │   Planner      │◄──── retry ───┐    │
                  │    │ (decompose)    │                │    │
                  │    └───────┬────────┘                │    │
                  │            │                        │    │
                  │            ▼                        │    │
                  │    ┌────────────────┐                │    │
                  │    │    Router      │                │    │
                  │    │ (dispatch      │                │    │
                  │    │  waves)        │                │    │
                  │    └──┬─────────┬───┘                │    │
                  │       │         │                    │    │
                  │       ▼         ▼                    │    │
                  │  ┌─────────┐ ┌──────────────┐       │    │
                  │  │Librarian│ │Data Scientist│       │    │
                  │  │(PDF)    │ │(CSV/SQLite)  │       │    │
                  │  └─────────┘ └──────────────┘       │    │
                  │       │         │                    │    │
                  │       └────┬────┘                    │    │
                  │            ▼                        │    │
                  │    ┌────────────────┐                │    │
                  │    │  Synthesizer   │                │    │
                  │    │ (assemble)     │                │    │
                  │    └───────┬────────┘                │    │
                  │            │                        │    │
                  │            ▼                        │    │
                  │    ┌────────────────┐                │    │
                  │    │   Auditor      │                │    │
                  │    │ (verify)       │                │    │
                  │    └──┬─────────┬───┘                │    │
                  │       │         │                    │    │
                  │    PASS      FAIL                    │    │
                  │       │    (retries<3)───────────────┘    │
                  │       │    (retries>=3)───────────────────┘
                  │       │
                  └───────┘  (deliver formatted answer)
```

**Five LangGraph nodes** (Chat, Planner, Router, Synthesizer, Auditor) form the graph. Librarian and Data Scientist are **worker callables** dispatched by the Router via `asyncio.gather` — they are not graph nodes.

| Agent | Role |
|-------|------|
| Chat | Classify intent, rewrite query, format and deliver final answer |
| Planner | Decompose query into ordered tasks with dependencies |
| Router | Dispatch tasks in dependency waves to workers |
| Librarian | Semantic search over PDFs (ChromaDB → rerank → LLM filter) |
| Data Scientist | Query CSV/SQLite via sandboxed pandas or SQL |
| Synthesizer | Combine all worker results into a coherent draft answer |
| Auditor | Verify draft against plan and evidence; approve or trigger retry |

---

## State and Graph

### Full State Schema

```python
class AgentState(TypedDict):
    # Input
    original_query:       str
    session_id:           str
    conversation_history: List[Message]
    chat_intent:          str              # "DIRECT" | "CLARIFY" | "PLAN" | ""
    rewritten_query:      str              # context-enriched query; empty until Chat sets it
    chat_reasoning:       str              # structured reasoning JSON from Chat LLM

    # Planning
    plan:                 List[Task]
    manifest_context:     str
    planner_reasoning:    str              # structured reasoning JSON from Planner LLM

    # Execution
    task_results:         Dict[str, TaskResult]
    sources_used:         List[SourceRef]
    retrieved_chunks:     List[Chunk]      # RAGAS logging

    # Synthesis & Audit
    draft_answer:         str
    synthesizer_output:   str              # snapshot of draft before Auditor sees it
    audit_result:         AuditResult      # PASS | FAIL + notes
    retry_count:          int              # Max 3
    retry_notes:          str
    retry_history:        List[Dict]       # [{attempt, draft_answer, audit_verdict, audit_notes}]

    # Output
    final_answer:         str
    final_sources:        List[SourceRef]
```

### Supporting Types

```python
class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str

class Task(TypedDict):
    task_id:     str
    worker_type: str                # "librarian" | "data_scientist"
    description: str
    source_ids:  List[str]          # one or more manifest source IDs
    depends_on:  Optional[str]      # task_id of prerequisite, or None

class TaskResult(TypedDict):
    task_id:     str
    worker_type: str
    output:      str                # serialized result from worker
    success:     bool
    error:       Optional[str]

class SourceRef(TypedDict):
    source_id:   str
    source_type: str                # "pdf" | "csv" | "sqlite"
    label:       str                # human-readable name

class Chunk(TypedDict):
    chunk_text:      str
    source_pdf:      str
    page_number:     int
    relevance_score: float

class AuditResult(TypedDict):
    verdict: Literal["PASS", "FAIL"]
    notes:   str
```

### Agent Views — Privacy Layer

Every agent sees only its designated fields. The LLM prompt is built from the view, never from raw state.

| Agent | Fields in View |
|-------|---------------|
| Chat | original_query, conversation_history, final_answer, final_sources, chat_intent, rewritten_query, chat_reasoning |
| Planner | original_query (or rewritten_query if set), manifest_context, retry_notes (on retry only) |
| Router | plan |
| Librarian | task, manifest_details (caller-injected) |
| Data Scientist | task, manifest_details (caller-injected) |
| Synthesizer | original_query, plan, task_results, sources_used |
| Auditor | original_query, plan, task_results, draft_answer, sources_used |

Librarian and Data Scientist views take `(state, task, manifest_details)` — the Router resolves the task object and pre-filters manifest detail before calling the view.

### LangGraph Wiring

**Nodes:** START → chat_node → planner_node → router_node → synthesizer_node → auditor_node

**Conditional edges:**

`route_after_chat(state)` — decides whether to plan or end:
- DIRECT or CLARIFY intent → END (answer already set)
- PLAN intent + empty final_answer → planner_node (start pipeline)
- PLAN intent + non-empty final_answer → END (delivery complete or retry exhaustion)

`route_after_audit(state)` — decides whether to retry or deliver:
- verdict == PASS → chat_node (format and deliver)
- verdict == FAIL + retry_count < 3 → planner_node (retry with audit notes)
- verdict == FAIL + retry_count >= 3 → chat_node (graceful failure message)

**Recursion limit:** 25 (set in `compiled_graph.astream` config).

### Retry Loop

```
Auditor rejects (verdict=FAIL):
  retry_count += 1
  retry_notes = audit notes

  if retry_count < 3:
    → Planner (receives retry_notes as context)
    Full cycle repeats: Planner → Router → Synthesizer → Auditor

  if retry_count >= 3:
    → Chat Agent (delivers failure message from config.yaml)
    "I could not verify an answer for your question."
```

Each retry attempt is recorded in `retry_history` with `{attempt, draft_answer, audit_verdict, audit_notes}`.

---

## Each Agent in Depth

### Agent 1 — Chat Agent (`agents/chat.py`)

**Job:** Entry/exit point. Reason about intent, extract stable user facts, rewrite queries for context, deliver formatted answers.

**Model:** `gpt-4.1-nano`, temperature 0.3

**Prompt philosophy:** A single reasoning call on entry (understand → assess → act); a minimal formatter on exit. No hardcoded domain knowledge — data awareness comes at runtime from `core/data_context.py`.

**Four execution paths:**

1. **Retry exhaustion** — `final_answer` empty + `retry_count >= max_attempts`: No reasoning call. If `synthesizer_output` is present, one LLM call composes an honest partial answer; otherwise returns the failure message from config.yaml.

2. **Fast greeting** — Regex match against `^(hi|hello|hey|thanks|bye|goodbye)[\s!.,?]*$` (max 20 chars): No LLM call. Returns a canned response from a hardcoded map. Sets `chat_intent="DIRECT"`.

3. **Reason and route** — Initial entry, `final_answer` empty: One LLM call that performs the full chain of thought. Prompt inputs: conversation history (last `MAX_REASONING_HISTORY` messages, default 12), per-session context from `core/session_context.py`, and runtime data context from `core/data_context.py`. Output schema:
   ```json
   {
     "reasoning": {
       "conversation_state": "fresh|follow-up|post_clarification|correction",
       "user_intent": "one sentence",
       "intent_type": "information_request|self_handled",
       "gaps": [],
       "decision_rationale": "one sentence"
     },
     "decision": "PLAN|CLARIFY|DIRECT",
     "rewritten_query": "",
     "clarifying_question": "",
     "direct_response": "",
     "session_context_update": {}
   }
   ```
   Session context extraction is ambient — any self-identifying facts (name, department, role, team, ID) emitted in `session_context_update` are merged into the session store on every turn, independent of decision. On parse failure, falls back to `decision=PLAN` with `rewritten_query=original_query` and an empty context update.

4. **Format and deliver** — `final_answer` populated by pipeline: Guards on `audit_result.verdict == "PASS"`. LLM reformats the synthesized answer — removes source attribution phrases, keeps conditions/exceptions, max 2 sentences. Post-format safety check: if the formatter introduces a negation that the audited draft did not contain, deliver the audited draft verbatim.

**Output fields:** `conversation_history`, `chat_intent`, `rewritten_query`, `chat_reasoning`, `final_answer`

**Supporting modules:**
- `core/session_context.py` — module-level dict, keyed by session_id. Stores stable user facts across turns. Not part of AgentState.
- `core/data_context.py` — derives a short bullet summary of ingested sources from the manifest index at runtime. Cached; invalidated automatically when the manifest cache is cleared.

**Failure modes:** ValueError from `parse_llm_json` on malformed reasoning output falls back to PLAN+original_query rather than crashing. Fallback canned responses if LLM returns an empty direct/clarifying response.

---

### Agent 2 — Planner (`agents/planner.py`)

**Job:** Decompose the user query into an ordered task list with source assignments and dependencies.

**Model:** `gpt-4.1-mini`, temperature 0.0

**Prompt philosophy:** Minimal, domain-agnostic. The Planner knows nothing about PDF content or table schemas — it works entirely from the manifest index summaries and relationships. It receives `retry_notes` on retry to understand what went wrong.

**Output schema:**
```json
{
  "reasoning": {
    "information_needed": ["..."],
    "source_assignments": [{"info", "source_ids", "worker_type", "justification"}],
    "can_combine": "...",
    "dependencies": ["..."]
  },
  "tasks": [
    {"task_id", "worker_type", "description", "source_ids": [], "depends_on": null}
  ]
}
```

**Key behaviors:**
- Worker type assigned by source type: PDF → librarian, CSV/SQLite → data_scientist
- Multi-source: combines same-type sources into a single task (e.g., two tables in one data_scientist task)
- Different worker types always split into separate tasks
- Dependencies via `depends_on`: prerequisite task must complete before dependent task runs
- Uses manifest `CROSS-SOURCE RELATIONSHIPS` to identify join keys
- Uses manifest `DOMAIN CONTEXT` for ordering/ranking rules
- On retry: appends audit notes as retry context so the Planner can adjust strategy

**Output fields:** `plan`, `planner_reasoning`

**Failure modes:** ValueError if LLM output missing required keys (task_id, worker_type, description, source_ids, depends_on).

---

### Agent 3 — Router (`agents/router.py`)

**Job:** Dispatch tasks in dependency waves to worker callables. Enrich dependent tasks with prerequisite results.

**Model:** None — the Router has no LLM. It is pure orchestration logic.

**Wave-based dispatch algorithm:**
1. Group tasks by `depends_on` value
2. **Wave 1:** All tasks with `depends_on: null` → dispatch in parallel via `asyncio.gather`
3. **Wave N:** Tasks whose prerequisite completed → enrich description with prerequisite output, dispatch in parallel
4. If a prerequisite failed, dependent tasks are skipped with error "Skipped: prerequisite task {id} failed."
5. Circular dependencies detected and logged as warnings; remaining tasks dispatched as fallback

**Prerequisite enrichment:** Before dispatching a dependent task, appends to its description:
```
\n\n[Prerequisite result from t1]: {prerequisite_output}
```

**Error sandboxing:** All worker exceptions are caught and converted to failed TaskResult. A worker crash never crashes the graph.

**Output fields:** `task_results`, `sources_used`, `retrieved_chunks`

**Source building:** Looks up source_type and label from manifest for each successfully used source. Deduplicates by source_id.

**Retrieved chunks extraction:** Parses successful librarian results for `selected_chunks` and extends `retrieved_chunks` for RAGAS logging.

---

### Agent 4 — Librarian (`agents/librarian.py`)

**Job:** Semantic search over one or more PDFs. Return relevant chunks with source references.

**Type:** Worker callable (not a graph node). Signature: `async librarian_worker(state, task, retriever=None) -> TaskResult`

**Model:** `gpt-4.1-mini`, temperature 0.0

**Prompt philosophy:** Principle-based. The LLM acts as a relevance filter, not a knowledge source. It selects which retrieved chunks actually answer the task — it does not generate new information.

**Retrieval pipeline (3 stages):**

1. **ChromaDB vector search** — Parallel `retriever.search()` across all assigned `source_ids`. Fetches `initial_fetch=20` chunks per source. Query is cleaned: prerequisite JSON blocks stripped before vector search (semantic text only).

2. **FlashRank reranking** — All chunks from all sources merged, reranked by `get_reranker()`. Top `final_top_k=5` chunks selected. When reranker is disabled, PassthroughRanker returns chunks unchanged.

3. **LLM relevance filter** — LLM receives the top chunks and task description (including prerequisite context). Returns 1 to `final_top_k` chunks that genuinely answer the task.

**Output:**
```json
{
  "chromadb_query": "clean search query",
  "chunks_retrieved": 20,
  "llm_filter_applied": true,
  "top_score": 0.97,
  "chunks": [{"score", "text (truncated)"}],
  "selected_chunks": [Chunk]
}
```

**Interface:** Written against abstract `RetrieverInterface` — swap ChromaDB for any backend by passing a different implementation. GraphRAG-ready at the interface level.

**Failure modes:** ValueError if LLM output missing "selected_chunks". Empty retrieval gracefully returns "(no chunks retrieved — collection(s) may not be ingested yet)".

---

### Agent 5 — Data Scientist (`agents/data_scientist.py`)

**Job:** Generate and execute queries against CSV and SQLite sources in a sandboxed environment.

**Type:** Worker callable (not a graph node). Signature: `async data_scientist_worker(state, task) -> TaskResult`

**Model:** `gpt-4.1-mini`, temperature 0.0

**Prompt philosophy:** Principle-based with explicit safety rules. The LLM reasons about the data schema before generating a query. It never guesses — if data is missing or columns don't exist, it sets `query=""` with an explanation.

**Reasoning step:** Before query generation, the LLM outputs structured reasoning:
```json
{
  "reasoning": {
    "tables_used": ["table_name"],
    "relationships": "how tables relate or null",
    "join_keys": ["col_to_join"],
    "result_description": "what query returns"
  },
  "query_type": "pandas" | "sql",
  "query": "exact query string",
  "explanation": "one sentence"
}
```

**Execution modes:**
- **Single CSV:** Pandas expression (e.g., `df[df['col'] == val]`)
- **Single SQLite:** SQL SELECT statement
- **Multi-source:** Forces SQL — all sources loaded into in-memory SQLite for JOINs

**Sandbox design:**
- **Safe builtins:** int, float, str, bool, list, dict, set, len, sum, min, max, abs, round, pow, range, enumerate, zip, map, filter, sorted, reversed, isinstance, type, any, all, repr, hash, callable
- **Excluded:** open, exec, eval, compile, `__import__`, getattr
- Pandas syntax validated via `compile()` before execution
- SQL validated: must start with SELECT (INSERT/DELETE/UPDATE rejected)
- Numpy/pandas types normalized to JSON-safe Python types via `_make_json_safe`

**Multi-table loading (`_load_into_memory_sqlite`):**
- CSVs: `pd.read_csv()` → `to_sql(table_name)` into in-memory SQLite
- SQLites: ATTACH DATABASE → copy tables → DETACH
- Detects duplicate table names

**Output on success:**
```json
{
  "result_value": [...],
  "query_used": "SELECT ...",
  "table_name": "employees",
  "row_count": 42,
  "reasoning": {...},
  "tables_loaded": ["employees"],
  "injected_context": "[Prerequisite result from t1]: {...}"
}
```

**Error categories:** FILE_NOT_FOUND, SYNTAX_ERROR, EMPTY_QUERY, NON_SELECT, EXECUTION_ERROR, NO_RESULTS — all return `success=False` with descriptive error.

---

### Agent 6 — Synthesizer (`agents/synthesizer.py`)

**Job:** Combine all worker results into one coherent draft answer.

**Model:** `gpt-4.1-mini`, temperature 0.1

**Prompt philosophy:** Strict fidelity. The Synthesizer must not add information beyond what the workers returned. Every fact must have an inline source citation. When multiple entities are involved, values must come from each entity's own task result — never swapped.

**Key rules in prompt:**
- Use ONLY information from task results — no invention or inference
- Address every task in the plan
- If tasks failed, acknowledge missing info but answer from what's available
- Inline source citations for every fact
- Use exact values from each entity's own result (multi-entity accuracy)

**All-failed handling:** When every task failed, the prompt injects a special block listing each failure reason so the LLM can explain specifically what went wrong rather than giving a generic "no data found" response.

**Output:** `{draft_answer, synthesizer_output, sources_used}`

---

### Agent 7 — Auditor (`agents/auditor.py`)

**Job:** Verify the draft answer against the plan and evidence. Approve or trigger retry.

**Model:** `gpt-5-mini`, temperature 0.0

**Prompt philosophy:** Adversarial verification. The Auditor's job is to find problems. It checks three specific criteria and must identify exactly what fails, not just that something is wrong.

**Three verification checks:**

1. **COMPLETENESS** — Every task addressed if its result contributed to the answer
2. **ACCURACY** — Factual claims match task results. Compares specific values (numbers, categories, dates). Logical chains acceptable — claims derivable from intermediate lookups pass without explicit source attribution.
3. **NO UNSUPPORTED ASSERTIONS** — Nothing claimed beyond what task results support

**Verdict rules:**
- **PASS:** All three checks pass. Notes briefly confirm. Sets `final_answer = draft_answer`, `final_sources = sources_used`.
- **FAIL:** One or more checks fail. Notes describe exactly what's wrong. Increments `retry_count`, sets `retry_notes`. Does NOT set `final_answer`.

**Output schema:**
```json
{
  "verdict": "PASS" | "FAIL",
  "notes": "...",
  "failed_checks": []  // e.g., ["ACCURACY", "COMPLETENESS"]
}
```

**Retry history:** Every audit (PASS or FAIL) appends `{attempt, draft_answer, audit_verdict, audit_notes}` to `retry_history` for debugging.

**Routing:** The Auditor only writes state. Routing decisions (→ Planner or → Chat) are handled by the graph's `route_after_audit` conditional edge, not by the Auditor.

---

## Manifest System — Two-Tier Design

### Why Two Tiers

The Planner needs to see all sources on every query to decide which ones to use — but full schemas would blow the token budget. Workers need deep schema detail — but only for their assigned source. Two tiers solve this: a lightweight index for discovery, a detailed manifest for execution.

### Tier 1: Manifest Index (`data/manifest_index.yaml`)

**Read by:** Planner (every query)

**Structure:**
```yaml
domain_context: "optional global context for ranking/ordering"

pdfs:
  - id: travel_policy_2024
    name: Travel Policy 2024
    summary: "one-line summary from LLM"
    contains: ["flight class entitlements", "booking rules", ...]
    notes: "human-authored context, preserved on reingest"

tables:
  - id: employees
    name: Employees
    summary: "one-line summary from LLM"
    contains: ["employee names", "departments", ...]
    notes: "human-authored context"

relationships:
  - sources: [employees, salary_bands]
    shared_key: "department/clearance_level"
    description: "salary bands by department and clearance"
    verified: true
```

**Key fields:**
- `contains` — Topic phrases auto-generated by the ingestion LLM. Enables the Planner to match queries to sources without reading full schemas.
- `notes` — Human-authored field preserved across reingestion. Use this to add context the LLM cannot infer (e.g., "this table is updated quarterly").
- `domain_context` — Optional top-level field for global context (e.g., "clearance A is highest").
- `relationships` — Cross-source join keys with verification status. The Planner uses these to identify multi-source tasks.

### Tier 2: Manifest Detail (`data/manifest_detail.yaml`)

**Read by:** Workers (only their assigned source, per query)

**PDF entry:**
```yaml
pdfs:
  - id: travel_policy_2024
    filename: travel_policy_2024.pdf
    type: pdf
    pages: 4
    sections:
      - heading: "Flight Class by Clearance"
        summary: "one-sentence section summary"
    tags: ["corporate travel policy", "flight class", ...]
    notes: "human-authored"
```

**Table entry:**
```yaml
tables:
  - id: employees
    filename: employees.csv
    type: csv                           # or "sqlite"
    base_path: "data/tables/"
    row_count_approx: 10
    table_name: employees               # SQLite only; omitted for CSV
    columns:
      - name: employee_id
        type: integer
        nullable: false
        null_count: 0
        format: "integer"
        min: 1
        max: 10
        description: "Unique employee identifier"
      - name: department
        type: string
        nullable: false
        null_count: 0
        unique_values: ["Engineering", "Finance", "HR", "Sales"]
        description: "Department the employee belongs to"
      - name: hire_date
        type: date
        nullable: false
        null_count: 0
        format: "YYYY-MM-DD"
        unique_values: [10 dates]
        description: "Date the employee was hired"
    relationships:
      - from_column: department
        to_table: salary_bands
        to_column: department
        verified: true
    notes: "human-authored"
```

**Key fields in detail manifest:**
- `unique_values` — Listed if column has <= 30 distinct values. Enables the LLM to use exact values in queries (e.g., `WHERE department = 'Engineering'` not `WHERE department = 'engineering'`).
- `format` — Date format (YYYY-MM-DD, MM/DD/YYYY) or "integer". Prevents query syntax errors.
- `min`/`max` — Range for numeric columns. Helps the LLM write reasonable filters.
- `nullable`/`null_count` — Null statistics. Alerts the LLM to handle missing data.
- `relationships` — Per-table foreign key references with verification status.

### Manifest Cache

Manifests are cached in memory after first load. `invalidate_manifest_cache()` clears the cache and notifies registered callbacks (including the manifest pre-filter's stale flag). Called automatically after ingestion.

### Manifest Formatting

`core/manifest.py` provides formatted views:
- `get_manifest_index()` → formatted index string for Planner prompt
- `get_manifest_index_raw()` → parsed manifest_index.yaml as a dict
- `format_manifest_index(raw)` → format a (possibly filtered) dict into Planner-ready text
- `get_manifest_detail(source_id)` → formatted detail for single source
- `get_manifest_details(source_ids)` → combined detail blocks for multiple sources

### Manifest Pre-Filter (`core/manifest_prefilter.py`)

RAG-based source pre-selection that narrows the manifest before the Planner sees it.

**Problem:** With 8+ sources, the Planner sometimes confuses semantically similar sources (e.g., a "projects" table mentioning "budgets" vs. a finance policy defining "budget limits"). Passing all sources on every query creates a keyword-matching trap.

**Solution:** Embed each source's metadata into a ChromaDB collection (`source_index`), then retrieve only the top-K most relevant sources per query.

**Ingest-time:** `build_source_index()` concatenates each source's `name + summary + contains` into a single text block, embeds it using ChromaDB's default embedding (all-MiniLM-L6-v2), and stores it in the `source_index` collection. PDFs and tables are treated identically — no type-specific handling at the embedding layer.

**Query-time:** `prefilter_manifest(query)` runs inside `planner_node`, using `rewritten_query` (or `original_query` as fallback) for the embedding search.

Steps:
1. Retrieve top-K sources (K=5 default, configurable via `retrieval.prefilter_top_k` in config.yaml)
2. **Relationship expansion** — one-hop walk: if a retrieved source has a declared relationship to a non-retrieved source, pull it in
3. **Minimum diversity** — ensure at least one `record` and one `policy` source (using the `kind` field) when both kinds exist in the manifest
4. **Fallback** — if fewer than 3 sources remain after all expansion, return the full manifest
5. Filter the raw manifest dict and format via `format_manifest_index()`

**Invalidation:** `invalidate_manifest_cache()` triggers a stale flag via callback. The next `prefilter_manifest()` call lazily rebuilds the source_index before querying.

**Trace output:** Each call produces a list of `{source_id, score, expanded_via_relationship}` dicts, exposed in the `/chat` response trace as `prefilter_sources`.

---

## Multi-Source Task Design

### How It Works

1. **Planner assigns multiple source_ids** to a single task when the same worker type can handle them together. Example: a question about two employees' salaries creates one data_scientist task with `source_ids: ["employees", "salary_bands"]`.

2. **Data Scientist multi-table loading:** When a task has multiple source_ids, all sources are loaded into an in-memory SQLite database regardless of original format. CSVs are converted via `pd.read_csv()` → `to_sql()`. SQLite databases are attached and their tables copied. The LLM generates a SQL query with JOINs across the loaded tables.

3. **Librarian parallel search:** When a task has multiple source_ids, the Librarian searches all collections in parallel via `asyncio.gather`. All chunks from all sources are merged before reranking and LLM filtering.

4. **Dependency chains:** When information from one source is needed to query another (e.g., look up employee clearance level, then find their flight entitlement), the Planner creates two tasks with `depends_on`. The Router dispatches them in waves: prerequisite first, then enriches the dependent task's description with the prerequisite result.

### Planner Reasoning

The `can_combine` field in the Planner's reasoning output explains why sources are grouped or split:
```json
"can_combine": "employees and salary_bands are both tables with shared department key — combine into one data_scientist task with JOIN"
```

---

## Retrieval Pipeline

```
User Query
    │
    ▼
Clean Query ──► ChromaDB (per source_id collection)
(strip prereq     │ initial_fetch = 20 chunks per source
 JSON block)      │
                  ▼
            FlashRank Reranker ──► top final_top_k = 5 chunks
            (or PassthroughRanker     │
             if disabled)             │
                                      ▼
                                LLM Relevance Filter
                                (gpt-4.1-mini)
                                      │
                                      ▼
                              1–5 selected chunks
                              returned to Router
```

**Clean query injection:** The Librarian strips `\n\n[Prerequisite result from ...]` blocks from the task description before using it as the ChromaDB search query. The vector search gets a clean semantic query. The full enriched description (with prerequisite context) goes to the LLM filter prompt so the LLM can reason about the prerequisite results.

**Dependency injection:** The Librarian takes an optional `retriever` parameter (defaults to `ChromaRetriever()`). Swap the retrieval backend by passing a different `RetrieverInterface` implementation — no Librarian code changes needed.

**Reranker toggle:** `config.yaml` → `retrieval.reranker_enabled`. When true, `get_reranker()` returns `FlashRanker` (ms-marco-MiniLM-L-12-v2). When false, returns `PassthroughRanker` (chunks unchanged).

**ChromaDB details:**
- One collection per source_id (created at ingest time)
- Default embedding: all-MiniLM-L6-v2 (ChromaDB built-in)
- Distance to score conversion: `relevance_score = 1 / (1 + distance)`

---

## Data Scientist Execution

### Query Generation

The LLM receives the task description, manifest detail (column schemas with types, unique_values, format hints), and prerequisite context. It first reasons about the data structure, then generates a query.

**Prompt rules:**
- CSV: pandas DataFrame expression (`df[df['col'] == val]`)
- SQLite: SQL SELECT statement
- Multi-table: SQL with JOINs (forced regardless of source formats)
- Select only needed columns (never `SELECT *`)
- Case-insensitive filtering: `df['col'].str.lower() == 'value'` or `LOWER(col)`
- Use `unique_values` from schema for exact spelling
- Verify column names exist in schema before writing query
- If column missing or task unanswerable: set `query=""` with explanation

### Sandbox Safety

**Pandas execution:**
- Runs in restricted namespace with whitelisted builtins only
- Dangerous builtins (`open`, `exec`, `eval`, `compile`, `__import__`, `getattr`) excluded
- Syntax validated via `compile()` before execution
- Numpy/pandas types normalized to JSON-safe Python types

**SQL execution:**
- Only SELECT statements permitted
- Non-SELECT (INSERT, DELETE, UPDATE, DROP) raises ValueError
- Single-source: opens file directly
- Multi-source: loads into `:memory:` SQLite

### Multi-Table Loading

`_load_into_memory_sqlite(sources)`:
1. CSVs: `pd.read_csv()` → `df.to_sql(table_name, conn)` into in-memory SQLite
2. SQLites: `ATTACH DATABASE` → copy all tables → `DETACH`
3. Duplicate table names detected and reported
4. Connection closed gracefully on error

---

## Ingestion Pipeline

### PDF Ingestion (`ingestion/pdf_ingestor.py`)

```
PDF file
    │
    ▼
pymupdf4llm.to_markdown(page_chunks=True)
    │ per-page text extraction
    ▼
LLM Cataloging (first 6000 chars)
    │ → summary, sections, tags, contains
    ▼
Manifest Writing
    │ → index entry (id, name, summary, contains, notes)
    │ → detail entry (id, filename, type, pages, sections, tags, notes)
    ▼
Text Chunking
    │ chunk_size=500 chars, chunk_overlap=50 chars
    │ prefers whitespace boundaries
    ▼
ChromaDB Upsert
    │ collection = source_id
    │ clears existing entries (idempotent reingest)
    │ batches of 100
    │ metadata: source_pdf, page_number, chunk_index
    ▼
Return: {source_id, chunks_ingested, summary, tags}
```

### Table Ingestion (`ingestion/table_ingestor.py`)

```
CSV or SQLite file
    │
    ▼
Schema Reading
    │ CSV: pd.read_csv, dtypes
    │ SQLite: PRAGMA table_info, type normalization
    ▼
Column Statistics Computation
    │ unique_values (if <= 30 distinct)
    │ format detection (YYYY-MM-DD, MM/DD/YYYY, integer)
    │ min/max for numeric columns
    │ nullable, null_count
    ▼
LLM Cataloging (schema + sample rows)
    │ → summary, column_descriptions, relationships, contains
    ▼
Manifest Writing
    │ → index entry (id, name, summary, contains, notes)
    │ → detail entry (id, filename, type, base_path, row_count,
    │                  columns with stats + descriptions, relationships, notes)
    ▼
Return: {source_id, row_count, columns, summary, table_name}
```

### Auto-Generated vs Human-Authored

| Field | Generated By | Preserved On Reingest |
|-------|-------------|----------------------|
| summary | LLM | No (regenerated) |
| contains | LLM | No (regenerated) |
| sections, tags | LLM | No (regenerated) |
| column descriptions | LLM | No (regenerated) |
| unique_values, format, min/max | Statistics | No (recomputed) |
| relationships | LLM + manual | Per-table yes; cross-source separate |
| notes | Human | **Yes** (preserved across reingest) |
| domain_context | Human | **Yes** (never overwritten by ingestion) |

### Manifest Writer (`ingestion/manifest_writer.py`)

- `write_source_to_manifest()` — Upserts a source in both manifest files. Preserves existing `notes` field. Invalidates manifest cache.
- `write_cross_source_relationships()` — Writes relationships to manifest_index.yaml top level.
- `update_table_relationships()` — Updates per-table relationships in manifest_detail.yaml.
- `delete_source_from_manifest()` — Removes from both files. Raises ValueError if not found.

---

## Evaluation — RAGAS (`tests/test_ragas.py`)

### Benchmark Dataset

18 questions across 4 difficulty tiers:

| Tier | Count | Description | Example |
|------|-------|-------------|---------|
| Simple Lookup | 5 | Single-source fact retrieval | "What is the password rotation policy?" |
| Filtering | 4 | Single-table query with conditions | "Which projects are currently on hold?" |
| Cross-Source | 5 | Two sources, different types | "What flight class is Dan Cohen entitled to?" |
| Multi-Hop | 4 | Reasoning chain across sources | "Compare leave entitlements of Dan Cohen and Shira Goldman" |

### Metrics

| Metric | Threshold | What It Measures |
|--------|-----------|-----------------|
| Faithfulness | 0.80 | Does the answer stick to retrieved context? |
| Context Precision | 0.75 | Are retrieved chunks relevant to the question? |
| Context Recall | 0.70 | Did retrieval find the information needed for the reference answer? |

### CLI Usage

```bash
# Run all 18 questions + evaluate once
python tests/test_ragas.py

# Run all + evaluate 3x for score stability
python tests/test_ragas.py --multi-eval

# Re-evaluate saved results 3x (no API calls)
python tests/test_ragas.py --scores-only

# Run specific questions by index, merge into results, evaluate
python tests/test_ragas.py --questions 6,10,15
```

### Known Measurement Artifacts

The evaluation script detects and flags cases where RAGAS metrics penalize correct answers due to format mismatch:

- **Faithfulness artifact:** Data Scientist contexts are structured ("From employees: department is Engineering"), not prose. RAGAS faithfulness checks expect passage-style text and may score lower.
- **Context precision artifact:** Extra retrieved chunks don't hurt if the answer is correct and audit passes.
- **Context recall artifact:** Data Scientist results are computed values (query results), not passage matches. The recall metric expects text overlap with reference answer passages.

### Diagnostic Output

Each question produces a diagnostic JSON with: classification stage (intent, rewritten_query, timing), planning stage (reasoning, tasks, source coverage, timing), retrieval stage (per-task details), synthesis stage (draft_answer, timing), audit stage (verdict, notes, retry history, timing), RAGAS scores, and artifact detection.

---

## Configuration (`config.yaml`)

```yaml
llm:
  provider: openai                    # top-level default; agents can override
  agents:
    chat:           { model: gpt-4.1-nano,  temperature: 0.3 }
    planner:        { model: gpt-4.1-mini,  temperature: 0.0 }
    router:         { model: gpt-4.1-nano,  temperature: 0.0 }
    librarian:      { model: gpt-4.1-mini,  temperature: 0.0 }
    data_scientist: { model: gpt-4.1-mini,  temperature: 0.0 }
    synthesizer:    { model: gpt-4.1-mini,  temperature: 0.1 }
    auditor:        { model: gpt-5-mini,    temperature: 0.0 }

retry:
  max_attempts: 3
  failure_message: "I could not verify an answer for your question."

retrieval:
  initial_fetch: 20                   # ChromaDB chunks per source
  final_top_k: 5                      # chunks after reranking
  reranker_enabled: true              # FlashRank on/off
  chunk_size: 500                     # characters per chunk at ingest
  chunk_overlap: 50                   # character overlap between chunks

paths:
  pdfs:             data/pdfs/
  tables:           data/tables/
  chroma_db:        data/chroma_db/
  manifest_index:   data/manifest_index.yaml
  manifest_detail:  data/manifest_detail.yaml

langsmith:
  enabled: true
  project: multi-agent-rag

session:
  memory: in_memory                   # Options: in_memory | sqlite (future) | redis (future)

api:
  host: 0.0.0.0
  port: 8000
  reload: false
```

**Per-agent provider override:** Any agent can add `provider: ollama` to use a local Ollama model instead of OpenAI. The top-level `provider` is the default.

**Model selection rationale:**
- **gpt-4.1-nano** for Chat and Router — fast, cheap, no complex reasoning needed (classification and dispatch)
- **gpt-4.1-mini** for Planner, Librarian, Data Scientist, Synthesizer — good balance of capability and cost for structured reasoning and query generation
- **gpt-5-mini** for Auditor — highest capability for the most critical verification step; a missed error costs a full retry cycle or an incorrect answer

---

## Developer Dashboard (`frontend/index.html`)

Single-page application served at `/static`. Two-panel layout:

### Left Panel — Data Sources
- Lists all indexed sources from `GET /sources`
- Colored badges by type: PDF (blue), CSV (green), SQLite (purple)
- Expandable cards showing: summary, page count/tags (PDFs), row count/columns (tables)
- Upload buttons for PDF and table files (SQLite prompts for table name)
- Delete button per source (removes from manifests + ChromaDB)

### Right Panel — Chat
- Session management: UUID-based, "New Session" and "Copy Conversation" buttons
- Message display: user messages right-aligned (blue), assistant messages left-aligned
- Collapsible trace details per response:
  - **Classification:** intent badge (PLAN/DIRECT/CLARIFY with color), rewritten query, timing
  - **Planning:** tasks, source coverage, timing
  - **Retrieval:** per-task worker type, success/failure, query used, chunks retrieved
  - **Synthesis:** draft answer, timing
  - **Audit:** verdict, notes, retry history
  - **RAGAS scores** (if evaluated)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Run the full agent graph. Returns `final_answer`, `final_sources`, `session_id`, `trace`, `conversation_entry`. |
| GET | `/health` | Liveness check. Returns `{status: "ok", version: "1.0"}`. |
| GET | `/conversation/{session_id}` | Returns the full conversation log for a session (list of conversation entries). |
| POST | `/ingest/pdf` | Multipart file upload. Saves to `data/pdfs/`, parses, chunks, upserts into ChromaDB, writes both manifests. |
| POST | `/ingest/table` | Multipart file upload + optional `table_name` form field. Saves to `data/tables/`, catalogs via LLM, writes both manifests. |
| GET | `/sources` | Returns all indexed sources merged from both manifest files. |
| DELETE | `/sources/{source_id}` | Removes from both manifests and deletes ChromaDB collection if present. |

### Trace Information

Every `/chat` response includes a `trace` object with:
- `retry_count`, `audit_verdict`, `audit_notes`
- `plan`, `task_results`, `synthesizer_output`
- `chat_intent`, `rewritten_query`, `query_sent_to_planner`
- `chat_formatted_response` (boolean)
- `step_timings` (per-node elapsed_ms: classification_ms, formatting_ms, planning_ms, routing_ms, synthesis_ms, audit_ms)
- `retry_history`, `planner_reasoning`
- `prefilter_sources` (list of `{source_id, score, expanded_via_relationship}` — which sources the pre-filter selected and why)

### Session Management

`api.py` maintains two module-level dicts:
- `_sessions[session_id]` → conversation_history (list of Message)
- `_conversation_log[session_id]` → list of conversation entries (one per turn)

History persists for the lifetime of the server process. Lost on restart (v1).

---

## Project File Structure

```
project/
├── agents/
│   ├── __init__.py
│   ├── chat.py              # Agent 1: Chat Interface (classify/deliver)
│   ├── planner.py           # Agent 2: Planner (decompose into tasks)
│   ├── router.py            # Agent 3: Router (wave dispatch)
│   ├── librarian.py         # Agent 4: Librarian (PDF search worker)
│   ├── data_scientist.py    # Agent 5: Data Scientist (CSV/SQLite worker)
│   ├── synthesizer.py       # Agent 6: Synthesizer (assemble draft)
│   └── auditor.py           # Agent 7: Auditor (verify/retry)
├── core/
│   ├── __init__.py
│   ├── state.py             # AgentState TypedDict + view functions
│   ├── llm_config.py        # Per-agent LLM loader (OpenAI/Ollama)
│   ├── manifest.py          # Manifest loader, cache, formatters
│   ├── manifest_prefilter.py # RAG pre-filter over manifest source index
│   ├── session_context.py   # Per-session user-fact store (module-level dict)
│   ├── data_context.py      # Runtime domain summary derived from manifest
│   ├── registry.py          # Worker Registry (worker_type → callable)
│   ├── retriever.py         # RetrieverInterface + ChromaRetriever
│   ├── reranker.py          # RerankerInterface + FlashRanker + PassthroughRanker
│   └── parse.py             # parse_llm_json (markdown fence stripping)
├── data/
│   ├── pdfs/                # Real user-supplied PDFs — gitignored
│   ├── tables/              # Real CSV and SQLite files — gitignored
│   ├── chroma_db/           # ChromaDB vector store — gitignored
│   ├── manifest_index.yaml  # Planner-facing manifest — committed
│   └── manifest_detail.yaml # Worker-facing manifest — committed
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── pdfs/            # Test fixture PDFs — committed
│   │   └── tables/          # Test fixture CSVs/SQLite — committed
│   ├── test_state.py
│   ├── test_manifest.py
│   ├── test_agents.py
│   ├── test_ragas.py        # RAGAS evaluation benchmark (18 questions)
│   └── fixtures.py          # Shared test data and fake states
├── ingestion/
│   ├── __init__.py
│   ├── manifest_writer.py   # write/update/delete manifest entries
│   ├── pdf_ingestor.py      # PDF parse → catalog → chunk → ChromaDB
│   └── table_ingestor.py    # Table schema → stats → catalog → manifest
├── scripts/
│   ├── create_test_data.py  # Generate 4 PDFs + 4 tables as fixtures
│   └── ingest_pdfs.py       # Batch ingest (fixtures + real data)
├── frontend/
│   └── index.html           # Developer dashboard SPA
├── graph.py                 # LangGraph wiring: nodes + conditional edges
├── api.py                   # FastAPI endpoints + session management
├── config.yaml              # All configuration (models, paths, retrieval)
├── .env                     # API keys — never committed
├── .env.example             # Template for .env
├── requirements.txt
├── .gitignore
└── ARCHITECTURE.md          # This file
```

### Data Path Rules

| Location | Purpose | Git |
|---|---|---|
| `tests/fixtures/pdfs/` | Test fixture PDFs generated by `scripts/create_test_data.py` | **Committed** |
| `tests/fixtures/tables/` | Test fixture tables generated by `scripts/create_test_data.py` | **Committed** |
| `data/pdfs/` | Real user-supplied PDFs | Gitignored |
| `data/tables/` | Real CSV and SQLite files | Gitignored |
| `data/chroma_db/` | ChromaDB vector store (auto-created at runtime) | Gitignored |
| `data/manifest_*.yaml` | Manifest config files (managed by ingestion scripts) | **Committed** |

**Rule: never use `data/` paths in tests.** Tests must reference files under `tests/fixtures/` so they work on any machine after `git clone`.

**Rule: test files must never write to `data/pdfs/`, `data/tables/`, or either manifest file.** Use `tmp_path` or `tests/fixtures/` paths instead.

---

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Framework | LangGraph | Explicit graph nodes + edges, full state control |
| Language | Python 3.11+ | |
| LLM Provider | OpenAI (default) | Per-agent config; Ollama supported as override |
| Vector DB | ChromaDB (local) | Fully local, no infrastructure |
| Embedding | all-MiniLM-L6-v2 | ChromaDB default, used at ingest + search |
| Reranker | FlashRank | ms-marco-MiniLM-L-12-v2, configurable on/off |
| PDF Parsing | pymupdf4llm | Handles tables inside PDFs |
| Structured Data | CSV + SQLite | CSV via Pandas, relational via SQLite |
| State Typing | Pydantic TypedDict | |
| Web Framework | FastAPI | |
| Observability | LangSmith | Full node-level tracing |
| Frontend | Vanilla JS + Pico CSS | Single-page developer dashboard |

---

## Utilities

### `core/parse.py` — `parse_llm_json(raw: str) -> dict`

All agents that expect structured JSON from the LLM call this function instead of `json.loads` directly. It strips optional markdown code fences (`` ```json … ``` `` or `` ``` … ``` ``) before parsing, then raises `ValueError` with the raw LLM output included if parsing fails.

### `core/llm_config.py` — `get_llm(agent_name: str)`

Returns a configured `ChatOpenAI` or `ChatOllama` client for the named agent. Model, temperature, and provider resolved from `config.yaml`. Agent-level `provider` overrides the top-level default. Raises `EnvironmentError` if OpenAI provider is used without `OPENAI_API_KEY`.

### `core/registry.py` — Worker Registry

```python
WORKER_REGISTRY = {
    "librarian":      librarian_worker,
    "data_scientist": data_scientist_worker,
}
```

Planner outputs `worker_type` strings. Router calls `get_worker(worker_type)` to retrieve the async callable. Adding a new data type = register one entry + write the worker function. Planner and Router never change.

---

## Observability

- LangSmith enabled from day one (`langsmith.enabled: true` in config.yaml)
- Every node execution traced: input state, prompt sent, LLM response, output state
- Set `LANGSMITH_API_KEY` in `.env`

---

## Logging

All production code uses Python's `logging` module. Each file declares `logger = logging.getLogger(__name__)`.

Convention:
- `logger.info()` — normal operational flow
- `logger.warning()` — handled errors and degraded paths
- `logger.error()` — failures that prevent a result

`api.py` configures the root logger at startup:
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
```
Module loggers for `agents` and `core` are set to DEBUG level. Third-party loggers (httpcore, openai, langsmith, urllib3, httpx) are suppressed.

---

## Known Limitations and Open Items

- **In-memory session storage** — Conversation history lost on server restart. Persistent storage (SQLite or Redis) is a planned future feature.
- **Sequential PDF ingestion** — `scripts/ingest_pdfs.py` processes PDFs one at a time. Parallel ingestion would speed up bulk imports.
- **GraphRAG** — `RetrieverInterface` is abstracted and ready, but no GraphRAG backend is implemented.
- **No user authentication** — Any client can access any session. Not suitable for multi-tenant production.
- **No streaming responses** — The API returns the full answer after the entire pipeline completes.
- **No Docker/deployment** — Development server only. No containerization or deployment configuration.

---

## How to Run

### Start the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and LANGSMITH_API_KEY

# Generate test fixtures (first time only)
python -m scripts.create_test_data

# Ingest PDFs into ChromaDB (fixtures + any real data in data/pdfs/)
python -m scripts.ingest_pdfs

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run the RAGAS Benchmark

```bash
# Requires running API server on localhost:8000

# Full benchmark (18 questions)
python tests/test_ragas.py

# Specific questions
python tests/test_ragas.py --questions 10,15,17

# Re-evaluate saved results (no API calls)
python tests/test_ragas.py --scores-only
```

### Run Tests

```bash
pytest tests/
```

---

## Environment Variables Required

```
OPENAI_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=multi-agent-rag
```

---

## Key Constraints — Claude Code Must Respect These

- **Never** add fields to AgentState not listed above without explicit instruction.
- **Never** let an agent access state fields outside its view table.
- **Never** hardcode an LLM client inside an agent — always inject via `get_llm()`.
- **Never** hardcode ChromaDB inside the Librarian — use the RetrieverInterface.
- **Always** write an isolated test function after each agent is built.
- **Always** use `asyncio.gather` for parallel worker dispatch in the Router.
- **State updates return only changed fields** — never return the full state from a node.
- **Never** use `print()` in production code — use `logging.getLogger(__name__)`.
- **Never** import private functions (underscore-prefixed) across module boundaries.
- **Test files must never write to `data/` paths** — use `tmp_path` or `tests/fixtures/`.

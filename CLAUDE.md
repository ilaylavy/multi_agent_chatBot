# Claude Code — Session Primer

> Read ARCHITECTURE.md before doing anything. It is the source of truth for design.
> This file covers session rules and operational guidance only — it does not duplicate architecture.
> be more concise

---

## Project State

A 7-agent RAG system on LangGraph that answers natural-language questions over mixed PDFs and structured tables. Backend is complete. Dev-facing frontend exists for debugging and iteration; production frontend comes later. Current focus is maintenance, tuning, and incremental improvements — not greenfield agent construction.

If a task looks like it requires building an agent from scratch, confirm with the user before proceeding — the usual work is modifying existing agents, not adding new ones.

---

## Non-Negotiable Rules

1. **Read ARCHITECTURE.md at the start of every session.** It defines the full state schema, agent views, graph wiring, manifest system, and data paths. Do not guess — look.
2. **Do not add fields to `AgentState`** beyond what ARCHITECTURE.md specifies.
3. **Do not modify ARCHITECTURE.md** without explicit user instruction. If an architectural change seems needed, flag it and ask.
4. **Agents never instantiate their own LLM.** Always inject via `core.llm_config.get_llm()`.
5. **Agents never access state directly in prompts.** Always route through the agent's view function in `core/state.py` or the agent module.
6. **State nodes return only changed fields** — never the full state object.
7. **Router uses `asyncio.gather`** for parallel worker dispatch — never sequential loops.
8. **Librarian uses `RetrieverInterface`** — never calls ChromaDB directly.
9. **Never use `print()` in production code** — use `logging.getLogger(__name__)` with `info`, `warning`, or `error`.
10. **Never import private (underscore-prefixed) functions across module boundaries.** Request a public wrapper instead.
11. **Tests never write to `data/`.** See the Data and Fixtures section below.
12. **One file at a time.** Do not create files that were not explicitly requested.

---

## Data and Fixtures

This project has four distinct data locations. Confusing them is the single most common source of broken tests and dirty diffs.

| Location | Purpose | Committed? | Who writes here |
|---|---|---|---|
| `tests/fixtures/pdfs/`, `tests/fixtures/tables/` | Frozen test fixtures | Yes | `scripts/create_test_data.py` only, and only on explicit user instruction |
| `data/pdfs/`, `data/tables/` | Real user-supplied data | No (gitignored) | User uploads via frontend or CLI, `scripts/ingest_all.py` when re-ingesting |
| `data/chroma_db/` | ChromaDB vector store | No (gitignored) | Ingestion code only |
| `data/manifest_index.yaml`, `data/manifest_detail.yaml` | Source manifests | Yes | Ingestion code only, via `ingestion/manifest_writer.py` |

### Rules

- **Tests must read from `tests/fixtures/` or use `tmp_path`.** Never reference `data/` paths in test code.
- **Tests must never write to `data/pdfs/`, `data/tables/`, `data/chroma_db/`, or either manifest file.** Use `tmp_path` for any test that needs to write.
- **Fixtures are frozen.** Do not regenerate them. Do not run `scripts/create_test_data.py` without explicit user instruction.
- **Do not re-ingest without explicit user instruction.** Running `scripts/ingest_all.py` rewrites both committed manifest YAML files via non-deterministic LLM calls, producing a noisy diff that is almost never what the user wants in a PR. Re-ingestion is appropriate only when fixture data itself has changed.
- **If a fixture file is missing on disk**, do not silently regenerate it. Tell the user. The only exception is `conftest.py`'s session fixture for `employees.csv`, which restores that one file — do not extend this pattern to other fixtures without asking.
- **The manifests in `data/manifest_*.yaml` are committed but not reviewed line-by-line.** Do not edit them by hand. Changes flow through ingestion code.
- **All LLM prompts must be domain-agnostic** Prompts are product code — they ship to customers with completely different data. Never reference specific source names, table names, column names, entity types, or domain concepts from the development dataset in any system prompt or user template. If a prompt needs to reference what the system contains, it must derive that information at runtime from the manifest or data context, not hardcode it. A prompt that mentions "employees", "clearance levels", or "travel policy" only works for our test data. A prompt that says "the available data sources" or "the entities in your records" works for any deployment.

---

## Working on Agents

### Signature patterns

LangGraph node (chat, planner, router, synthesizer, auditor):
```python
async def agent_name_node(state: AgentState) -> dict:
```

Registry worker callable (librarian, data_scientist):
```python
async def worker_name(state: AgentState, task: Task) -> TaskResult:
```

Nodes receive state only — LangGraph calls them. Workers receive `(state, task)` — the Router dispatches them via `asyncio.gather`.

### LLM output parsing

Any agent that expects structured output from the LLM must:

1. Prompt the LLM to return JSON matching a named schema.
2. Parse with `core.parse.parse_llm_json(response.content)` — not `json.loads` directly. `parse_llm_json` strips optional markdown fences and raises `ValueError` with the raw output included on parse failure.
3. Never use LangChain's `with_structured_output`.

### View functions

Every agent has a view function that filters `AgentState` down to the fields that agent's prompt is allowed to see. The LLM prompt is built from the view, never from raw state. Do not bypass this to "just include one more field" — if a field truly needs to be visible, update ARCHITECTURE.md first.

### Adding a new worker type

If the user asks for a new worker (e.g., a SQL-only worker, an API-calling worker):

1. Write the worker as an async callable with signature `(state, task) -> TaskResult`.
2. Register it in `WORKER_REGISTRY` in `core/registry.py`.
3. Update the Planner's prompt so it knows when to route to the new `worker_type`.
4. Planner and Router code itself does not change.

---

## Common Mistakes to Avoid

The following are mistakes that are easy to make in this codebase and expensive to catch late:

- **Writing test files into `data/pdfs/` or `data/tables/`.** Tests must use `tests/fixtures/` or `tmp_path`. See the Data and Fixtures section.
- **Regenerating fixtures or re-ingesting "to be safe".** This rewrites committed manifests via LLM calls. Do not do this without explicit user instruction.
- **Adding fields to `AgentState` to pass information between nodes.** The state schema is fixed. If you think you need a new field, there is almost certainly an existing field or view function that already carries that information, or the information belongs inside an existing nested structure (like `TaskResult.output`).
- **Duplicating logic that already exists in `core/`.** Before writing anything that parses LLM JSON, loads manifests, picks an LLM client, or reranks chunks, check `core/parse.py`, `core/manifest.py`, `core/llm_config.py`, and `core/reranker.py`. If you need a variant, extend the existing module rather than forking it.
- **Importing underscore-prefixed functions across modules.** If you find yourself writing `from core.llm_config import _load_config` outside `core/`, stop. Either the function should be made public, or there's a higher-level helper you should be using instead. Ask.
- **Calling ChromaDB directly from an agent.** All retrieval goes through `RetrieverInterface`. The whole point of that abstraction is GraphRAG swap-ability.
- **Using sequential loops in the Router.** Dispatch must be `asyncio.gather`. A sequential loop is a regression even if it produces identical output.
- **Editing ARCHITECTURE.md to reflect a local change.** ARCHITECTURE.md is the design spec — it leads, code follows. If a change doesn't match ARCHITECTURE.md, either the change is wrong or ARCHITECTURE.md needs a deliberate update. In either case, ask the user first.
- **Running the full RAGAS suite to check a small change.** See Operational Guidance below.
- **Using `print()` anywhere in `agents/`, `core/`, `ingestion/`, or `api.py`.** Use `logger.info/warning/error`.

---

## Operational Guidance

### Running tests

- **Fast feedback loop:** `pytest tests/test_mock_suite.py -v`. All LLM calls mocked, no credentials needed, runs in seconds.
- **Per-agent test:** each agent file has a `test_<name>()` function and `if __name__ == "__main__"` block. Run with `python -m agents.<name>`. Use this when iterating on a single agent.
- **Integration tests:** `pytest tests/test_integration.py -v` makes real OpenAI calls. Skip with `-m 'not integration'`. Only run these when the user asks or when verifying an end-to-end change.
- **Do not run the RAGAS benchmark** (`python tests/test_ragas.py`) on every change. It requires a running API, takes several minutes, and costs money. Run it only when the user asks or when explicitly tuning retrieval or synthesis quality.

### Ingestion

- Re-ingest only when fixture data itself has changed, or when the user asks. Re-ingestion rewrites both committed manifest YAML files.
- If the user uploads real data through the frontend, that goes to `data/pdfs/` or `data/tables/` and is gitignored — this is expected and fine.
- `core/manifest.py` caches manifest reads. After any ingestion, `invalidate_manifest_cache()` is called automatically. If you're debugging a "stale manifest" issue, check that.

### When the audit keeps failing

If a query loops through all 3 retries and exhausts, the issue is usually one of:
- Planner chose the wrong sources (check `planner_reasoning` in the trace)
- Retrieval returned empty or low-relevance chunks (check `retrieved_chunks`)
- Synthesizer dropped a fact the Auditor expected (compare `synthesizer_output` to `draft_answer`)
- Auditor is over-strict for a legitimately partial answer

Look at the `retry_history` in state — it records every attempt's draft and audit notes. Don't guess; read the trace.

---

## When to Ask Before Acting

Ask the user before:

- Regenerating fixtures (`scripts/create_test_data.py`)
- Re-ingesting (`scripts/ingest_all.py` or equivalent)
- Running the RAGAS benchmark
- Modifying ARCHITECTURE.md
- Adding a field to `AgentState`
- Adding a new file that wasn't explicitly requested
- Changing `config.yaml` in ways that affect behavior (model choices, retry count, chunk sizes)
- Running integration tests that make real LLM calls when a mock test would do

When in doubt, one quick question beats a noisy diff.
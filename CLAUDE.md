# Claude Code — Session Primer

> Read this file and ARCHITECTURE.md before doing anything in this project.

## Current State

Backend complete. All 13 tests passing. Moving to frontend.

---

## What You Are Building

A 7-agent RAG system that answers natural language questions by querying
PDFs and structured tables, verifying the answer, and returning it to the user.

Full design is in ARCHITECTURE.md. Do not deviate from it.

---

## Non-Negotiable Rules

1. **Read ARCHITECTURE.md first.** Always. Every session.
2. **One file at a time.** Do not create files not explicitly requested.
3. **Do not add fields to AgentState** beyond what is specified in ARCHITECTURE.md.
4. **Agents never instantiate their own LLM.** Always inject via `core/llm_config.py`.
5. **Agents never access state directly.** Always use the agent's view function.
6. **Do not implement deferred features.** See the Deferred Features list in ARCHITECTURE.md.
7. **After each agent: write an isolated test function** the user can run immediately.
8. **State nodes return only changed fields** — never the full state object.
9. **Router uses asyncio.gather** for parallel worker dispatch — never sequential loops.
10. **Librarian uses RetrieverInterface** — never calls ChromaDB directly.
11. **Test files must never write to `data/pdfs/`, `data/tables/`, or either manifest file directly** — use temp directories or `tests/fixtures/` paths only.
12. **Never use `print()` in production code** — use `logging.getLogger(__name__)` and the appropriate level (`info`, `warning`, `error`).
13. **Never import private functions (underscore-prefixed) across module boundaries** — request a public wrapper instead.
14. **Test files must never write to `data/pdfs/`, `data/tables/`, or manifest files** — use `tmp_path` or `tests/fixtures/` only.

---

## Build Order

Do not skip steps. Do not reorder.

```
1.  core/state.py
2.  core/llm_config.py
3.  core/manifest.py
4.  core/registry.py
5.  graph.py skeleton (empty nodes)
6.  agents/planner.py
7.  agents/librarian.py
8.  agents/data_scientist.py
9.  agents/router.py
10. agents/synthesizer.py
11. agents/auditor.py
12. agents/chat.py
13. graph.py (wire all edges)
14. api.py
```

---

## Before Writing Any Agent

Check:
- Does `core/state.py` exist and match ARCHITECTURE.md? If not, stop and fix it first.
- Does `core/llm_config.py` exist? If not, stop and build it first.
- Is the agent in the build order above? Build only the next one in sequence.

---

## How to Structure Each Agent

Every agent file must follow this pattern:

```python
# 1. View function — filters state down to only what this agent needs
def agent_name_view(state: AgentState) -> dict:
    ...

# 2. Node function — the LangGraph node
async def agent_name_node(state: AgentState) -> dict:
    view = agent_name_view(state)
    llm = get_llm("agent_name")   # from core/llm_config.py
    # ... agent logic ...
    return { "only_the_fields_this_agent_changes": value }

# 3. Isolated test function at the bottom
def test_agent_name():
    fake_state = { ... }  # minimal state needed for this agent
    result = asyncio.run(agent_name_node(fake_state))
    assert "expected_field" in result
    print("PASS:", result)

if __name__ == "__main__":
    test_agent_name()
```

---

## Signature Patterns

**LangGraph node** — used for all agents that are graph nodes (planner, router, synthesizer, auditor, chat, librarian, data_scientist). Receives state only:
```python
async def agent_name_node(state: AgentState) -> dict:
```

**Registry worker callable** — used for internal worker functions dispatched by the Router via `asyncio.gather`. Receives both state and its specific task:
```python
async def worker_name(state: AgentState, task: Task) -> TaskResult:
```

Note: LangGraph nodes receive `(state)` only — LangGraph calls them with state. Worker callables receive `(state, task)` — the Router passes the task explicitly via `asyncio.gather`.

---

## LLM Output Parsing

All agents that need structured output from the LLM must:
1. Prompt the LLM to respond in JSON matching a specific schema.
2. Parse the response with `json.loads` inside a `try/except` that raises a `ValueError` including the raw LLM output in the message.

Never use `with_structured_output`.

```python
try:
    data = json.loads(response.content)
except (json.JSONDecodeError, KeyError) as exc:
    raise ValueError(f"Failed to parse LLM output: {exc}\nRaw output: {response.content}") from exc
```

---

## Current Status

Track progress here. Update after each file is completed and tested.

- [x] core/state.py
- [x] core/llm_config.py
- [x] core/manifest.py
- [x] core/registry.py
- [x] core/parse.py
- [x] core/retriever.py
- [x] graph.py (skeleton)
- [x] agents/planner.py
- [x] agents/librarian.py
- [x] agents/data_scientist.py
- [x] agents/router.py
- [x] agents/synthesizer.py
- [x] agents/auditor.py
- [x] agents/chat.py
- [x] graph.py (wired)
- [x] api.py
- [x] ingestion/__init__.py
- [x] ingestion/manifest_writer.py
- [x] ingestion/pdf_ingestor.py
- [x] ingestion/table_ingestor.py
- [x] scripts/create_test_data.py
- [x] scripts/ingest_pdfs.py

"""
agents/librarian.py — Librarian worker (unstructured PDF search).

This is a registry worker callable, NOT a LangGraph node.
Signature: async librarian_worker(state, task) -> TaskResult

Retrieval flow:
  ChromaDB (retrieval.initial_fetch) → [RERANKER SLOT — v2] → retrieval.final_top_k → LLM relevance filter → TaskResult

The Librarian always codes against RetrieverInterface.
To swap to a GraphRAG backend: pass a different RetrieverInterface subclass.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from core.llm_config import _load_config, get_llm
from core.manifest import get_manifest_detail
from core.parse import parse_llm_json
from core.retriever import ChromaRetriever, RetrieverInterface
from core.state import AgentState, Chunk, Task, TaskResult

logger = logging.getLogger(__name__)

# Read retrieval limits from config.yaml — change there, no agent code changes needed.
# _load_config() is lru_cache'd so this does not re-read the file on every call.
_retrieval_cfg = _load_config()["retrieval"]
_INITIAL_FETCH: int = _retrieval_cfg["initial_fetch"]
_FINAL_TOP_K:   int = _retrieval_cfg["final_top_k"]

# Default retriever — shared across calls; injected in tests
_default_retriever = ChromaRetriever()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are a document retrieval specialist. You have been given a task and a set of
text chunks retrieved from a PDF document. Your job is to identify which chunks
are genuinely relevant to the task and return them.

Rules:
  - Only return chunks that directly help answer the task description.
  - Do not alter chunk_text — copy it exactly as provided.
  - Return between 1 and {_FINAL_TOP_K} chunks. If none are relevant, return an empty list.
  - Preserve the source_pdf and page_number from the input.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{{
  "selected_chunks": [
    {{
      "chunk_text":      "exact text of the chunk",
      "source_pdf":      "filename as provided",
      "page_number":     3,
      "relevance_score": 0.95
    }}
  ]
}}
"""

_USER_TEMPLATE = """\
SOURCE MANIFEST DETAIL:
{manifest_detail}

TASK:
{task_description}

RETRIEVED CHUNKS (ranked by vector similarity, most similar first):
{chunks_block}

Select the chunks that best answer the task. Return as JSON.
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def librarian_view(state: AgentState, task: Task, manifest_detail: str) -> dict:
    """
    Returns only the assigned task and manifest detail for that source.
    The LLM prompt is built from this view — nothing else from state is visible.
    """
    return {
        "task":            task,
        "manifest_detail": manifest_detail,
    }


# ---------------------------------------------------------------------------
# Worker callable
# ---------------------------------------------------------------------------

async def librarian_worker(
    state: AgentState,
    task: Task,
    retriever: RetrieverInterface = None,
) -> TaskResult:
    """
    Registry worker callable — dispatched by the Router via asyncio.gather.

    Parameters
    ----------
    state     : Full AgentState — filtered by librarian_view before use.
    task      : The single Task assigned to this worker.
    retriever : RetrieverInterface instance. Defaults to ChromaRetriever.
                Pass a mock or alternative backend in tests / future backends.
    """
    if retriever is None:
        retriever = _default_retriever

    manifest_detail = get_manifest_detail(task["source_id"])
    view = librarian_view(state, task, manifest_detail)

    # ── Retrieval ────────────────────────────────────────────────
    chunks: List[Chunk] = await retriever.search(
        query=task["description"],
        source_id=task["source_id"],
        top_k=_INITIAL_FETCH,
    )

    # ── RERANKER SLOT — pass chunks through FlashRank here in v2 ─
    # For now: take top _FINAL_TOP_K by relevance_score
    chunks = sorted(chunks, key=lambda c: c["relevance_score"], reverse=True)
    chunks = chunks[:_FINAL_TOP_K]

    # ── Build chunks block for LLM prompt ────────────────────────
    if chunks:
        chunks_block = "\n\n".join(
            f"[Chunk {i+1} | {c['source_pdf']} p.{c['page_number']} "
            f"| score={c['relevance_score']}]\n{c['chunk_text']}"
            for i, c in enumerate(chunks)
        )
    else:
        chunks_block = "(no chunks retrieved — collection may not be ingested yet)"

    user_message = _USER_TEMPLATE.format(
        manifest_detail=view["manifest_detail"],
        task_description=view["task"]["description"],
        chunks_block=chunks_block,
    )

    llm = get_llm("librarian")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    # ── Parse LLM response — per CLAUDE.md LLM Output Parsing rule ──
    data = parse_llm_json(response.content)
    try:
        selected: List[Chunk] = [
            Chunk(
                chunk_text=c["chunk_text"],
                source_pdf=c["source_pdf"],
                page_number=int(c["page_number"]),
                relevance_score=float(c["relevance_score"]),
            )
            for c in data["selected_chunks"]
        ]
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    return TaskResult(
        task_id=task["task_id"],
        worker_type="librarian",
        output=json.dumps(selected),
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.librarian
# ---------------------------------------------------------------------------

def test_librarian():
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    fake_task: Task = {
        "task_id":     "t2",
        "worker_type": "librarian",
        "description": "Find flight class entitlements for clearance level A",
        "source_id":   "travel_policy_2024",
    }

    fake_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "plan":                 [fake_task],
        "manifest_context":     "",
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "final_answer":         "",
        "final_sources":        [],
    }

    # ── Mock retriever ────────────────────────────────────────────
    fake_chunks: List[Chunk] = [
        {"chunk_text": "Employees with clearance level A are entitled to Business Class on flights exceeding 4 hours.",
         "source_pdf": "travel_policy.pdf", "page_number": 3, "relevance_score": 0.97},
        {"chunk_text": "Clearance level B entitles employees to Premium Economy on flights exceeding 6 hours.",
         "source_pdf": "travel_policy.pdf", "page_number": 3, "relevance_score": 0.81},
    ]

    mock_retriever = MagicMock(spec=RetrieverInterface)
    mock_retriever.search = AsyncMock(return_value=fake_chunks)

    # ── Mock LLM ──────────────────────────────────────────────────
    fake_llm_output = json.dumps({
        "selected_chunks": [
            {"chunk_text": "Employees with clearance level A are entitled to Business Class on flights exceeding 4 hours.",
             "source_pdf": "travel_policy.pdf", "page_number": 3, "relevance_score": 0.97}
        ]
    })
    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    patch_target = f"{__name__}.get_llm"

    # ── Test 1: librarian_view returns only task + manifest_detail ─
    view = librarian_view(fake_state, fake_task, "some detail")
    assert set(view.keys()) == {"task", "manifest_detail"}
    assert view["task"] is fake_task
    print("PASS: librarian_view returns only task and manifest_detail")

    # ── Test 2: worker returns TaskResult with success=True ────────
    with patch(patch_target, return_value=mock_llm), \
         patch("agents.librarian.get_manifest_detail", return_value="fake manifest detail") \
         if __name__ == "__main__" else \
         patch("agents.librarian.get_manifest_detail", return_value="fake manifest detail"):

        result: TaskResult = asyncio.run(
            librarian_worker(fake_state, fake_task, retriever=mock_retriever)
        )

    assert result["task_id"]     == "t2"
    assert result["worker_type"] == "librarian"
    assert result["success"]     is True
    assert result["error"]       is None
    print("PASS: librarian_worker returns TaskResult with success=True")

    # ── Test 3: output contains the selected chunks ────────────────
    output_chunks = json.loads(result["output"])
    assert len(output_chunks) == 1
    assert "Business Class" in output_chunks[0]["chunk_text"]
    assert output_chunks[0]["page_number"] == 3
    print(f"PASS: output contains {len(output_chunks)} selected chunk(s)")

    # ── Test 4: retriever was called with correct args ─────────────
    mock_retriever.search.assert_called_once_with(
        query=fake_task["description"],
        source_id=fake_task["source_id"],
        top_k=_INITIAL_FETCH,
    )
    print("PASS: retriever.search called with correct query, source_id, top_k=20")

    # ── Test 5: bad LLM JSON raises ValueError ─────────────────────
    bad_response = MagicMock()
    bad_response.content = "I cannot retrieve that information."
    mock_llm.ainvoke = AsyncMock(return_value=bad_response)

    with patch(patch_target, return_value=mock_llm), \
         patch("agents.librarian.get_manifest_detail", return_value="fake manifest detail") \
         if __name__ == "__main__" else \
         patch("agents.librarian.get_manifest_detail", return_value="fake manifest detail"):
        try:
            asyncio.run(librarian_worker(fake_state, fake_task, retriever=mock_retriever))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "I cannot retrieve that information." in str(exc)
            print("PASS: ValueError raised with raw output on bad JSON")

    print("\nPASS: all librarian tests passed")


if __name__ == "__main__":
    test_librarian()

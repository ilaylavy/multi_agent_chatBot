"""
tests/test_integration.py — Full integration test suite.

Tests the real system end-to-end: real ChromaDB, real CSV/SQLite, real OpenAI API.

Test markers:
  @pytest.mark.integration — requires a live OpenAI API key and makes real API calls.
                              Skip during development with: pytest -m 'not integration'

Run all:              pytest tests/test_integration.py -v
Run without LLM:      pytest tests/test_integration.py -v -m 'not integration'
Run only LLM tests:   pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that make real OpenAI API calls "
        "(skip with: pytest -m 'not integration')",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state(query: str) -> dict:
    """Build a minimal but complete initial AgentState for graph invocation."""
    from core.manifest import get_manifest_index
    from core.state import AgentState

    return AgentState(
        original_query=query,
        session_id="integration-test",
        conversation_history=[],
        plan=[],
        manifest_context=get_manifest_index(),
        task_results={},
        sources_used=[],
        retrieved_chunks=[],
        draft_answer="",
        audit_result={"verdict": "PASS", "notes": ""},
        retry_count=0,
        retry_notes="",
        final_answer="",
        final_sources=[],
    )


# ---------------------------------------------------------------------------
# Test 1 — Retrieval (no LLM, no API key required)
# ---------------------------------------------------------------------------

def test_retrieval():
    """
    Calls ChromaRetriever.search() directly against the ingested travel_policy_2024
    collection. Confirms chunks are returned and the top chunk is relevant.
    No LLM call — requires only that scripts/ingest_pdfs.py has been run.
    """
    from core.retriever import ChromaRetriever

    retriever = ChromaRetriever()
    chunks = asyncio.run(
        retriever.search(
            query="Business Class flight entitlement",
            source_id="travel_policy_2024",
            top_k=5,
        )
    )

    assert len(chunks) >= 1, (
        "Expected at least one chunk. Has scripts/ingest_pdfs.py been run?"
    )

    top_chunk = chunks[0]
    assert "chunk_text"      in top_chunk
    assert "source_pdf"      in top_chunk
    assert "page_number"     in top_chunk
    assert "relevance_score" in top_chunk

    combined_text = " ".join(c["chunk_text"] for c in chunks).lower()
    assert any(kw in combined_text for kw in ["business class", "clearance", "entitle"]), (
        f"Expected travel policy content in top chunks, got: {combined_text[:300]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Data Scientist with real CSV (real OpenAI call)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_data_scientist_real():
    """
    Calls data_scientist_worker with a real Task against the employees.csv fixture.
    Makes a real OpenAI API call to generate the pandas query.
    Confirms TaskResult.success=True and result_value contains 'Noa Levi'.
    """
    from agents.data_scientist import data_scientist_worker
    from core.state import AgentState, Task

    task: Task = {
        "task_id":     "t1",
        "worker_type": "data_scientist",
        "description": "Find the full name and clearance level of the employee named Noa Levi",
        "source_id":   "employees",
    }

    state: AgentState = AgentState(
        original_query="What is Noa Levi's clearance level?",
        session_id="integration-test",
        conversation_history=[],
        plan=[task],
        manifest_context="",
        task_results={},
        sources_used=[],
        retrieved_chunks=[],
        draft_answer="",
        audit_result={"verdict": "PASS", "notes": ""},
        retry_count=0,
        retry_notes="",
        final_answer="",
        final_sources=[],
    )

    result = asyncio.run(data_scientist_worker(state, task))

    assert result["task_id"]     == "t1"
    assert result["worker_type"] == "data_scientist"
    assert result["success"] is True, (
        f"data_scientist_worker failed. error: {result.get('error')}\n"
        f"output: {result.get('output')}"
    )
    assert result["error"] is None

    output = json.loads(result["output"])
    result_value = output.get("result_value", "")
    result_str   = json.dumps(result_value).lower()
    assert "noa levi" in result_str, (
        f"Expected 'Noa Levi' in result_value, got: {result_str[:300]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Full graph, happy path (real OpenAI call)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_graph_pass():
    """
    Runs the full compiled graph with a clear, answerable query.
    Uses real OpenAI API for all agents.
    Confirms:
      - final_answer is non-empty
      - final_sources is non-empty
      - retry_count is 0 or 1 (passes on first or second attempt)
      - no exception is raised
    """
    from graph import compiled_graph

    initial_state = _make_initial_state("What is Noa Levi's clearance level?")

    final_state = asyncio.run(
        compiled_graph.ainvoke(
            initial_state,
            config={"recursion_limit": 25},
        )
    )

    assert final_state["final_answer"], (
        "final_answer is empty — graph did not produce an answer"
    )
    assert len(final_state["final_sources"]) > 0, (
        "final_sources is empty — no sources were attributed"
    )
    assert final_state["retry_count"] <= 1, (
        f"Expected 0 or 1 retries for a clear query, got {final_state['retry_count']}"
    )

    # Answer should mention clearance level A (Noa's actual level in the fixture)
    answer_lower = final_state["final_answer"].lower()
    assert any(kw in answer_lower for kw in ["clearance", "level a", "level-a"]), (
        f"Expected clearance level mention in answer: {final_state['final_answer']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Full graph, retry / graceful failure (real OpenAI call)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_graph_retry():
    """
    Runs the full compiled graph with a deliberately vague query that is
    likely to be incomplete or trigger an Auditor FAIL.
    Confirms the system:
      - Never raises an exception
      - Always returns either a non-empty final_answer OR the configured failure message
      - retry_count is between 0 and 3 (inclusive)
    The test does not assert answer correctness — only that the system degrades gracefully.
    """
    from agents.chat import _FAILURE_MESSAGE
    from graph import compiled_graph

    # Vague query — references data sources loosely, likely to cause partial results
    initial_state = _make_initial_state(
        "Tell me everything about all employees, all salary ranges, "
        "all travel rules, and all hotel policies in full detail."
    )

    try:
        final_state = asyncio.run(
            compiled_graph.ainvoke(
                initial_state,
                config={"recursion_limit": 25},
            )
        )
    except Exception as exc:
        pytest.fail(f"Graph raised an unexpected exception: {exc}")

    assert 0 <= final_state["retry_count"] <= 3, (
        f"retry_count out of expected range: {final_state['retry_count']}"
    )

    final_answer = final_state["final_answer"]
    assert final_answer, "final_answer must never be empty — either answer or failure message"
    assert final_answer == _FAILURE_MESSAGE or len(final_answer) > 20, (
        f"final_answer is unexpectedly short or wrong: {final_answer!r}"
    )

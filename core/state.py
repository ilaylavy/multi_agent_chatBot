"""
core/state.py — Shared state schema and agent view functions.

Full AgentState TypedDict is the single LangGraph container.
View functions are the privacy layer: each agent's LLM prompt
is built from its view — never from the raw state.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Task(TypedDict):
    task_id:     str
    worker_type: str   # "librarian" | "data_scientist"
    description: str
    source_id:   str


class TaskResult(TypedDict):
    task_id:     str
    worker_type: str
    output:      str   # serialised result from the worker
    success:     bool
    error:       Optional[str]


class SourceRef(TypedDict):
    source_id:   str
    source_type: str   # "pdf" | "csv" | "sqlite"
    label:       str   # human-readable name


class Chunk(TypedDict):
    chunk_text:       str
    source_pdf:       str
    page_number:      int
    relevance_score:  float


class AuditResult(TypedDict):
    verdict: Literal["PASS", "FAIL"]
    notes:   str


# ---------------------------------------------------------------------------
# Full State Schema — the one LangGraph container
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Input
    original_query:       str
    session_id:           str
    conversation_history: List[Message]

    # Planning
    plan:                 List[Task]
    manifest_context:     str

    # Execution
    task_results:         Dict[str, TaskResult]
    sources_used:         List[SourceRef]
    retrieved_chunks:     List[Chunk]          # RAGAS logging

    # Synthesis & Audit
    draft_answer:         str
    audit_result:         AuditResult          # PASS | FAIL + notes
    retry_count:          int                  # Max 3
    retry_notes:          str

    # Output
    final_answer:         str
    final_sources:        List[SourceRef]


# ---------------------------------------------------------------------------
# Agent View Functions — privacy layer
# Each function returns only the fields that agent's LLM prompt may see.
# ---------------------------------------------------------------------------

def chat_agent_view(state: AgentState) -> dict:
    """Chat Agent sees: original_query, conversation_history, final_answer, final_sources."""
    return {
        "original_query":       state["original_query"],
        "conversation_history": state["conversation_history"],
        "final_answer":         state["final_answer"],
        "final_sources":        state["final_sources"],
    }


def planner_view(state: AgentState) -> dict:
    """Planner sees: original_query, manifest_context, retry_notes (on retry only)."""
    view: dict = {
        "original_query":   state["original_query"],
        "manifest_context": state["manifest_context"],
    }
    if state.get("retry_notes"):
        view["retry_notes"] = state["retry_notes"]
    return view


def router_view(state: AgentState) -> dict:
    """Router sees: plan (task list only)."""
    return {
        "plan": state["plan"],
    }


def librarian_view(state: AgentState, task: Task, manifest_detail: str) -> dict:
    """
    Librarian sees: its single assigned task + manifest detail for its source only.
    Callers must pass the resolved task object and the pre-filtered manifest detail string.
    """
    return {
        "task":            task,
        "manifest_detail": manifest_detail,
    }


def data_scientist_view(state: AgentState, task: Task, manifest_detail: str) -> dict:
    """
    Data Scientist sees: its single assigned task + manifest detail for its table only.
    Callers must pass the resolved task object and the pre-filtered manifest detail string.
    """
    return {
        "task":            task,
        "manifest_detail": manifest_detail,
    }


def synthesizer_view(state: AgentState) -> dict:
    """Synthesizer sees: original_query, plan, task_results, sources_used."""
    return {
        "original_query": state["original_query"],
        "plan":           state["plan"],
        "task_results":   state["task_results"],
        "sources_used":   state["sources_used"],
    }


def auditor_view(state: AgentState) -> dict:
    """Auditor sees: original_query, plan, draft_answer, sources_used."""
    return {
        "original_query": state["original_query"],
        "plan":           state["plan"],
        "draft_answer":   state["draft_answer"],
        "sources_used":   state["sources_used"],
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python core/state.py
# ---------------------------------------------------------------------------

def test_state():
    fake_state: AgentState = {
        "original_query":       "What was revenue in Q3?",
        "session_id":           "test-session-001",
        "conversation_history": [{"role": "user", "content": "What was revenue in Q3?"}],
        "plan":                 [{"task_id": "t1", "worker_type": "data_scientist",
                                  "description": "Look up Q3 revenue", "source_id": "financials"}],
        "manifest_context":     "financials: quarterly revenue table",
        "task_results":         {"t1": {"task_id": "t1", "worker_type": "data_scientist",
                                        "output": "Q3 revenue: $4.2M", "success": True, "error": None}},
        "sources_used":         [{"source_id": "financials", "source_type": "csv", "label": "Financials CSV"}],
        "retrieved_chunks":     [],
        "draft_answer":         "Revenue in Q3 was $4.2M.",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "final_answer":         "Revenue in Q3 was $4.2M.",
        "final_sources":        [{"source_id": "financials", "source_type": "csv", "label": "Financials CSV"}],
    }

    task = fake_state["plan"][0]
    manifest_detail = "financials: columns = [quarter, revenue, expenses]"

    assert list(chat_agent_view(fake_state).keys()) == [
        "original_query", "conversation_history", "final_answer", "final_sources"
    ]
    assert "retry_notes" not in planner_view(fake_state)   # empty string → omitted
    assert list(router_view(fake_state).keys()) == ["plan"]
    assert list(librarian_view(fake_state, task, manifest_detail).keys()) == ["task", "manifest_detail"]
    assert list(data_scientist_view(fake_state, task, manifest_detail).keys()) == ["task", "manifest_detail"]
    assert list(synthesizer_view(fake_state).keys()) == [
        "original_query", "plan", "task_results", "sources_used"
    ]
    assert list(auditor_view(fake_state).keys()) == [
        "original_query", "plan", "draft_answer", "sources_used"
    ]

    # Verify planner includes retry_notes when present
    fake_state["retry_notes"] = "Previous answer was incomplete."
    assert "retry_notes" in planner_view(fake_state)

    print("PASS: all view functions return correct fields")


if __name__ == "__main__":
    test_state()

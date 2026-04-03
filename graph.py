"""
graph.py — LangGraph wired graph.

5 graph nodes, real agent imports:
  chat_node, planner_node, router_node, synthesizer_node, auditor_node

Librarian and Data Scientist are worker callables dispatched inside
router_node via asyncio.gather — they are NOT graph nodes.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.chat import chat_node
from agents.planner import planner_node
from agents.router import router_node
from agents.synthesizer import synthesizer_node
from agents.auditor import auditor_node
from core.state import AgentState, AuditResult


# ---------------------------------------------------------------------------
# Routing function for auditor conditional edge
# ---------------------------------------------------------------------------

def route_after_chat(
    state: AgentState,
) -> Literal["planner_node", "__end__"]:
    """
    final_answer empty  → planner_node  (initial entry — pipeline not started yet)
    final_answer set    → END           (answer delivered, or retry exhaustion)
    """
    if state["final_answer"]:
        return END
    return "planner_node"


def route_after_audit(
    state: AgentState,
) -> Literal["chat_node", "planner_node"]:
    """
    PASS                → chat_node    (release to user)
    FAIL + retries < 3  → planner_node (retry cycle)
    FAIL + retries >= 3 → chat_node    (graceful failure)
    """
    audit: AuditResult = state["audit_result"]
    if audit["verdict"] == "PASS":
        return "chat_node"
    if state["retry_count"] < 3:
        return "planner_node"
    return "chat_node"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 5 graph nodes — librarian and data_scientist are registry workers, not nodes
    graph.add_node("chat_node",        chat_node)
    graph.add_node("planner_node",     planner_node)
    graph.add_node("router_node",      router_node)
    graph.add_node("synthesizer_node", synthesizer_node)
    graph.add_node("auditor_node",     auditor_node)

    # Linear edges
    graph.add_edge(START,          "chat_node")
    graph.add_edge("planner_node", "router_node")
    graph.add_edge("router_node",      "synthesizer_node")
    graph.add_edge("synthesizer_node", "auditor_node")

    # Conditional edge out of chat — initial entry vs final delivery/exhaustion
    graph.add_conditional_edges(
        "chat_node",
        route_after_chat,
        {"planner_node": "planner_node", END: END},
    )

    # Conditional edge out of auditor
    graph.add_conditional_edges(
        "auditor_node",
        route_after_audit,
        {
            "chat_node":    "chat_node",
            "planner_node": "planner_node",
        },
    )

    return graph


compiled_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# End-to-end test — run with: python -m graph
# ---------------------------------------------------------------------------

def test_graph_e2e():
    import json
    from contextlib import ExitStack
    from unittest.mock import AsyncMock, MagicMock, patch

    from core.manifest import get_manifest_index
    from core.state import TaskResult, SourceRef

    # ── Fake LLM responses per agent ─────────────────────────────
    planner_json = json.dumps({"tasks": [
        {"task_id": "t1", "worker_type": "data_scientist",
         "description": "Get Noa's clearance level from employees table",
         "source_id": "employees"},
        {"task_id": "t2", "worker_type": "librarian",
         "description": "Find flight class entitlements for clearance level A",
         "source_id": "travel_policy_2024"},
    ]})

    synthesizer_json = json.dumps({
        "draft_answer": (
            "Noa holds clearance level A. Per Section 1 of the Travel Policy 2024, "
            "employees with clearance level A are entitled to Business Class on "
            "flights exceeding 4 hours."
        ),
        "confidence": "high",
    })

    auditor_json = json.dumps({
        "verdict":       "PASS",
        "notes":         "",
        "failed_checks": [],
    })

    chat_formatted = (
        "Yes, Noa can fly Business Class. She holds clearance level A, which per "
        "Section 1 of the Travel Policy 2024 entitles her to Business Class on "
        "flights exceeding 4 hours."
    )

    def make_llm_mock(content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        llm  = MagicMock()
        llm.ainvoke = AsyncMock(return_value=resp)
        return llm

    # ── Fake worker callables for the router ─────────────────────
    ds_result = TaskResult(
        task_id="t1", worker_type="data_scientist",
        output=json.dumps({
            "result_value": [{"full_name": "Noa Levi", "clearance_level": "A"}],
            "query_used":   "df[df['full_name'] == 'Noa Levi'][['full_name','clearance_level']]",
            "table_name":   "employees.csv",
            "row_count":    1,
        }),
        success=True, error=None,
    )
    lib_result = TaskResult(
        task_id="t2", worker_type="librarian",
        output=json.dumps([{
            "chunk_text":      "Employees with clearance level A are entitled to Business Class on flights exceeding 4 hours.",
            "source_pdf":      "travel_policy.pdf",
            "page_number":     3,
            "relevance_score": 0.97,
        }]),
        success=True, error=None,
    )

    def make_worker(result: TaskResult):
        async def _worker(state, task):
            return result
        return _worker

    def mock_get_worker(worker_type: str):
        return make_worker(ds_result if worker_type == "data_scientist" else lib_result)

    # ── Initial state ─────────────────────────────────────────────
    initial_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-e2e-001",
        "conversation_history": [],
        "plan":                 [],
        "manifest_context":     get_manifest_index(),
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

    print("=== End-to-end graph trace ===")
    print(f"Query: {initial_state['original_query']}\n")

    async def _run():
        with ExitStack() as stack:
            stack.enter_context(patch("agents.planner.get_llm",     return_value=make_llm_mock(planner_json)))
            stack.enter_context(patch("agents.router.get_worker",    side_effect=mock_get_worker))
            stack.enter_context(patch("agents.synthesizer.get_llm",  return_value=make_llm_mock(synthesizer_json)))
            stack.enter_context(patch("agents.auditor.get_llm",      return_value=make_llm_mock(auditor_json)))
            stack.enter_context(patch("agents.chat.get_llm",         return_value=make_llm_mock(chat_formatted)))

            async for update in compiled_graph.astream(
                initial_state,
                stream_mode="updates",
                config={"recursion_limit": 25},
            ):
                for node_name, node_output in update.items():
                    print(f"[{node_name}]")
                    for k, v in node_output.items():
                        # Truncate long values for readability
                        v_str = str(v)
                        print(f"  {k}: {v_str[:120]}{'...' if len(v_str) > 120 else ''}")
                    print()

    asyncio.run(_run())

    # Structure note for the wiring pass
    print("=== Graph edges ===")
    for edge in compiled_graph.get_graph().edges:
        print(f"  {edge.source} -> {edge.target}")


if __name__ == "__main__":
    test_graph_e2e()

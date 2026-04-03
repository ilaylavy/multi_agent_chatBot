"""
graph.py — LangGraph skeleton.

5 graph nodes: chat, planner, router, synthesizer, auditor.
Librarian and Data Scientist are NOT graph nodes — they are worker
callables dispatched internally by router_node via asyncio.gather
using the Worker Registry (core/registry.py).

All nodes are stubs returning empty dicts until agents/ are built.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from core.state import AgentState, AuditResult


# ---------------------------------------------------------------------------
# Stub node functions — replaced when agents/ are built
# ---------------------------------------------------------------------------

async def chat_node(state: AgentState) -> dict:
    return {}


async def planner_node(state: AgentState) -> dict:
    return {}


async def router_node(state: AgentState) -> dict:
    # Will call librarian/data_scientist workers internally via asyncio.gather
    return {}


async def synthesizer_node(state: AgentState) -> dict:
    return {}


async def auditor_node(state: AgentState) -> dict:
    return {}


# ---------------------------------------------------------------------------
# Routing function for auditor conditional edge
# ---------------------------------------------------------------------------

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
    graph.add_edge(START,              "chat_node")
    graph.add_edge("chat_node",        "planner_node")
    graph.add_edge("planner_node",     "router_node")
    graph.add_edge("router_node",      "synthesizer_node")
    graph.add_edge("synthesizer_node", "auditor_node")

    # Conditional edge out of auditor
    graph.add_conditional_edges(
        "auditor_node",
        route_after_audit,
        {
            "chat_node":    "chat_node",
            "planner_node": "planner_node",
        },
    )

    # chat_node exits to END
    graph.add_edge("chat_node", END)

    return graph


compiled_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m graph
# ---------------------------------------------------------------------------

def test_graph():
    print("=== Compiled graph nodes ===")
    for node in compiled_graph.nodes:
        print(f"  {node}")

    print("\n=== Compiled graph edges ===")
    draw = compiled_graph.get_graph()
    for edge in draw.edges:
        print(f"  {edge.source} -> {edge.target}")

    expected_nodes = {
        "__start__",
        "chat_node",
        "planner_node",
        "router_node",
        "synthesizer_node",
        "auditor_node",
    }
    assert expected_nodes.issubset(set(str(n) for n in compiled_graph.nodes)), \
        "One or more expected nodes are missing"

    absent = {"librarian_node", "data_scientist_node"}
    actual = set(str(n) for n in compiled_graph.nodes)
    for name in absent:
        assert name not in actual, f"{name} should not be a graph node"

    print("\nPASS: 5 graph nodes present, librarian/data_scientist correctly absent")


if __name__ == "__main__":
    test_graph()

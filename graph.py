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
    DIRECT or CLARIFY              → END          (answer already in final_answer)
    PLAN + final_answer empty      → planner_node (initial classify, pipeline not started)
    PLAN + final_answer non-empty  → END          (format/deliver done, or retry exhaustion)
    """
    intent = state.get("chat_intent", "")
    if intent in ("DIRECT", "CLARIFY"):
        return END
    if intent == "PLAN" and not state["final_answer"]:
        return "planner_node"
    return END


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
# Routing unit tests
# ---------------------------------------------------------------------------

def test_routing():
    """Unit tests for route_after_chat and route_after_audit."""

    # Minimal fake state — only the fields the routing functions read
    _base: dict = {
        "chat_intent":  "",
        "final_answer": "",
        "audit_result": {"verdict": "PASS", "notes": ""},
        "retry_count":  0,
    }

    # ── route_after_chat ──────────────────────────────────────────

    # DIRECT → END
    result = route_after_chat({**_base, "chat_intent": "DIRECT", "final_answer": "Hello!"})
    assert result == END, f"DIRECT must route to END, got {result!r}"
    print("PASS: route_after_chat — DIRECT routes to END")

    # CLARIFY → END
    result = route_after_chat({**_base, "chat_intent": "CLARIFY", "final_answer": "Which Noa?"})
    assert result == END, f"CLARIFY must route to END, got {result!r}"
    print("PASS: route_after_chat — CLARIFY routes to END")

    # PLAN, final_answer empty → planner_node
    result = route_after_chat({**_base, "chat_intent": "PLAN", "final_answer": ""})
    assert result == "planner_node", f"PLAN (initial) must route to planner_node, got {result!r}"
    print("PASS: route_after_chat — PLAN (initial entry) routes to planner_node")

    # PLAN, final_answer set (format/deliver complete) → END
    result = route_after_chat({**_base, "chat_intent": "PLAN", "final_answer": "Formatted answer."})
    assert result == END, f"PLAN (after delivery) must route to END, got {result!r}"
    print("PASS: route_after_chat — PLAN (after delivery) routes to END")

    # Empty intent (exhaustion sets final_answer, no intent change) → END
    result = route_after_chat({**_base, "chat_intent": "", "final_answer": "I could not verify..."})
    assert result == END, f"Empty intent must route to END, got {result!r}"
    print("PASS: route_after_chat — empty intent routes to END")

    # ── route_after_audit ─────────────────────────────────────────

    # PASS → chat_node
    result = route_after_audit({**_base, "audit_result": {"verdict": "PASS", "notes": ""}, "retry_count": 0})
    assert result == "chat_node", f"PASS must route to chat_node, got {result!r}"
    print("PASS: route_after_audit — PASS routes to chat_node")

    # FAIL + retries remaining → planner_node
    result = route_after_audit({**_base, "audit_result": {"verdict": "FAIL", "notes": "bad"}, "retry_count": 1})
    assert result == "planner_node", f"FAIL with retries must route to planner_node, got {result!r}"
    print("PASS: route_after_audit — FAIL with retries routes to planner_node")

    # FAIL + retry_count >= 3 → chat_node (graceful failure)
    result = route_after_audit({**_base, "audit_result": {"verdict": "FAIL", "notes": "bad"}, "retry_count": 3})
    assert result == "chat_node", f"FAIL exhausted must route to chat_node, got {result!r}"
    print("PASS: route_after_audit — FAIL exhausted routes to chat_node")

    print("\nPASS: all routing tests passed")


# ---------------------------------------------------------------------------
# End-to-end test — run with: python -m graph
# ---------------------------------------------------------------------------

def test_graph_e2e():
    import json
    from contextlib import ExitStack
    from unittest.mock import AsyncMock, MagicMock, patch

    from core.manifest import get_manifest_index
    from core.state import TaskResult, SourceRef

    # ── Fake LLM responses — each embeds a unique marker ─────────
    # Markers let us assert each mock was actually invoked and its
    # output flowed through the graph into the accumulated state.
    #
    # Auditor exception: it hardcodes notes="" on PASS — nothing from
    # the LLM notes field survives. Its marker is structural: it is the
    # ONLY node that writes the "audit_result" key, so we collect output
    # as "field=value" strings and assert "audit_result=" appears.
    planner_json = json.dumps({"tasks": [
        {"task_id": "t1", "worker_type": "data_scientist",
         "description": "MARKER_PLANNER Get Noa's clearance level from employees table",
         "source_ids": ["employees"]},
        {"task_id": "t2", "worker_type": "librarian",
         "description": "Find flight class entitlements for clearance level A",
         "source_ids": ["travel_policy_2024"]},
    ]})

    synthesizer_json = json.dumps({
        "draft_answer": (
            "MARKER_SYNTHESIZER Noa holds clearance level A. Per Section 1 of the "
            "Travel Policy 2024, employees with clearance level A are entitled to "
            "Business Class on flights exceeding 4 hours."
        ),
        "confidence": "high",
    })

    auditor_json = json.dumps({
        "verdict":       "PASS",
        "notes":         "",
        "failed_checks": [],
    })

    chat_formatted = (
        "MARKER_CHAT Yes, Noa can fly Business Class. She holds clearance level A, "
        "which per Section 1 of the Travel Policy 2024 entitles her to Business Class "
        "on flights exceeding 4 hours."
    )

    # chat_node now makes two LLM calls per pipeline run:
    #   call 1 (initial entry): classify intent — must return JSON
    #   call 2 (format+deliver): format the auditor-verified answer — returns plain text
    chat_classify_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": "Can Noa fly Business Class?",
        "response":        None,
    })

    def make_llm_mock(content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        llm  = MagicMock()
        llm.ainvoke = AsyncMock(return_value=resp)
        llm.bind.return_value = llm   # .bind(max_tokens=...) returns self
        return llm

    def make_chat_llm_mock() -> MagicMock:
        """Returns a mock that serves classify JSON on call 1, formatted text on call 2."""
        responses = [
            MagicMock(content=chat_classify_json),
            MagicMock(content=chat_formatted),
        ]
        call_count = {"n": 0}

        async def _side_effect(messages):
            idx = call_count["n"]
            call_count["n"] += 1
            return responses[idx] if idx < len(responses) else responses[-1]

        llm = MagicMock()
        llm.ainvoke = _side_effect
        return llm

    # ── Fake worker callables for the router ─────────────────────
    # ds_result embeds MARKER_ROUTER in query_used — it flows into task_results
    ds_result = TaskResult(
        task_id="t1", worker_type="data_scientist",
        output=json.dumps({
            "result_value": [{"full_name": "Noa Levi", "clearance_level": "A"}],
            "query_used":   "MARKER_ROUTER df[df['full_name'] == 'Noa Levi'][['full_name','clearance_level']]",
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
        "chat_intent":          "",
        "rewritten_query":      "",
        "plan":                 [],
        "manifest_context":     get_manifest_index(),
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "synthesizer_output":   "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "final_answer":         "",
        "final_sources":        [],
    }

    EXPECTED_NODES = {
        "chat_node", "planner_node", "router_node",
        "synthesizer_node", "auditor_node",
    }
    EXPECTED_MARKERS = {
        "MARKER_PLANNER",     # planner LLM mock → task description → plan field
        "MARKER_ROUTER",      # ds_result.output → task_results field
        "MARKER_SYNTHESIZER", # synthesizer LLM mock → draft_answer field
        "audit_result=",      # auditor hardcodes notes="" on PASS; unique because only
                              # auditor_node writes the "audit_result" key — collected as
                              # "field=value" strings so the key itself is the marker
        "MARKER_CHAT",        # chat LLM mock → final_answer field
    }

    print("=== End-to-end graph trace ===")
    print(f"Query: {initial_state['original_query']}\n")

    async def _run():
        fired_nodes:      set[str]  = set()
        collected_output: list[str] = []          # stringified value of every node output field
        accumulated_state: dict     = dict(initial_state)

        with ExitStack() as stack:
            stack.enter_context(patch("agents.planner.get_llm",     return_value=make_llm_mock(planner_json)))
            stack.enter_context(patch("agents.router.get_worker",    side_effect=mock_get_worker))
            stack.enter_context(patch("agents.synthesizer.get_llm",  return_value=make_llm_mock(synthesizer_json)))
            stack.enter_context(patch("agents.auditor.get_llm",      return_value=make_llm_mock(auditor_json)))
            stack.enter_context(patch("agents.chat.get_llm",         return_value=make_chat_llm_mock()))

            async for update in compiled_graph.astream(
                initial_state,
                stream_mode="updates",
                config={"recursion_limit": 25},
            ):
                for node_name, node_output in update.items():
                    fired_nodes.add(node_name)
                    print(f"[{node_name}]")
                    for k, v in node_output.items():
                        v_str = str(v)
                        collected_output.append(f"{k}={v_str}")   # key= prefix enables structural markers
                        accumulated_state[k] = v
                        print(f"  {k}: {v_str[:120]}{'...' if len(v_str) > 120 else ''}")
                    print()

        return fired_nodes, collected_output, accumulated_state

    fired_nodes, collected_output, final_state = asyncio.run(_run())

    # ── Assertions ────────────────────────────────────────────────
    all_output = " ".join(collected_output)

    # 1. All 5 graph nodes must have fired
    missing_nodes = EXPECTED_NODES - fired_nodes
    assert not missing_nodes, (
        f"These graph nodes never fired: {missing_nodes}\n"
        f"Nodes that did fire: {fired_nodes}"
    )

    # 2. All 5 markers must appear in the collected node outputs
    for marker in EXPECTED_MARKERS:
        assert marker in all_output, (
            f"Marker '{marker}' not found in any node output — "
            f"the corresponding node may not have run or its mock was bypassed"
        )

    # 3. final_answer is a non-empty string
    final_answer = final_state.get("final_answer", "")
    assert isinstance(final_answer, str) and final_answer, (
        f"final_answer must be a non-empty string, got: {final_answer!r}"
    )

    # 4. final_sources is a list
    final_sources = final_state.get("final_sources")
    assert isinstance(final_sources, list), (
        f"final_sources must be a list, got: {type(final_sources).__name__}"
    )

    # 5. retry_count is an integer between 0 and 3
    retry_count = final_state.get("retry_count", -1)
    assert isinstance(retry_count, int) and 0 <= retry_count <= 3, (
        f"retry_count must be an integer 0–3, got: {retry_count!r}"
    )

    print("=== Graph edges ===")
    for edge in compiled_graph.get_graph().edges:
        print(f"  {edge.source} -> {edge.target}")

    print("\nPASS: all graph assertions passed")
    print(f"  Nodes fired  : {sorted(fired_nodes)}")
    print(f"  Markers found: {sorted(EXPECTED_MARKERS)}")
    print(f"  final_answer : {final_answer[:80]}{'...' if len(final_answer) > 80 else ''}")
    print(f"  final_sources: {final_sources}")
    print(f"  retry_count  : {retry_count}")


if __name__ == "__main__":
    test_routing()
    print()
    test_graph_e2e()

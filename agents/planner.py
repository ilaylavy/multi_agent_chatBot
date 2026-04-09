"""
agents/planner.py — Planner agent.

Decomposes the user query into an ordered list of Tasks, each assigned
to a worker type (librarian | data_scientist) and a specific source_id.

View    : original_query, manifest_context, retry_notes (only on retry)
Returns : { plan: List[Task] }
"""

from __future__ import annotations

import asyncio
import json
import logging

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.state import AgentState, Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query planner for a multi-agent RAG system.
Your job is to reason through the user's question and then decompose it into
an ordered list of tasks. Each task must be assigned to exactly one worker
and one data source.

Available worker types:
  - librarian      : searches PDF documents (unstructured text)
  - data_scientist : queries CSV or SQLite tables (structured data)

You will be given:
  - The user's question.
  - A manifest listing all available data sources with their IDs, summaries,
    and a contains field listing what types of information each source holds.

== PHASE 1: REASONING (required before generating tasks) ==

Think through these four questions explicitly:
  1. What specific pieces of information are needed to answer this query?
  2. For each piece, is it a factual lookup from structured data (table)
     or a rule/policy from a document (PDF)?
  3. Which manifest source contains each piece? Use the contains field
     and source summaries to justify each assignment. If a question is about
     access control or data classification requirements, check it_security_policy
     not salary_bands. If a question is about leave or performance, check
     hr_handbook_v3 not employees.
  4. What dependencies exist between tasks? Does any task need the output
     of another before it can run?

== PHASE 2: PLAN ==

Only after completing the reasoning, output the task list.

Rules:
  - Only assign tasks to sources listed in the manifest.
  - Use the minimum number of tasks required to answer the question.
  - task_id values must be unique strings: t1, t2, t3, ...
  - worker_type must be exactly "librarian" or "data_scientist".
  - source_id must exactly match an id from the manifest.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "reasoning": {
    "information_needed": ["list of specific facts or rules needed"],
    "source_assignments": [
      {
        "info": "what is needed",
        "source_id": "exact_id_from_manifest",
        "worker_type": "librarian or data_scientist",
        "justification": "why this source — reference its summary"
      }
    ],
    "dependencies": ["e.g. t2 needs t1 output because ..."]
  },
  "tasks": [
    {
      "task_id": "t1",
      "worker_type": "librarian" | "data_scientist",
      "description": "what this task should find or retrieve",
      "source_id": "exact_id_from_manifest",
      "depends_on": null | "t1"
    }
  ]
}

depends_on rules:
  - Set depends_on to null if the task can run independently.
  - Set depends_on to the task_id of a prerequisite task if this task requires
    the output of that task to be useful (e.g. first retrieve an employee's role,
    then look up policy rules for that role).
  - A task may depend on at most one other task.
  - Never create circular dependencies.

Cross-table questions:
  When a question requires comparing or joining data across multiple sources,
  create multiple tasks with correct dependencies. Never assign a cross-table
  question to a single task against one source.

  Example — "Which employee has the highest salary?":
    t1: data_scientist against salary_bands — find the clearance_level and
        department with the highest salary_max.
    t2 (depends_on t1): data_scientist against employees — find the employee
        whose clearance_level and department match the result from t1.
"""

_RETRY_NOTE_SECTION = """\

RETRY CONTEXT — a previous answer was rejected. Use these notes to plan better:
{retry_notes}
"""

_USER_TEMPLATE = """\
AVAILABLE DATA SOURCES:
{manifest_context}

USER QUESTION:
{original_query}
{retry_section}"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def planner_view(state: AgentState) -> dict:
    """
    Returns only the fields the Planner's LLM prompt may see.
    When rewritten_query is non-empty it takes precedence over original_query
    so the Planner plans against the context-enriched version of the question.
    retry_notes is included only when non-empty AND retry_count > 0.
    """
    view = {
        "original_query":   state.get("rewritten_query") or state["original_query"],
        "manifest_context": state["manifest_context"],
    }
    if state.get("retry_count", 0) > 0 and state.get("retry_notes", ""):
        view["retry_notes"] = state["retry_notes"]
    return view


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def planner_node(state: AgentState) -> dict:
    view = planner_view(state)

    retry_section = ""
    if "retry_notes" in view:
        retry_section = _RETRY_NOTE_SECTION.format(retry_notes=view["retry_notes"])

    user_message = _USER_TEMPLATE.format(
        manifest_context=view["manifest_context"],
        original_query=view["original_query"],
        retry_section=retry_section,
    )

    llm = get_llm("planner")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    data = parse_llm_json(response.content)
    try:
        tasks: list[Task] = [
            Task(
                task_id=t["task_id"],
                worker_type=t["worker_type"],
                description=t["description"],
                source_id=t["source_id"],
                depends_on=t.get("depends_on"),
            )
            for t in data["tasks"]
        ]
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    # Log reasoning for trace visibility
    reasoning = data.get("reasoning", {})
    reasoning_str = json.dumps(reasoning, indent=2) if reasoning else ""

    return {"plan": tasks, "planner_reasoning": reasoning_str}


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.planner
# ---------------------------------------------------------------------------

def test_planner():
    import os
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import PLANNER_STATE, PLANNER_RETRY_STATE

    # ── Fake LLM response ────────────────────────────────────────
    fake_llm_output = json.dumps({
        "reasoning": {
            "information_needed": [
                "Noa's clearance level from the employee table",
                "Flight class entitlements for that clearance level from the travel policy"
            ],
            "source_assignments": [
                {
                    "info": "Noa's clearance level",
                    "source_id": "employees",
                    "worker_type": "data_scientist",
                    "justification": "employees table contains clearance levels for each employee"
                },
                {
                    "info": "Flight class entitlements by clearance level",
                    "source_id": "travel_policy_2024",
                    "worker_type": "librarian",
                    "justification": "travel_policy_2024 PDF contains flight class entitlements by clearance level"
                }
            ],
            "dependencies": ["t2 needs t1 output because we need Noa's clearance level to look up entitlements"]
        },
        "tasks": [
            {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "description": "Look up Noa's clearance level in the employee table",
                "source_id":   "employees",
                "depends_on":  None,
            },
            {
                "task_id":     "t2",
                "worker_type": "librarian",
                "description": "Find flight class entitlements for clearance level A",
                "source_id":   "travel_policy_2024",
                "depends_on":  "t1",
            },
        ]
    })

    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    # ── Test 1: view excludes retry_notes when retry_count == 0 ──
    view = planner_view(PLANNER_STATE)
    assert "retry_notes" not in view, "retry_notes must be absent when retry_count == 0"
    assert "original_query" in view
    assert "manifest_context" in view
    # When rewritten_query is absent/empty, original_query is used
    assert view["original_query"] == PLANNER_STATE["original_query"]
    print("PASS: planner_view excludes retry_notes on first pass; uses original_query when rewritten_query is empty")

    # ── Test 1b: view uses rewritten_query when non-empty ─────────
    state_with_rewrite = {**PLANNER_STATE, "rewritten_query": "Can Noa Levi travel Business Class internationally?"}
    rewrite_view = planner_view(state_with_rewrite)
    assert rewrite_view["original_query"] == "Can Noa Levi travel Business Class internationally?", \
        "Planner must use rewritten_query when it is non-empty"
    print("PASS: planner_view uses rewritten_query when non-empty")

    # ── Test 2: view includes retry_notes on retry ────────────────
    retry_view = planner_view(PLANNER_RETRY_STATE)
    assert "retry_notes" in retry_view, "retry_notes must be present when retry_count > 0"
    print("PASS: planner_view includes retry_notes on retry")

    # patch target must match the module's __name__ at runtime:
    # "agents.planner" when imported, "__main__" when run directly
    patch_target = f"{__name__}.get_llm"

    # ── Test 3: node returns correct plan ────────────────────────
    with patch(patch_target, return_value=mock_llm):
        result = asyncio.run(planner_node(PLANNER_STATE))

    assert "plan" in result, "Result must contain 'plan'"
    assert len(result["plan"]) == 2
    assert result["plan"][0]["task_id"] == "t1"
    assert result["plan"][0]["worker_type"] == "data_scientist"
    assert result["plan"][0]["source_id"] == "employees"
    assert result["plan"][0]["depends_on"] is None
    assert result["plan"][1]["task_id"] == "t2"
    assert result["plan"][1]["worker_type"] == "librarian"
    assert result["plan"][1]["source_id"] == "travel_policy_2024"
    assert result["plan"][1]["depends_on"] == "t1"
    assert set(result.keys()) == {"plan", "planner_reasoning"}, "Node must return plan + planner_reasoning"
    assert isinstance(result["planner_reasoning"], str)
    assert len(result["planner_reasoning"]) > 0, "planner_reasoning must be non-empty when reasoning is present"
    print(f"PASS: planner_node returns correct plan with reasoning")

    # ── Test 4: bad JSON raises ValueError with raw output ────────
    bad_response = MagicMock()
    bad_response.content = "sorry, I cannot help with that"
    mock_llm.ainvoke = AsyncMock(return_value=bad_response)

    with patch(patch_target, return_value=mock_llm):
        try:
            asyncio.run(planner_node(PLANNER_STATE))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "sorry, I cannot help with that" in str(exc)
            print(f"PASS: ValueError raised with raw output on bad JSON")

    # ── Test 5: cross-table query produces ≥2 tasks with a dependency ──
    cross_table_output = json.dumps({
        "reasoning": {
            "information_needed": ["highest salary_max from salary_bands", "employee name matching that band"],
            "source_assignments": [
                {"info": "highest salary_max", "source_id": "salary_bands", "worker_type": "data_scientist",
                 "justification": "salary_bands contains min/max salary ranges"},
                {"info": "employee name", "source_id": "employees", "worker_type": "data_scientist",
                 "justification": "employees table has names and clearance levels"}
            ],
            "dependencies": ["t2 needs t1 to know which clearance_level/department to match"]
        },
        "tasks": [
            {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "description": "Find the clearance_level and department with the highest salary_max from salary_bands",
                "source_id":   "salary_bands",
                "depends_on":  None,
            },
            {
                "task_id":     "t2",
                "worker_type": "data_scientist",
                "description": "Find the employee name in employees whose clearance_level and department match t1 result",
                "source_id":   "employees",
                "depends_on":  "t1",
            },
        ]
    })
    mock_response_cross = MagicMock()
    mock_response_cross.content = cross_table_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_cross)

    cross_state = {
        **PLANNER_STATE,
        "original_query": "Which employee has the highest salary?",
        "manifest_context": """
pdfs: []
tables:
  - id: employees
    summary: Master employee list with names, departments, and clearance levels.
  - id: salary_bands
    summary: Salary bands by department and clearance level with min/max ranges.
""",
    }

    with patch(patch_target, return_value=mock_llm):
        result_cross = asyncio.run(planner_node(cross_state))

    plan = result_cross["plan"]
    assert len(plan) >= 2, \
        f"Cross-table query must produce at least 2 tasks, got {len(plan)}"
    source_ids = {t["source_id"] for t in plan}
    assert len(source_ids) >= 2, \
        f"Cross-table plan must reference at least 2 different sources, got {source_ids}"
    deps = [t["depends_on"] for t in plan if t["depends_on"] is not None]
    assert len(deps) >= 1, \
        "Cross-table plan must have at least one task with a dependency"
    assert plan[1]["depends_on"] == "t1", \
        f"Second task must depend on t1, got depends_on={plan[1]['depends_on']!r}"
    print(f"PASS: cross-table query produces {len(plan)} tasks across {source_ids} with dependency chain")

    # ── Test 6: PDF query uses librarian, table query uses data_scientist, reasoning has justifications ──
    mixed_output = json.dumps({
        "reasoning": {
            "information_needed": [
                "Noa's department from the employees table",
                "Password rotation policy from the IT security policy document"
            ],
            "source_assignments": [
                {
                    "info": "Noa's department",
                    "source_id": "employees",
                    "worker_type": "data_scientist",
                    "justification": "employees table contains department information for each employee"
                },
                {
                    "info": "Password rotation policy",
                    "source_id": "it_security_policy",
                    "worker_type": "librarian",
                    "justification": "it_security_policy PDF contains password requirements and rotation rules"
                }
            ],
            "dependencies": []
        },
        "tasks": [
            {
                "task_id": "t1",
                "worker_type": "data_scientist",
                "description": "Look up Noa's department from the employees table",
                "source_id": "employees",
                "depends_on": None,
            },
            {
                "task_id": "t2",
                "worker_type": "librarian",
                "description": "Find the password rotation policy",
                "source_id": "it_security_policy",
                "depends_on": None,
            },
        ]
    })
    mock_response_mixed = MagicMock()
    mock_response_mixed.content = mixed_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_mixed)

    mixed_state = {
        **PLANNER_STATE,
        "original_query": "What department is Noa in and what is the password rotation policy?",
        "manifest_context": """
pdfs:
  - id: it_security_policy
    summary: IT Security Policy covering password requirements, data classification, and incidents.
tables:
  - id: employees
    summary: Master employee list with names, departments, and clearance levels.
""",
    }

    with patch(patch_target, return_value=mock_llm):
        result_mixed = asyncio.run(planner_node(mixed_state))

    plan_m = result_mixed["plan"]
    # Table task must use data_scientist
    table_task = next(t for t in plan_m if t["source_id"] == "employees")
    assert table_task["worker_type"] == "data_scientist", \
        f"Table source must use data_scientist, got {table_task['worker_type']}"
    # PDF task must use librarian
    pdf_task = next(t for t in plan_m if t["source_id"] == "it_security_policy")
    assert pdf_task["worker_type"] == "librarian", \
        f"PDF source must use librarian, got {pdf_task['worker_type']}"
    # Reasoning must include justifications referencing the sources
    reasoning_str = result_mixed["planner_reasoning"]
    reasoning_obj = json.loads(reasoning_str)
    justifications = [sa["justification"] for sa in reasoning_obj["source_assignments"]]
    assert any("employees" in j for j in justifications), \
        "Reasoning must include justification referencing the employees table"
    assert any("it_security_policy" in j or "password" in j.lower() for j in justifications), \
        "Reasoning must include justification referencing the IT security policy PDF"
    print("PASS: PDF source uses librarian, table uses data_scientist, reasoning has justified source assignments")

    print("\nPASS: all planner tests passed")


if __name__ == "__main__":
    test_planner()

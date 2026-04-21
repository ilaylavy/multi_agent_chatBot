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
from core.manifest_prefilter import prefilter_manifest, _prefilter_traces
from core.parse import parse_llm_json
from core.state import AgentState, Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query planner. Decompose the user's question into tasks. Each task \
retrieves information from one or more sources listed in the manifest.

Source selection — IMPORTANT: every source has a kind field. \
kind=record means the source holds actual entity data and values (e.g. a \
table of project budgets or employee records). kind=policy means the source \
holds rules, limits, entitlements, or procedures (e.g. a PDF defining budget \
limit rules or expense policies). \
When the question asks about a rule, limit, threshold, or procedure, you MUST \
route to a kind=policy source. When it asks for a specific entity's actual \
data or values, route to a kind=record source. \
Use the contains field to pick the most specific match within the chosen kind. \
Do not confuse a record source that stores data about a topic with a policy \
source that defines rules about the same topic.

Routing: assign worker_type by source type in the manifest. PDF sources use \
"librarian". CSV or SQLite sources use "data_scientist".

Multi-source tasks: when a task needs data from multiple sources and those \
sources share the same worker_type, combine them in a single task — \
source_ids: ["table1", "table2", "table3"]. The worker can query all of \
them together. This applies to any number of same-type sources: tables that \
need to be combined, compared, or cross-referenced, or PDFs that need to be \
searched together. Only split into separate tasks when sources use different \
worker types (e.g. one PDF and one CSV — these cannot share a runtime) or \
when the information needed is truly independent.

Cross-source relationships: when the manifest lists a CROSS-SOURCE \
RELATIONSHIPS block, use it to identify which tables share join keys. \
Prefer combining those tables into a single multi-source task.

Domain context: when the manifest includes a DOMAIN CONTEXT block, treat \
those rules as ground truth for ordering, ranking, or interpreting \
categorical values.

Dependencies: if a task requires a value that must first be retrieved from \
another source, create that retrieval as a separate task and set depends_on. \
If a task can run independently, set depends_on to null. Apply this per entity.

Use the minimum number of tasks. Only use sources from the manifest. \
Keep justifications to one sentence.

Respond with ONLY JSON — no explanation, no markdown:
{
  "reasoning": {
    "information_needed": ["what facts or rules are needed"],
    "source_assignments": [
      {"info": "...", "kind_required": "record|policy", "source_ids": ["..."], "worker_type": "...", "justification": "..."}
    ],
    "can_combine": "Before splitting: for source_assignments that share the same worker_type, can they be combined into a single task? If yes, merge them.",
    "dependencies": ["e.g. t2 needs t1 because ..."]
  },
  "tasks": [
    {"task_id": "t1", "worker_type": "librarian"|"data_scientist",
     "description": "...", "source_ids": ["..."], "depends_on": null|"t1"}
  ]
}
"""

_RETRY_NOTE_SECTION = """\

RETRY CONTEXT — a previous answer was rejected.

Your previous reasoning:
{previous_reasoning}

Auditor feedback:
{retry_notes}

Decide whether the plan needs to change. If the plan was correct, produce the same plan — the synthesis layer will correct itself.
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
    On retry (retry_count > 0 AND non-empty retry_notes): also retry_notes + previous_reasoning.
    """
    view = {
        "original_query":   state.get("rewritten_query") or state["original_query"],
        "manifest_context": state["manifest_context"],
    }
    if state.get("retry_count", 0) > 0 and state.get("retry_notes", ""):
        view["retry_notes"]        = state["retry_notes"]
        view["previous_reasoning"] = state.get("planner_reasoning", "")
    return view


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def planner_node(state: AgentState) -> dict:
    view = planner_view(state)

    # Pre-filter manifest using rewritten_query (or original_query fallback)
    query = view["original_query"]  # planner_view already resolves rewritten
    filtered_manifest, prefilter_trace = prefilter_manifest(query)
    _prefilter_traces[state.get("session_id", "")] = prefilter_trace

    retry_section = ""
    if "retry_notes" in view:
        retry_section = _RETRY_NOTE_SECTION.format(
            previous_reasoning=view.get("previous_reasoning", "") or "(no previous reasoning available)",
            retry_notes=view["retry_notes"],
        )

    user_message = _USER_TEMPLATE.format(
        manifest_context=filtered_manifest,
        original_query=view["original_query"],
        retry_section=retry_section,
    )

    llm = get_llm("planner").bind(max_tokens=1500)
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
                source_ids=t["source_ids"] if isinstance(t["source_ids"], list) else [t["source_ids"]],
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

    logger.debug(
        "[%s] Planner — reasoning=%s tasks=%s",
        state.get("session_id", "?"),
        reasoning_str or "(none)",
        json.dumps(
            [{"task_id": t["task_id"], "worker_type": t["worker_type"],
              "source_ids": t["source_ids"]} for t in tasks],
        ),
    )

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
                    "source_ids": ["employees"],
                    "worker_type": "data_scientist",
                    "justification": "employees table contains clearance levels for each employee"
                },
                {
                    "info": "Flight class entitlements by clearance level",
                    "source_ids": ["travel_policy_2024"],
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
                "source_ids":  ["employees"],
                "depends_on":  None,
            },
            {
                "task_id":     "t2",
                "worker_type": "librarian",
                "description": "Find flight class entitlements for clearance level A",
                "source_ids":  ["travel_policy_2024"],
                "depends_on":  "t1",
            },
        ]
    })

    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind.return_value = mock_llm  # .bind(max_tokens=...) returns self

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

    # ── Test 2: view includes retry_notes and previous_reasoning on retry ──
    retry_view = planner_view(PLANNER_RETRY_STATE)
    assert "retry_notes" in retry_view, "retry_notes must be present when retry_count > 0"
    assert "previous_reasoning" in retry_view, "previous_reasoning must be present on retry"
    assert retry_view["previous_reasoning"] == PLANNER_RETRY_STATE["planner_reasoning"]
    print("PASS: planner_view includes retry_notes + previous_reasoning on retry")

    # ── Test 2b: retry prompt renders RETRY CONTEXT with previous reasoning and audit notes ─
    captured_retry_msgs: list = []

    async def capture_retry_msgs(messages):
        captured_retry_msgs.extend(messages)
        return mock_response

    mock_llm.ainvoke = capture_retry_msgs

    with patch(f"{__name__}.get_llm", return_value=mock_llm):
        asyncio.run(planner_node(PLANNER_RETRY_STATE))

    retry_user_prompt = next(m["content"] for m in captured_retry_msgs if m["role"] == "user")
    assert "RETRY CONTEXT" in retry_user_prompt
    assert "Your previous reasoning" in retry_user_prompt
    assert "information_needed" in retry_user_prompt, \
        "Previous reasoning content must be rendered in the retry prompt"
    assert "did not cite which section" in retry_user_prompt, \
        "Auditor feedback must be rendered in the retry prompt"
    print("PASS: planner retry prompt renders previous reasoning and audit notes")

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
    assert result["plan"][0]["source_ids"] == ["employees"]
    assert result["plan"][0]["depends_on"] is None
    assert result["plan"][1]["task_id"] == "t2"
    assert result["plan"][1]["worker_type"] == "librarian"
    assert result["plan"][1]["source_ids"] == ["travel_policy_2024"]
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

    # ── Test 5: cross-table query with same worker_type → single multi-source task ──
    cross_table_output = json.dumps({
        "reasoning": {
            "information_needed": ["employee with the highest salary by joining employees and salary_bands"],
            "source_assignments": [
                {"info": "employee name and highest salary", "source_ids": ["salary_bands", "employees"],
                 "worker_type": "data_scientist",
                 "justification": "both tables share clearance_level/department keys and can be JOINed"}
            ],
            "can_combine": "salary_bands and employees are both data_scientist sources — combining into one task for a JOIN.",
            "dependencies": []
        },
        "tasks": [
            {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "description": "JOIN employees and salary_bands to find the employee with the highest salary_max",
                "source_ids":  ["salary_bands", "employees"],
                "depends_on":  None,
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
    assert len(plan) == 1, \
        f"Same-type cross-table query must produce 1 multi-source task, got {len(plan)}"
    assert len(plan[0]["source_ids"]) == 2, \
        f"Multi-source task must have 2 source_ids, got {plan[0]['source_ids']}"
    assert plan[0]["depends_on"] is None, \
        f"Multi-source task must have no dependency, got depends_on={plan[0]['depends_on']!r}"
    assert plan[0]["worker_type"] == "data_scientist", \
        f"Multi-source table task must use data_scientist, got {plan[0]['worker_type']}"
    print(f"PASS: cross-table query produces single multi-source task with source_ids={plan[0]['source_ids']}")

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
                    "source_ids": ["employees"],
                    "worker_type": "data_scientist",
                    "justification": "employees table contains department information for each employee"
                },
                {
                    "info": "Password rotation policy",
                    "source_ids": ["it_security_policy"],
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
                "source_ids": ["employees"],
                "depends_on": None,
            },
            {
                "task_id": "t2",
                "worker_type": "librarian",
                "description": "Find the password rotation policy",
                "source_ids": ["it_security_policy"],
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
    table_task = next(t for t in plan_m if "employees" in t["source_ids"])
    assert table_task["worker_type"] == "data_scientist", \
        f"Table source must use data_scientist, got {table_task['worker_type']}"
    # PDF task must use librarian
    pdf_task = next(t for t in plan_m if "it_security_policy" in t["source_ids"])
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

    # ── Test 7: entity requiring unknown property → prerequisite task ─
    # "What is Dan Cohen's flight entitlement?" requires looking up Dan's
    # property first, then using it to find the entitlement.
    prereq_output = json.dumps({
        "reasoning": {
            "information_needed": [
                "Dan Cohen's property from a structured source",
                "Entitlement rule from a policy document using that property"
            ],
            "source_assignments": [
                {"info": "Dan's property", "source_ids": ["employees"],
                 "worker_type": "data_scientist",
                 "justification": "employees table contains entity properties"},
                {"info": "Entitlement rule", "source_ids": ["travel_policy_2024"],
                 "worker_type": "librarian",
                 "justification": "travel policy contains entitlement rules"}
            ],
            "dependencies": ["t2 needs t1 output because the entitlement depends on a property not yet known"]
        },
        "tasks": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "Look up Dan Cohen's property from employees",
             "source_ids": ["employees"], "depends_on": None},
            {"task_id": "t2", "worker_type": "librarian",
             "description": "Find entitlement rule using the property from t1",
             "source_ids": ["travel_policy_2024"], "depends_on": "t1"},
        ]
    })
    mock_response_prereq = MagicMock()
    mock_response_prereq.content = prereq_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_prereq)

    prereq_state = {
        **PLANNER_STATE,
        "original_query": "What is Dan Cohen's flight entitlement?",
    }

    with patch(patch_target, return_value=mock_llm):
        result_prereq = asyncio.run(planner_node(prereq_state))

    plan_prereq = result_prereq["plan"]
    assert len(plan_prereq) >= 2, \
        f"Query requiring unknown property must produce at least 2 tasks, got {len(plan_prereq)}"
    # First task is the property lookup (no dependency)
    assert plan_prereq[0]["depends_on"] is None, \
        "Prerequisite lookup task must have no dependency"
    # Second task depends on first
    assert plan_prereq[1]["depends_on"] == "t1", \
        f"Policy task must depend on prerequisite, got depends_on={plan_prereq[1]['depends_on']!r}"
    print("PASS: entity requiring unknown property produces prerequisite task with correct depends_on")

    # ── Test 8: direct query needing no property lookup → single task ─
    # "What is the password rotation policy?" can be answered directly from
    # a document — no entity property lookup needed.
    direct_output = json.dumps({
        "reasoning": {
            "information_needed": ["Password rotation policy from a document"],
            "source_assignments": [
                {"info": "Password rotation policy", "source_ids": ["it_security_policy"],
                 "worker_type": "librarian",
                 "justification": "IT security policy contains password rules"}
            ],
            "dependencies": []
        },
        "tasks": [
            {"task_id": "t1", "worker_type": "librarian",
             "description": "Find the password rotation policy",
             "source_ids": ["it_security_policy"], "depends_on": None},
        ]
    })
    mock_response_direct = MagicMock()
    mock_response_direct.content = direct_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_direct)

    direct_state = {
        **PLANNER_STATE,
        "original_query": "What is the password rotation policy?",
        "manifest_context": """
pdfs:
  - id: it_security_policy
    summary: IT Security Policy covering password requirements, data classification, and incidents.
""",
    }

    with patch(patch_target, return_value=mock_llm):
        result_direct = asyncio.run(planner_node(direct_state))

    plan_direct = result_direct["plan"]
    assert len(plan_direct) == 1, \
        f"Direct query must produce exactly 1 task, got {len(plan_direct)}"
    assert plan_direct[0]["depends_on"] is None, \
        "Direct task must have no dependency"
    print("PASS: direct query needing no property lookup produces single task with no dependency")

    # ── Test 9: multi-source single task round-trip ──────────────
    # Validates that planner_node parsing correctly handles a task
    # with multiple source_ids (the isinstance check on line 128).
    multi_src_output = json.dumps({
        "reasoning": {
            "information_needed": ["employees matched to their salary bands"],
            "source_assignments": [
                {"info": "employees and their salary bands", "source_ids": ["employees", "salary_bands"],
                 "worker_type": "data_scientist",
                 "justification": "both are tables sharing clearance_level key — JOINable in one query"}
            ],
            "can_combine": "employees and salary_bands are both data_scientist sources — merging into one task.",
            "dependencies": []
        },
        "tasks": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "JOIN employees and salary_bands to find employees with the highest salary in their band",
             "source_ids": ["employees", "salary_bands"], "depends_on": None},
        ]
    })
    mock_response_multi = MagicMock()
    mock_response_multi.content = multi_src_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_multi)

    multi_src_state = {
        **PLANNER_STATE,
        "original_query": "Which employees have the highest salary in their band?",
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
        result_multi = asyncio.run(planner_node(multi_src_state))

    plan_multi = result_multi["plan"]
    assert len(plan_multi) == 1, \
        f"Multi-source query must produce exactly 1 task, got {len(plan_multi)}"
    assert len(plan_multi[0]["source_ids"]) == 2, \
        f"Task must have 2 source_ids, got {plan_multi[0]['source_ids']}"
    assert set(plan_multi[0]["source_ids"]) == {"employees", "salary_bands"}, \
        f"Task must reference both sources, got {plan_multi[0]['source_ids']}"
    assert plan_multi[0]["depends_on"] is None, \
        f"Multi-source task must have no dependency, got {plan_multi[0]['depends_on']!r}"
    assert plan_multi[0]["worker_type"] == "data_scientist", \
        f"Table task must use data_scientist, got {plan_multi[0]['worker_type']}"
    print("PASS: multi-source single task round-trip — parsing handles source_ids array correctly")

    print("\nPASS: all planner tests passed")


if __name__ == "__main__":
    test_planner()

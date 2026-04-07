# tests/fixtures.py
# ─────────────────────────────────────────────────────────────
# Shared fake state objects for isolated agent tests.
# Import these in any test file — never rebuild fake states inline.
# ─────────────────────────────────────────────────────────────

from typing import Any

# ── Minimal state for Chat Agent tests ───────────────────────
CHAT_AGENT_STATE: dict[str, Any] = {
    "original_query": "Can Noa fly Business Class?",
    "session_id": "test-session-001",
    "conversation_history": [
        {"role": "user", "content": "Can Noa fly Business Class?"}
    ],
    "final_answer": "Yes. Noa holds clearance level A, which entitles her to Business Class on flights over 4 hours.",
    "final_sources": [
        {"source_id": "travel_policy_2024", "source_type": "pdf", "label": "Travel Policy 2024"},
        {"source_id": "employees",           "source_type": "csv", "label": "Employees"},
    ],
    # Fields this agent should NOT see — present in state but must not reach prompt
    "plan": [],
    "task_results": {},
    "draft_answer": "",
    "audit_result": None,
    "retry_count": 0,
    "retry_notes": "",
}

# ── Minimal state for Planner tests ──────────────────────────
PLANNER_STATE: dict[str, Any] = {
    "original_query": "Can Noa fly Business Class?",
    "session_id": "test-session-001",
    "manifest_context": """
pdfs:
  - id: travel_policy_2024
    summary: Flight class entitlements, hotel limits, and expense caps by clearance level.
tables:
  - id: employees
    summary: Master employee list with names, departments, and clearance levels.
""",
    "retry_notes": "",
    "retry_count": 0,
    # All other fields empty — Planner should not need them
    "conversation_history": [],
    "plan": [],
    "task_results": {},
    "sources_used": [],
    "retrieved_chunks": [],
    "draft_answer": "",
    "audit_result": None,
    "final_answer": "",
    "final_sources": [],
}

# ── Minimal state for Planner RETRY test ─────────────────────
PLANNER_RETRY_STATE: dict[str, Any] = {
    **PLANNER_STATE,
    "retry_count": 1,
    "retry_notes": "The draft answer claimed Noa can fly Business Class but did not cite which section of the travel policy supports this. Re-run and ensure the specific entitlement rule is quoted.",
}

# ── Minimal state for Auditor tests ──────────────────────────
AUDITOR_STATE_PASS: dict[str, Any] = {
    "original_query": "Can Noa fly Business Class?",
    "plan": [
        {"task_id": "t1", "worker_type": "data_scientist", "description": "Get Noa's clearance level", "source_id": "employees"},
        {"task_id": "t2", "worker_type": "librarian", "description": "Find flight rules for clearance level A", "source_id": "travel_policy_2024"},
    ],
    "draft_answer": "Noa holds clearance level A. According to Section 1 of the Travel Policy 2024, employees with clearance level A are entitled to Business Class on flights over 4 hours.",
    "sources_used": [
        {"source_id": "employees",            "source_type": "csv", "label": "Employees"},
        {"source_id": "travel_policy_2024",   "source_type": "pdf", "label": "Travel Policy 2024"},
    ],
    # Fields Auditor should NOT see
    "conversation_history": [],
    "task_results": {},
    "retry_count": 0,
    "retry_notes": "",
    "final_answer": "",
    "final_sources": [],
}

AUDITOR_STATE_FAIL: dict[str, Any] = {
    **AUDITOR_STATE_PASS,
    "draft_answer": "Noa can fly Business Class.",  # No source cited — should fail audit
}

# ── Minimal state for Synthesizer tests ──────────────────────
SYNTHESIZER_STATE: dict[str, Any] = {
    "original_query": "Can Noa fly Business Class?",
    "plan": [
        {"task_id": "t1", "worker_type": "data_scientist", "description": "Get Noa's clearance level", "source_id": "employees"},
        {"task_id": "t2", "worker_type": "librarian", "description": "Find flight rules for clearance level A", "source_id": "travel_policy_2024"},
    ],
    "task_results": {
        "t1": {
            "task_id": "t1",
            "result_value": "clearance_level = A",
            "query_used": "SELECT clearance_level FROM employees WHERE full_name = 'Noa'",
            "table_name": "employees",
        },
        "t2": {
            "task_id": "t2",
            "chunks": [
                {"chunk_text": "Employees with clearance level A are entitled to Business Class on flights exceeding 4 hours.", "page_number": 3}
            ],
            "source_pdf": "travel_policy_2024",
        },
    },
    "sources_used": [
        {"source_id": "employees",           "source_type": "csv", "label": "Employees"},
        {"source_id": "travel_policy_2024",  "source_type": "pdf", "label": "Travel Policy 2024"},
    ],
    # Empty — Synthesizer should not need these
    "conversation_history": [],
    "draft_answer": "",
    "audit_result": None,
    "retry_count": 0,
    "retry_notes": "",
    "final_answer": "",
    "final_sources": [],
}

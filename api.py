"""
api.py — FastAPI application.

Endpoints:
  POST /chat   — Run the full agent graph for a user query.
  GET  /health — Liveness check.

Startup:
  If langsmith.enabled is true in config.yaml, sets LANGSMITH_TRACING=true
  so LangSmith picks up traces without requiring it to be pre-set in the shell.
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from core.llm_config import _load_config
from core.manifest import get_manifest_index
from core.state import AgentState, SourceRef
from graph import compiled_graph


# ---------------------------------------------------------------------------
# Startup — LangSmith tracing
# ---------------------------------------------------------------------------

_cfg = _load_config()
if _cfg.get("langsmith", {}).get("enabled", False):
    os.environ["LANGSMITH_TRACING"] = "true"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query:      str
    session_id: str


class ChatResponse(BaseModel):
    final_answer:  str
    final_sources: List[SourceRef]
    session_id:    str


class ErrorResponse(BaseModel):
    error:      str
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(req: ChatRequest):
    initial_state: AgentState = {
        "original_query":       req.query,
        "session_id":           req.session_id,
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

    try:
        final_state = await compiled_graph.ainvoke(
            initial_state,
            config={"recursion_limit": 25},
        )
    except Exception as exc:  # noqa: BLE001
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "session_id": req.session_id},
        )

    return ChatResponse(
        final_answer=final_state["final_answer"],
        final_sources=final_state["final_sources"],
        session_id=req.session_id,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m api  (requires pytest + httpx)
# ---------------------------------------------------------------------------

def test_api():
    import asyncio
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    client = TestClient(app)

    # ── Test 1: GET /health returns 200 with correct body ─────────
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "1.0"}
    print("PASS: GET /health returns 200 {status: ok, version: 1.0}")

    # ── Test 2: POST /chat with mocked graph — correct shape ──────
    fake_final_state = {
        "final_answer":  "Noa Levi has clearance level A.",
        "final_sources": [{"source_id": "employees", "source_type": "csv", "label": "Employees CSV"}],
        "session_id":    "test-session-001",
        # (other fields omitted — API only reads final_answer and final_sources)
    }

    patch_target = f"{__name__}.compiled_graph"

    with patch(patch_target) as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=fake_final_state)
        resp = client.post(
            "/chat",
            json={"query": "What is Noa Levi's clearance level?", "session_id": "test-session-001"},
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["final_answer"]  == "Noa Levi has clearance level A."
    assert body["session_id"]    == "test-session-001"
    assert isinstance(body["final_sources"], list)
    assert len(body["final_sources"]) == 1
    assert body["final_sources"][0]["source_id"] == "employees"
    print(f"PASS: POST /chat returns correct shape: {body['final_answer']}")

    # ── Test 3: POST /chat when graph raises → HTTP 500 ──────────
    with patch(patch_target) as mock_graph:
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("upstream service timeout"))
        resp = client.post(
            "/chat",
            json={"query": "crash this", "session_id": "err-session"},
        )

    assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "error"      in body
    assert "session_id" in body
    assert body["session_id"]               == "err-session"
    assert "upstream service timeout" in body["error"]
    print(f"PASS: POST /chat exception returns 500 with error: {body['error']}")

    print("\nPASS: all api tests passed")


if __name__ == "__main__":
    test_api()

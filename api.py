"""
api.py — FastAPI application.

Endpoints:
  POST   /chat                       — Run the full agent graph for a user query.
  GET    /health                     — Liveness check.
  GET    /conversation/{session_id}  — Full conversation log for a session.
  POST   /ingest/pdf                 — Upload, save, and ingest a PDF file.
  POST   /ingest/table               — Catalog and ingest a CSV or SQLite table.
  GET    /sources                    — List all indexed sources (index + full detail).
  DELETE /sources/{source_id}        — Remove a source from manifests and ChromaDB.

Startup:
  If langsmith.enabled is true in config.yaml, sets LANGSMITH_TRACING=true
  so LangSmith picks up traces without requiring it to be pre-set in the shell.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.llm_config import _load_config
from core.manifest import get_manifest_detail_raw, get_manifest_index, get_manifest_index_raw
from core.retriever import ChromaRetriever
from core.state import AgentState, Message, SourceRef
from graph import compiled_graph
from ingestion.manifest_writer import delete_source_from_manifest
from ingestion.pdf_ingestor import ingest_pdf
from ingestion.table_ingestor import ingest_table

_PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Enable DEBUG only for our own modules — keeps third-party noise (httpcore,
# openai, langsmith, urllib3, httpx …) at INFO/WARNING.
for _mod in ("agents", "core", "__main__"):
    logging.getLogger(_mod).setLevel(logging.DEBUG)


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

_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(_FRONTEND_DIR / "index.html")


# Per-session conversation history — keyed by session_id.
# Persists across requests within the same process lifetime.
_sessions: dict[str, list[Message]] = {}

# Per-session conversation log — keyed by session_id.
# Each value is a list of conversation_entry dicts (one per turn).
_conversation_log: dict[str, list] = {}

# Failure message from config — used to detect retry exhaustion vs real formatting.
_FAILURE_MESSAGE: str = _cfg.get("retry", {}).get("failure_message", "")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query:      str
    session_id: str


class TraceInfo(BaseModel):
    retry_count:             int
    audit_verdict:           Optional[str]            = None
    audit_notes:             Optional[str]            = None
    plan:                    Optional[List[dict]]     = None
    task_results:            dict
    synthesizer_output:      str
    chat_intent:             str
    rewritten_query:         str
    query_sent_to_planner:   Optional[str]            = None
    chat_formatted_response: bool
    step_timings:            Dict[str, float]
    retry_history:           List[dict]
    planner_reasoning:       Optional[str]            = None


class ChatResponse(BaseModel):
    final_answer:       str
    final_sources:      List[SourceRef]
    session_id:         str
    trace:              TraceInfo
    conversation_entry: Dict[str, Any]


class ErrorResponse(BaseModel):
    error:      str
    session_id: str = ""


class SourceListItem(BaseModel):
    source_id:    str
    name:         str
    type:         str            # "pdf" | "csv" | "sqlite"
    summary:      str
    # PDF-only
    page_count:   int | None = None
    tags:         List[str] | None = None
    # Table-only
    row_count:    int | None = None
    column_count: int | None = None




class IngestTableResponse(BaseModel):
    source_id:  str
    row_count:  int
    columns:    List[str]
    summary:    str
    table_name: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest/pdf", responses={400: {"model": ErrorResponse}})
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content=ErrorResponse(error="Only .pdf files are accepted").model_dump())

    source_id = Path(file.filename).stem
    dest = _PROJECT_ROOT / "data" / "pdfs" / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    try:
        result = await ingest_pdf(dest, source_id)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        return JSONResponse(status_code=400, content=ErrorResponse(error=str(exc)).model_dump())
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(exc)).model_dump())
    return result


@app.post("/ingest/table", response_model=IngestTableResponse, responses={400: {"model": ErrorResponse}})
async def ingest_table_endpoint(
    file: UploadFile = File(...),
    table_name: str | None = Form(None),
):
    if not file.filename.lower().endswith((".csv", ".sqlite")):
        return JSONResponse(status_code=400, content=ErrorResponse(error="Only .csv and .sqlite files are accepted").model_dump())

    source_id = Path(file.filename).stem
    dest = _PROJECT_ROOT / "data" / "tables" / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    try:
        result = await ingest_table(dest, source_id, table_name)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        return JSONResponse(status_code=400, content=ErrorResponse(error=str(exc)).model_dump())
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(exc)).model_dump())
    return result


@app.get("/sources", response_model=List[SourceListItem])
def get_sources():
    index_raw  = get_manifest_index_raw()
    detail_raw = get_manifest_detail_raw()

    pdf_detail   = {e["id"]: e for e in detail_raw.get("pdfs",   [])}
    table_detail = {e["id"]: e for e in detail_raw.get("tables", [])}

    sources: list[SourceListItem] = []

    for entry in index_raw.get("pdfs", []):
        detail = pdf_detail.get(entry["id"], {})
        sources.append(SourceListItem(
            source_id  = entry["id"],
            name       = entry["name"],
            type       = "pdf",
            summary    = entry["summary"],
            page_count = detail.get("pages"),
            tags       = detail.get("tags", []),
        ))

    for entry in index_raw.get("tables", []):
        detail = table_detail.get(entry["id"], {})
        sources.append(SourceListItem(
            source_id    = entry["id"],
            name         = entry["name"],
            type         = detail.get("type", "csv"),
            summary      = entry["summary"],
            row_count    = detail.get("row_count_approx"),
            column_count = len(detail.get("columns", [])),
        ))

    return sources


@app.delete("/sources/{source_id}")
def delete_source(source_id: str):
    try:
        delete_source_from_manifest(source_id)
    except ValueError as exc:
        return JSONResponse(status_code=404, content=ErrorResponse(error=str(exc)).model_dump())

    # ChromaDB collection only exists for PDF sources; silently skip if absent
    chromadb_removed = False
    try:
        ChromaRetriever()._client.delete_collection(source_id)
        chromadb_removed = True
    except Exception:
        chromadb_removed = False

    return {"deleted": source_id, "chromadb_collection_removed": chromadb_removed}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.get("/conversation/{session_id}")
def get_conversation(session_id: str):
    return _conversation_log.get(session_id, [])


@app.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(req: ChatRequest):
    history: list[Message] = list(_sessions.get(req.session_id, []))

    initial_state: AgentState = {
        "original_query":       req.query,
        "session_id":           req.session_id,
        "conversation_history": history,
        "chat_intent":          "",
        "rewritten_query":      "",
        "plan":                 [],
        "manifest_context":     get_manifest_index(),
        "planner_reasoning":    "",
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "synthesizer_output":   "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "retry_history":        [],
        "final_answer":         "",
        "final_sources":        [],
    }

    # Stream graph execution to capture per-node timings
    step_timings: dict[str, float] = {}
    accumulated: dict = dict(initial_state)
    chat_node_count = 0

    _NODE_KEY_MAP = {
        "planner_node":     "planning_ms",
        "router_node":      "routing_ms",
        "synthesizer_node": "synthesis_ms",
        "auditor_node":     "audit_ms",
    }

    try:
        t0 = time.perf_counter()
        async for update in compiled_graph.astream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 25},
        ):
            t1 = time.perf_counter()
            elapsed_ms = round((t1 - t0) * 1000, 1)
            for node_name, node_output in update.items():
                accumulated.update(node_output)
                if node_name == "chat_node":
                    chat_node_count += 1
                    key = "classification_ms" if chat_node_count == 1 else "formatting_ms"
                else:
                    key = _NODE_KEY_MAP.get(node_name, node_name)
                step_timings[key] = step_timings.get(key, 0) + elapsed_ms
            t0 = t1
        final_state = accumulated
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(exc), session_id=req.session_id).model_dump(),
        )

    _sessions[req.session_id] = list(final_state["conversation_history"])

    rewritten = final_state.get("rewritten_query", "")
    synthesizer_output = final_state.get("synthesizer_output", "")
    intent = final_state.get("chat_intent", "")
    ran_pipeline = intent not in ("DIRECT", "CLARIFY")

    if ran_pipeline:
        audit = final_state.get("audit_result") or {"verdict": "", "notes": ""}
        audit_verdict = audit.get("verdict", "")
        audit_notes = audit.get("notes", "")
        plan = [dict(t) for t in final_state.get("plan", [])]
        query_sent_to_planner = rewritten if rewritten else req.query
    else:
        audit_verdict = None
        audit_notes = None
        plan = None
        query_sent_to_planner = None

    chat_formatted = bool(synthesizer_output) and final_state["final_answer"] != _FAILURE_MESSAGE

    trace = TraceInfo(
        retry_count=final_state.get("retry_count", 0),
        audit_verdict=audit_verdict,
        audit_notes=audit_notes,
        plan=plan,
        task_results={k: dict(v) for k, v in final_state.get("task_results", {}).items()},
        synthesizer_output=synthesizer_output,
        chat_intent=intent,
        rewritten_query=rewritten,
        query_sent_to_planner=query_sent_to_planner,
        chat_formatted_response=chat_formatted,
        step_timings=step_timings,
        retry_history=final_state.get("retry_history", []),
        planner_reasoning=final_state.get("planner_reasoning", "") if ran_pipeline else None,
    )

    # Build and store conversation entry
    session_log = _conversation_log.setdefault(req.session_id, [])
    turn_number = len(session_log) + 1
    entry: Dict[str, Any] = {
        "turn_number":    turn_number,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "session_id":     req.session_id,
        "original_query": req.query,
        "rewritten_query": rewritten,
        "chat_intent":    final_state.get("chat_intent", ""),
        "final_answer":   final_state["final_answer"],
        "final_sources":  final_state["final_sources"],
        "trace":          trace.model_dump(),
    }
    session_log.append(entry)

    return ChatResponse(
        final_answer=final_state["final_answer"],
        final_sources=final_state["final_sources"],
        session_id=req.session_id,
        trace=trace,
        conversation_entry=entry,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m api  (requires pytest + httpx)
# ---------------------------------------------------------------------------

def test_api():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    client = TestClient(app)

    # ── Helper: mock astream from a final-state dict ─────────────
    def _as_astream(result_dict):
        """Convert a result dict to a mock astream async generator function."""
        async def _astream(state, stream_mode=None, config=None):
            yield {"chat_node": result_dict}
        return _astream

    def _as_failing_astream(exc):
        """Return a mock astream that raises the given exception."""
        async def _astream(state, stream_mode=None, config=None):
            raise exc
            yield  # noqa: unreachable — makes this function a generator
        return _astream

    # ── Test 1: GET /health returns 200 with correct body ─────────
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "1.0"}
    print("PASS: GET /health returns 200 {status: ok, version: 1.0}")

    # ── Test 2: POST /chat with mocked graph — correct shape ──────
    fake_final_state = {
        "final_answer":         "Noa Levi has clearance level A.",
        "final_sources":        [{"source_id": "employees", "source_type": "csv", "label": "Employees CSV"}],
        "session_id":           "test-session-001",
        "conversation_history": [],
        "chat_intent":          "PLAN",
        "rewritten_query":      "What is Noa Levi's clearance level?",
        "retry_count":          0,
        "audit_result":         {"verdict": "PASS", "notes": "All claims verified."},
        "plan":                 [{"task_id": "t1", "worker_type": "data_scientist",
                                  "description": "Look up clearance level", "source_ids": ["employees"]}],
        "task_results":         {"t1": {"task_id": "t1", "worker_type": "data_scientist",
                                        "output": "clearance_level: A", "success": True, "error": None}},
        "synthesizer_output":   "Noa Levi has clearance level A.",
    }

    patch_target = f"{__name__}.compiled_graph"
    _sessions.pop("test-session-001", None)
    _conversation_log.pop("test-session-001", None)

    with patch(patch_target) as mock_graph:
        mock_graph.astream = _as_astream(fake_final_state)
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
    assert "trace" in body, "Response must include trace field"
    trace = body["trace"]
    assert trace["retry_count"]   == 0
    assert trace["audit_verdict"] == "PASS"
    assert trace["audit_notes"]   == "All claims verified."
    assert isinstance(trace["plan"], list) and len(trace["plan"]) == 1
    assert trace["plan"][0]["task_id"]     == "t1"
    assert trace["plan"][0]["worker_type"] == "data_scientist"
    assert trace["plan"][0]["source_ids"]   == ["employees"]
    assert isinstance(trace["task_results"], dict), "trace.task_results must be a dict"
    assert "t1" in trace["task_results"], "task_results must contain t1"
    tr = trace["task_results"]["t1"]
    assert tr["task_id"]     == "t1"
    assert tr["worker_type"] == "data_scientist"
    assert tr["success"]     is True
    assert tr["output"]      == "clearance_level: A"
    assert tr["error"]       is None
    assert "synthesizer_output" in trace, "trace must include synthesizer_output"
    assert trace["synthesizer_output"] == "Noa Levi has clearance level A."
    assert trace["chat_intent"]     == "PLAN"
    assert trace["rewritten_query"] == "What is Noa Levi's clearance level?"
    # New trace fields
    assert trace["query_sent_to_planner"] == "What is Noa Levi's clearance level?", \
        "query_sent_to_planner must be rewritten_query when non-empty"
    assert trace["chat_formatted_response"] is True, \
        "chat_formatted_response must be True when synthesizer_output is non-empty"
    assert isinstance(trace["step_timings"], dict), "step_timings must be a dict"
    assert "classification_ms" in trace["step_timings"], \
        "step_timings must include classification_ms for chat_node"
    # conversation_entry
    assert "conversation_entry" in body, "Response must include conversation_entry"
    ce = body["conversation_entry"]
    assert ce["turn_number"]    == 1
    assert ce["session_id"]     == "test-session-001"
    assert ce["original_query"] == "What is Noa Levi's clearance level?"
    assert ce["rewritten_query"] == "What is Noa Levi's clearance level?"
    assert ce["chat_intent"]    == "PLAN"
    assert ce["final_answer"]   == "Noa Levi has clearance level A."
    assert isinstance(ce["final_sources"], list)
    assert isinstance(ce["trace"], dict)
    assert "timestamp" in ce
    print(f"PASS: POST /chat returns correct shape with enriched trace: {body['final_answer']}")

    # ── Test 3: POST /chat when graph raises → HTTP 500 ──────────
    with patch(patch_target) as mock_graph:
        mock_graph.astream = _as_failing_astream(RuntimeError("upstream service timeout"))
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

    # ── Test 4: POST /ingest/table — happy path (mocked, multipart) ─
    _test_csv_name = "__api_test_table__.csv"
    _test_csv_dest = _PROJECT_ROOT / "data" / "tables" / _test_csv_name
    fake_ingest_result = {
        "source_id":  "__api_test_table__",
        "row_count":  3,
        "columns":    ["employee_id", "full_name", "department"],
        "summary":    "Employee records.",
        "table_name": None,
    }
    ingest_patch = f"{__name__}.ingest_table"
    try:
        with patch(ingest_patch, new=AsyncMock(return_value=fake_ingest_result)):
            resp = client.post(
                "/ingest/table",
                files={"file": (_test_csv_name, b"id,name\n1,Alice", "text/csv")},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["source_id"]  == "__api_test_table__"
        assert body["row_count"]  == 3
        assert body["table_name"] is None
        print(f"PASS: POST /ingest/table returns correct shape: {body}")
    finally:
        if _test_csv_dest.exists():
            _test_csv_dest.unlink()

    # ── Test 5: POST /ingest/table — table_name form field passed through ──
    _test_sqlite_name = "__api_test_sqlite__.sqlite"
    _test_sqlite_dest = _PROJECT_ROOT / "data" / "tables" / _test_sqlite_name
    fake_multi_result = {
        "source_id":  "__api_test_sqlite__",
        "row_count":  2,
        "columns":    ["id", "label"],
        "summary":    "Alpha table.",
        "table_name": "alpha",
    }
    try:
        with patch(ingest_patch, new=AsyncMock(return_value=fake_multi_result)) as mock_ingest:
            resp = client.post(
                "/ingest/table",
                files={"file": (_test_sqlite_name, b"SQLite format 3\x00", "application/octet-stream")},
                data={"table_name": "alpha"},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["table_name"] == "alpha"
        called_kwargs = mock_ingest.call_args
        assert called_kwargs.args[2] == "alpha" or called_kwargs.kwargs.get("table_name") == "alpha", (
            f"table_name must be passed through to ingest_table, call_args={called_kwargs}"
        )
        print(f"PASS: POST /ingest/table passes table_name form field through correctly")
    finally:
        if _test_sqlite_dest.exists():
            _test_sqlite_dest.unlink()

    # ── Test 6: POST /ingest/table — ValueError → HTTP 400, file deleted ──
    _test_err_name = "__api_test_table_err__.csv"
    _test_err_dest = _PROJECT_ROOT / "data" / "tables" / _test_err_name
    try:
        with patch(ingest_patch, new=AsyncMock(side_effect=ValueError("multi-table error"))):
            resp = client.post(
                "/ingest/table",
                files={"file": (_test_err_name, b"id,name\n1,Bad", "text/csv")},
            )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "multi-table error" in body["error"]
        assert "session_id" in body, "Error response must always include session_id"
        assert body["session_id"] == ""
        assert not _test_err_dest.exists(), "File must be deleted after ingest failure"
        print(f"PASS: POST /ingest/table ValueError returns 400 with consistent error shape and deletes uploaded file")
    finally:
        if _test_err_dest.exists():
            _test_err_dest.unlink()

    # ── Test 7: GET /sources — returns clean source list ─────────
    fake_index_raw  = {
        "pdfs":   [{"id": "travel_policy_2024", "name": "Travel Policy 2024", "summary": "Travel rules."}],
        "tables": [{"id": "employees",          "name": "Employees",          "summary": "Employee list."}],
    }
    fake_detail_raw = {
        "pdfs":   [{"id": "travel_policy_2024", "type": "pdf", "pages": 5,
                    "tags": ["travel", "policy"]}],
        "tables": [{"id": "employees", "type": "csv", "row_count_approx": 10,
                    "columns": [{"name": "id"}, {"name": "full_name"}]}],
    }

    with patch(f"{__name__}.get_manifest_index_raw",  return_value=fake_index_raw), \
         patch(f"{__name__}.get_manifest_detail_raw", return_value=fake_detail_raw):
        resp = client.get("/sources")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert isinstance(body, list) and len(body) == 2, \
        f"Response must be a list of 2 source objects, got: {body}"

    pdf_src = next(s for s in body if s["source_id"] == "travel_policy_2024")
    assert pdf_src["name"]       == "Travel Policy 2024"
    assert pdf_src["type"]       == "pdf"
    assert pdf_src["summary"]    == "Travel rules."
    assert pdf_src["page_count"] == 5
    assert pdf_src["tags"]       == ["travel", "policy"]
    assert pdf_src["row_count"]    is None
    assert pdf_src["column_count"] is None

    tbl_src = next(s for s in body if s["source_id"] == "employees")
    assert tbl_src["name"]         == "Employees"
    assert tbl_src["type"]         == "csv"
    assert tbl_src["summary"]      == "Employee list."
    assert tbl_src["row_count"]    == 10
    assert tbl_src["column_count"] == 2
    assert tbl_src["page_count"] is None
    assert tbl_src["tags"]       is None
    print(f"PASS: GET /sources returns clean list of {len(body)} source objects")

    # ── Test 8: DELETE /sources/{source_id} — valid id ───────────
    mock_chroma_client = MagicMock()
    mock_chroma_client.delete_collection = MagicMock()

    with patch(f"{__name__}.delete_source_from_manifest") as mock_del, \
         patch(f"{__name__}.ChromaRetriever") as mock_cr:
        mock_cr.return_value._client = mock_chroma_client
        resp = client.delete("/sources/travel_policy_2024")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["deleted"]                    == "travel_policy_2024"
    assert body["chromadb_collection_removed"] is True
    mock_del.assert_called_once_with("travel_policy_2024")
    mock_chroma_client.delete_collection.assert_called_once_with("travel_policy_2024")
    print(f"PASS: DELETE /sources/travel_policy_2024 returns correct shape: {body}")

    # ── Test 9: DELETE /sources/{source_id} — unknown id → 404 ───
    with patch(f"{__name__}.delete_source_from_manifest",
               side_effect=ValueError("source_id 'ghost' not found")):
        resp = client.delete("/sources/ghost")

    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "ghost" in body["error"]
    assert "session_id" in body, "Error response must always include session_id"
    assert body["session_id"] == ""
    print(f"PASS: DELETE /sources/ghost returns 404 with consistent error shape")

    # ── Test 10: DELETE /sources/{source_id} — ChromaDB absent ───
    mock_chroma_client_no_coll = MagicMock()
    mock_chroma_client_no_coll.delete_collection = MagicMock(
        side_effect=Exception("Collection not found")
    )

    with patch(f"{__name__}.delete_source_from_manifest"), \
         patch(f"{__name__}.ChromaRetriever") as mock_cr2:
        mock_cr2.return_value._client = mock_chroma_client_no_coll
        resp = client.delete("/sources/employees")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["deleted"]                    == "employees"
    assert body["chromadb_collection_removed"] is False
    print(f"PASS: DELETE /sources/employees with no ChromaDB collection returns chromadb_collection_removed=False")

    # ── Test 11: POST /ingest/pdf — happy path (mocked) ──────────
    _test_pdf_name = "__api_test_upload__.pdf"
    _test_pdf_dest = _PROJECT_ROOT / "data" / "pdfs" / _test_pdf_name
    fake_pdf_result = {
        "source_id":       "__api_test_upload__",
        "chunks_ingested": 6,
        "summary":         "Travel policy covering flights and hotels.",
        "tags":            ["travel", "policy"],
    }
    pdf_patch = f"{__name__}.ingest_pdf"
    try:
        with patch(pdf_patch, new=AsyncMock(return_value=fake_pdf_result)):
            resp = client.post(
                "/ingest/pdf",
                files={"file": (_test_pdf_name, b"%PDF-1.4 fake", "application/pdf")},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["source_id"]       == "__api_test_upload__"
        assert body["chunks_ingested"] == 6
        assert body["summary"]
        assert isinstance(body["tags"], list)
        print(f"PASS: POST /ingest/pdf returns correct shape: {body['source_id']}, {body['chunks_ingested']} chunks")
    finally:
        if _test_pdf_dest.exists():
            _test_pdf_dest.unlink()

    # ── Test 12: POST /ingest/pdf — ValueError → HTTP 400, file deleted ─
    _test_bad_name = "__api_test_bad__.pdf"
    _test_bad_dest = _PROJECT_ROOT / "data" / "pdfs" / _test_bad_name
    try:
        with patch(pdf_patch, new=AsyncMock(side_effect=ValueError("PDF parse failed"))):
            resp = client.post(
                "/ingest/pdf",
                files={"file": (_test_bad_name, b"not a pdf", "application/pdf")},
            )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "PDF parse failed" in body["error"]
        assert "session_id" in body, "Error response must always include session_id"
        assert body["session_id"] == ""
        assert not _test_bad_dest.exists(), "File must be deleted after ingest failure"
        print(f"PASS: POST /ingest/pdf ValueError returns 400 with consistent error shape and deletes uploaded file")
    finally:
        if _test_bad_dest.exists():
            _test_bad_dest.unlink()

    # ── Test 13: POST /ingest/pdf — non-.pdf extension → HTTP 400 ────
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("report.docx", b"not a pdf", "application/octet-stream")},
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["error"] == "Only .pdf files are accepted"
    assert "session_id" in body, "Error response must always include session_id"
    assert body["session_id"] == ""
    assert not (_PROJECT_ROOT / "data" / "pdfs" / "report.docx").exists(), \
        "Non-pdf file must never be written to disk"
    print(f"PASS: POST /ingest/pdf rejects non-.pdf extension with 400 and consistent error shape")

    # ── Test 14: POST /ingest/pdf — unexpected exception → HTTP 500, file deleted ─
    _test_exc_name = "__api_test_exc__.pdf"
    _test_exc_dest = _PROJECT_ROOT / "data" / "pdfs" / _test_exc_name
    try:
        with patch(pdf_patch, new=AsyncMock(side_effect=RuntimeError("unexpected crash"))):
            resp = client.post(
                "/ingest/pdf",
                files={"file": (_test_exc_name, b"%PDF-1.4 fake", "application/pdf")},
            )
        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "unexpected crash" in body["error"]
        assert "session_id" in body, "Error response must always include session_id"
        assert body["session_id"] == ""
        assert not _test_exc_dest.exists(), "File must be deleted after unexpected ingest failure"
        print(f"PASS: POST /ingest/pdf unexpected exception returns 500 with consistent error shape and deletes uploaded file")
    finally:
        if _test_exc_dest.exists():
            _test_exc_dest.unlink()

    # ── Test 15: same session_id — history accumulates correctly across 3 turns ──
    patch_target = f"{__name__}.compiled_graph"

    def make_fake_state(answer: str, history_after: list) -> dict:
        return {
            "final_answer":         answer,
            "final_sources":        [],
            "session_id":           "shared-session",
            "conversation_history": history_after,
            "retry_count":          0,
            "audit_result":         {"verdict": "PASS", "notes": ""},
            "plan":                 [],
            "task_results":         {},
        }

    captured_states: list[AgentState] = []

    async def recording_astream(state, stream_mode=None, config=None):
        captured_states.append(state)
        turn_history = list(state["conversation_history"]) + [
            {"role": "user",      "content": state["original_query"]},
            {"role": "assistant", "content": "answer-" + str(len(captured_states))},
        ]
        yield {"chat_node": make_fake_state("answer-" + str(len(captured_states)), turn_history)}

    # Clear any leftover session state from earlier tests
    _sessions.pop("shared-session", None)
    _conversation_log.pop("shared-session", None)

    with patch(patch_target) as mock_graph:
        mock_graph.astream = recording_astream
        resp1 = client.post("/chat", json={"query": "first question",  "session_id": "shared-session"})
        resp2 = client.post("/chat", json={"query": "second question", "session_id": "shared-session"})
        resp3 = client.post("/chat", json={"query": "third question",  "session_id": "shared-session"})

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 200
    for idx, resp in enumerate((resp1, resp2, resp3), 1):
        rbody = resp.json()
        assert "trace" in rbody, "Response must include trace field"
        t = rbody["trace"]
        assert "retry_count"   in t
        assert "audit_verdict" in t
        assert "audit_notes"   in t
        assert isinstance(t["plan"], list)
        assert isinstance(t["step_timings"], dict), "trace must include step_timings dict"
        assert "query_sent_to_planner"   in t, "trace must include query_sent_to_planner"
        assert "chat_formatted_response" in t, "trace must include chat_formatted_response"
        assert "conversation_entry" in rbody, "Response must include conversation_entry"
        assert rbody["conversation_entry"]["turn_number"] == idx, \
            f"turn_number must be {idx}, got {rbody['conversation_entry']['turn_number']}"

    # First call must have started with empty history
    assert captured_states[0]["conversation_history"] == [], \
        "First call must start with empty history"

    # Second call must have received the two messages written after the first call
    second_history = captured_states[1]["conversation_history"]
    assert len(second_history) == 2, \
        f"Second call must receive 2-message history from first call, got: {second_history}"
    assert second_history[0] == {"role": "user",      "content": "first question"}
    assert second_history[1] == {"role": "assistant", "content": "answer-1"}

    # Third call must have accumulated all four messages from turns 1 and 2
    third_history = captured_states[2]["conversation_history"]
    assert len(third_history) == 4, \
        f"Third call must receive 4-message history (2 turns), got: {third_history}"
    assert third_history[0] == {"role": "user",      "content": "first question"}
    assert third_history[1] == {"role": "assistant", "content": "answer-1"}
    assert third_history[2] == {"role": "user",      "content": "second question"}
    assert third_history[3] == {"role": "assistant", "content": "answer-2"}
    print("PASS: same session_id — history accumulates correctly across 3 turns")

    # ── Test 16: different session_ids produce independent histories ───
    captured_states.clear()
    _sessions.pop("session-alpha", None)
    _sessions.pop("session-beta",  None)
    _conversation_log.pop("session-alpha", None)
    _conversation_log.pop("session-beta",  None)

    async def recording_astream_isolated(state, stream_mode=None, config=None):
        captured_states.append(state)
        sid   = state["session_id"]
        query = state["original_query"]
        updated_history = list(state["conversation_history"]) + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": f"reply-{sid}"},
        ]
        yield {"chat_node": {
            "final_answer":         f"reply-{sid}",
            "final_sources":        [],
            "session_id":           sid,
            "conversation_history": updated_history,
            "retry_count":          0,
            "audit_result":         {"verdict": "PASS", "notes": ""},
            "plan":                 [],
            "task_results":         {},
        }}

    with patch(patch_target) as mock_graph:
        mock_graph.astream = recording_astream_isolated
        # Alternate requests across two sessions
        client.post("/chat", json={"query": "alpha-q1", "session_id": "session-alpha"})
        client.post("/chat", json={"query": "beta-q1",  "session_id": "session-beta"})
        client.post("/chat", json={"query": "alpha-q2", "session_id": "session-alpha"})
        client.post("/chat", json={"query": "beta-q2",  "session_id": "session-beta"})

    # Third call (alpha-q2) must only see alpha's history, not beta's
    alpha_q2_state = captured_states[2]
    assert alpha_q2_state["session_id"] == "session-alpha"
    history_seen = alpha_q2_state["conversation_history"]
    assert len(history_seen) == 2, \
        f"session-alpha second call must see exactly 2 messages (its own first turn), got: {history_seen}"
    assert all(msg["content"] != "beta-q1" for msg in history_seen), \
        "session-alpha must never see session-beta messages"

    # Fourth call (beta-q2) must only see beta's history, not alpha's
    beta_q2_state = captured_states[3]
    assert beta_q2_state["session_id"] == "session-beta"
    beta_history = beta_q2_state["conversation_history"]
    assert len(beta_history) == 2, \
        f"session-beta second call must see exactly 2 messages (its own first turn), got: {beta_history}"
    assert beta_history[0] == {"role": "user",      "content": "beta-q1"}
    assert beta_history[1] == {"role": "assistant", "content": "reply-session-beta"}
    assert all(msg["content"] not in ("alpha-q1", "alpha-q2") for msg in beta_history), \
        "session-beta must never see session-alpha messages"

    # Trace field shape check
    _sessions.pop("session-trace", None)
    _conversation_log.pop("session-trace", None)
    with patch(patch_target) as mock_graph:
        mock_graph.astream = recording_astream_isolated
        trace_resp = client.post("/chat", json={"query": "trace-check", "session_id": "session-trace"})
    assert "trace" in trace_resp.json(), "Response must include trace field"
    t16 = trace_resp.json()["trace"]
    assert "retry_count"       in t16 and "audit_verdict" in t16 and "audit_notes" in t16
    assert "chat_intent"       in t16, "trace must include chat_intent"
    assert "rewritten_query"   in t16, "trace must include rewritten_query"
    assert "query_sent_to_planner"   in t16, "trace must include query_sent_to_planner"
    assert "chat_formatted_response" in t16, "trace must include chat_formatted_response"
    assert isinstance(t16["step_timings"], dict), "step_timings must be a dict"
    assert isinstance(t16["plan"], list)
    assert isinstance(t16["task_results"], dict)
    print("PASS: different session_ids produce fully independent conversation histories")

    # ── Test 17: DIRECT intent — pipeline-only fields are null ──────
    _sessions.pop("test-qsp", None)
    _conversation_log.pop("test-qsp", None)
    no_rewrite_state = {
        "final_answer":         "Hello!",
        "final_sources":        [],
        "conversation_history": [],
        "chat_intent":          "DIRECT",
        "rewritten_query":      "",
        "retry_count":          0,
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "plan":                 [],
        "task_results":         {},
        "synthesizer_output":   "",
    }
    with patch(patch_target) as mock_graph:
        mock_graph.astream = _as_astream(no_rewrite_state)
        resp = client.post("/chat", json={"query": "Hi there", "session_id": "test-qsp"})
    assert resp.status_code == 200
    body = resp.json()
    t17 = body["trace"]
    assert t17["chat_intent"] == "DIRECT"
    assert t17["audit_verdict"]        is None, "DIRECT must have null audit_verdict"
    assert t17["audit_notes"]          is None, "DIRECT must have null audit_notes"
    assert t17["plan"]                 is None, "DIRECT must have null plan"
    assert t17["query_sent_to_planner"] is None, "DIRECT must have null query_sent_to_planner"
    assert t17["chat_formatted_response"] is False
    assert isinstance(t17["step_timings"], dict)
    assert body["conversation_entry"]["chat_intent"] == "DIRECT"
    print("PASS: DIRECT intent sets pipeline-only trace fields to null")

    # ── Test 18: GET /conversation/{session_id} returns conversation log ──
    # Test 2 already created one entry for test-session-001
    resp = client.get("/conversation/test-session-001")
    assert resp.status_code == 200
    log = resp.json()
    assert isinstance(log, list), "Conversation log must be a list"
    assert len(log) >= 1, "Must have at least 1 entry from test 2"
    entry = log[0]
    assert entry["turn_number"]    == 1
    assert entry["session_id"]     == "test-session-001"
    assert entry["original_query"] == "What is Noa Levi's clearance level?"
    assert "timestamp" in entry
    assert isinstance(entry["trace"], dict)
    print(f"PASS: GET /conversation/test-session-001 returns {len(log)} entry(ies)")

    # ── Test 19: GET /conversation for unknown session returns empty list ──
    resp = client.get("/conversation/nonexistent-session")
    assert resp.status_code == 200
    assert resp.json() == [], "Unknown session must return empty list"
    print("PASS: GET /conversation for unknown session returns []")

    # ── Test 20: conversation_log accumulates turn_number across turns ──
    log = _conversation_log.get("shared-session", [])
    assert len(log) == 3, f"shared-session must have 3 entries from test 15, got {len(log)}"
    for i, entry in enumerate(log, 1):
        assert entry["turn_number"] == i, \
            f"Turn {i} must have turn_number={i}, got {entry['turn_number']}"
    resp = client.get("/conversation/shared-session")
    assert resp.status_code == 200
    assert len(resp.json()) == 3
    print("PASS: conversation_log accumulates correct turn_numbers across turns")

    print("\nPASS: all api tests passed")


if __name__ == "__main__":
    import uvicorn
    _api_cfg = _load_config().get("api", {})
    uvicorn.run(
        "api:app",
        host=_api_cfg.get("host", "0.0.0.0"),
        port=_api_cfg.get("port", 8000),
        reload=_api_cfg.get("reload", False),
    )

"""
api.py — FastAPI application.

Endpoints:
  POST   /chat                — Run the full agent graph for a user query.
  GET    /health              — Liveness check.
  POST   /ingest/pdf          — Upload, save, and ingest a PDF file.
  POST   /ingest/table        — Catalog and ingest a CSV or SQLite table.
  GET    /sources             — List all indexed sources (index + full detail).
  DELETE /sources/{source_id} — Remove a source from manifests and ChromaDB.

Startup:
  If langsmith.enabled is true in config.yaml, sets LANGSMITH_TRACING=true
  so LangSmith picks up traces without requiring it to be pre-set in the shell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Query, UploadFile
from pydantic import BaseModel

from core.llm_config import _load_config
from core.manifest import _load_detail_raw, _load_index_raw, get_manifest_index
from core.retriever import ChromaRetriever
from core.state import AgentState, SourceRef
from graph import compiled_graph
from ingestion.manifest_writer import delete_source_from_manifest
from ingestion.pdf_ingestor import ingest_pdf
from ingestion.table_ingestor import ingest_table

_PROJECT_ROOT = Path(__file__).resolve().parent


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


class IngestTableRequest(BaseModel):
    file_path: str
    source_id: str


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
    source_id = Path(file.filename).stem
    dest = _PROJECT_ROOT / "data" / "pdfs" / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    try:
        result = await ingest_pdf(dest, source_id)
    except ValueError as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": str(exc), "session_id": ""})
    return result


@app.post("/ingest/table", response_model=IngestTableResponse, responses={400: {"model": ErrorResponse}})
async def ingest_table_endpoint(
    req: IngestTableRequest,
    table_name: str | None = Query(default=None, description="SQLite only — required when the file contains multiple tables"),
):
    from pathlib import Path as _Path

    try:
        result = await ingest_table(_Path(req.file_path), req.source_id, table_name)
    except ValueError as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": str(exc), "session_id": ""})
    return result


@app.get("/sources")
def get_sources():
    index_raw  = _load_index_raw()
    detail_raw = _load_detail_raw()
    index_entries = index_raw.get("pdfs", []) + index_raw.get("tables", [])
    return {"index": index_entries, "detail": detail_raw}


@app.delete("/sources/{source_id}")
def delete_source(source_id: str):
    from fastapi.responses import JSONResponse

    try:
        delete_source_from_manifest(source_id)
    except ValueError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})

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
    from unittest.mock import AsyncMock, MagicMock, patch

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

    # ── Test 4: POST /ingest/table — happy path (mocked) ─────────
    fake_ingest_result = {
        "source_id":  "employees",
        "row_count":  3,
        "columns":    ["employee_id", "full_name", "department"],
        "summary":    "Employee records.",
        "table_name": None,
    }
    ingest_patch = f"{__name__}.ingest_table"
    with patch(ingest_patch, new=AsyncMock(return_value=fake_ingest_result)):
        resp = client.post(
            "/ingest/table",
            json={"file_path": "/data/tables/employees.csv", "source_id": "employees"},
        )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["source_id"]  == "employees"
    assert body["row_count"]  == 3
    assert body["table_name"] is None
    print(f"PASS: POST /ingest/table returns correct shape: {body}")

    # ── Test 5: POST /ingest/table — table_name query param passed through ──
    fake_multi_result = {
        "source_id":  "multi_alpha",
        "row_count":  2,
        "columns":    ["id", "label"],
        "summary":    "Alpha table.",
        "table_name": "alpha",
    }
    with patch(ingest_patch, new=AsyncMock(return_value=fake_multi_result)) as mock_ingest:
        resp = client.post(
            "/ingest/table?table_name=alpha",
            json={"file_path": "/data/tables/multi.sqlite", "source_id": "multi_alpha"},
        )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["table_name"] == "alpha"
    called_kwargs = mock_ingest.call_args
    assert called_kwargs.args[2] == "alpha" or called_kwargs.kwargs.get("table_name") == "alpha", (
        f"table_name must be passed through to ingest_table, call_args={called_kwargs}"
    )
    print(f"PASS: POST /ingest/table passes table_name query param through correctly")

    # ── Test 6: POST /ingest/table — ValueError → HTTP 400 ───────
    with patch(ingest_patch, new=AsyncMock(side_effect=ValueError("multi-table error"))):
        resp = client.post(
            "/ingest/table",
            json={"file_path": "/data/tables/multi.sqlite", "source_id": "bad"},
        )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    assert "multi-table error" in resp.json()["error"]
    print(f"PASS: POST /ingest/table ValueError returns 400")

    # ── Test 7: GET /sources — returns both keys with content ─────
    fake_index_raw  = {"pdfs": [{"id": "travel_policy_2024", "summary": "Travel rules."}],
                       "tables": [{"id": "employees", "summary": "Employee list."}]}
    fake_detail_raw = {"pdfs": [{"id": "travel_policy_2024", "type": "pdf"}],
                       "tables": [{"id": "employees", "type": "csv"}]}

    with patch(f"{__name__}._load_index_raw",  return_value=fake_index_raw), \
         patch(f"{__name__}._load_detail_raw", return_value=fake_detail_raw):
        resp = client.get("/sources")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "index"  in body and "detail" in body, "Response must contain 'index' and 'detail' keys"
    assert len(body["index"]) == 2, "index must be flat list combining pdfs + tables"
    assert {e["id"] for e in body["index"]} == {"travel_policy_2024", "employees"}
    assert "pdfs"   in body["detail"] and "tables" in body["detail"]
    print(f"PASS: GET /sources returns both keys with {len(body['index'])} entries")

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
    assert "ghost" in resp.json()["error"]
    print(f"PASS: DELETE /sources/ghost returns 404 for unknown source_id")

    # ── Test 10: DELETE /sources/{source_id} — ChromaDB absent ───
    # Table sources may have no ChromaDB collection; chromadb_collection_removed = False
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
    # Use a unique throwaway filename so the endpoint never touches real data files.
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

    # ── Test 12: POST /ingest/pdf — ValueError → HTTP 400 ────────
    _test_bad_name = "__api_test_bad__.pdf"
    _test_bad_dest = _PROJECT_ROOT / "data" / "pdfs" / _test_bad_name
    try:
        with patch(pdf_patch, new=AsyncMock(side_effect=ValueError("PDF parse failed"))):
            resp = client.post(
                "/ingest/pdf",
                files={"file": (_test_bad_name, b"not a pdf", "application/pdf")},
            )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
        assert "PDF parse failed" in resp.json()["error"]
        print(f"PASS: POST /ingest/pdf ValueError returns 400")
    finally:
        if _test_bad_dest.exists():
            _test_bad_dest.unlink()

    print("\nPASS: all api tests passed")


if __name__ == "__main__":
    test_api()

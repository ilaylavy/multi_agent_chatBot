"""
core/retriever.py — Abstract retriever interface and ChromaDB implementation.

The Librarian always codes against RetrieverInterface, never ChromaRetriever directly.
To add a new retrieval backend (e.g. GraphRAG), implement a new subclass of
RetrieverInterface and inject it into the Librarian — no agent code changes required.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import chromadb

from core.state import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class RetrieverInterface(ABC):
    """
    Abstract base class for all retrieval backends.

    Implementations:
      - ChromaRetriever  — local ChromaDB vector store (v1, active)
      - GraphRAGRetriever — graph-based retrieval (future, not yet implemented)

    Swap the backend by passing a different RetrieverInterface subclass to the
    Librarian at construction time. No Librarian code changes required.
    """

    @abstractmethod
    async def search(self, query: str, source_id: str, top_k: int) -> List[Chunk]:
        """
        Search for chunks relevant to query within the given source.

        Parameters
        ----------
        query     : Natural language query string.
        source_id : The manifest source_id identifying which collection to search.
        top_k     : Maximum number of chunks to return.

        Returns
        -------
        List[Chunk]
            Ranked list of at most top_k chunks, most relevant first.
        """


# ---------------------------------------------------------------------------
# ChromaDB implementation
# ---------------------------------------------------------------------------

class ChromaRetriever(RetrieverInterface):
    """
    Retrieval backend backed by a local ChromaDB vector store.

    Each source_id corresponds to a ChromaDB collection of the same name.
    Collections are created at ingest time by the ingestion script — this
    class only reads from them.
    """

    def __init__(self, chroma_path: str | None = None) -> None:
        """
        Parameters
        ----------
        chroma_path : Absolute or relative path to the ChromaDB persistence
                      directory. Defaults to data/chroma_db/ relative to the
                      project root (as configured in config.yaml).
        """
        if chroma_path is None:
            project_root = Path(__file__).resolve().parent.parent
            chroma_path = str(project_root / "data" / "chroma_db")

        self._client = chromadb.PersistentClient(path=chroma_path)

    async def search(self, query: str, source_id: str, top_k: int) -> List[Chunk]:
        """
        Query the ChromaDB collection for source_id and return top_k chunks.

        Uses ChromaDB's built-in embedding function (default: all-MiniLM-L6-v2).
        Returns an empty list if the collection does not exist yet (not yet ingested).
        """
        try:
            collection = self._client.get_collection(name=source_id)
        except Exception:
            # Collection not yet created — return empty rather than crashing
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count() or 1),
        )

        chunks: List[Chunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            chunks.append(
                Chunk(
                    chunk_text=doc,
                    source_pdf=meta.get("source_pdf", source_id),
                    page_number=int(meta.get("page_number", 0)),
                    # ChromaDB returns L2 distance; convert to a 0-1 relevance score
                    relevance_score=round(1 / (1 + distance), 4),
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.retriever
# ---------------------------------------------------------------------------

def test_retriever():
    import asyncio
    import inspect

    # 1. RetrieverInterface is abstract — cannot be instantiated directly
    try:
        RetrieverInterface()  # type: ignore[abstract]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    print("PASS: RetrieverInterface is correctly abstract")

    # 2. ChromaRetriever implements the interface
    retriever = ChromaRetriever()
    assert isinstance(retriever, RetrieverInterface), \
        "ChromaRetriever is not a subclass of RetrieverInterface"
    print("PASS: ChromaRetriever isinstance check passes")

    # 3. search method exists, is async, and has the correct signature
    method = getattr(retriever, "search", None)
    assert method is not None, "search method missing"
    assert inspect.iscoroutinefunction(method), "search must be async"

    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    assert params == ["query", "source_id", "top_k"], \
        f"Unexpected search signature: {params}"
    print("PASS: search is async with correct signature (query, source_id, top_k)")

    # 4. Calling search on a non-existent collection returns [] without crashing
    result = asyncio.run(retriever.search("test query", "nonexistent_source", top_k=5))
    assert isinstance(result, list), "search must return a list"
    assert result == [], f"Expected [] for missing collection, got {result}"
    print("PASS: search returns [] gracefully for a collection that does not exist yet")

    # 5. Return type is List[Chunk] — verify Chunk fields when non-empty
    # (Skipped here — requires an ingested collection. Covered in integration tests.)

    print("\nPASS: all retriever tests passed")


if __name__ == "__main__":
    test_retriever()

"""
core/reranker.py — Reranker interface and implementations.

Two implementations:
  FlashRanker      : reranks chunks using flashrank (ms-marco-MiniLM-L-12-v2)
  PassthroughRanker: returns chunks unchanged (used when reranking is disabled)

Factory:
  get_reranker()   : reads reranker_enabled from config.yaml, returns the
                     appropriate implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import yaml

from core.state import Chunk

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RerankerInterface(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks by relevance to query.

        Returns a new list of Chunk dicts with updated relevance_score,
        sorted highest score first.
        """
        ...


# ---------------------------------------------------------------------------
# FlashRank implementation
# ---------------------------------------------------------------------------

class FlashRanker(RerankerInterface):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker
        self._ranker = Ranker(model_name=model_name)
        logger.info("FlashRanker initialised with model %s", model_name)

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        from flashrank import RerankRequest

        if not chunks:
            return []

        # Convert Chunk dicts to the format flashrank expects
        passages = [
            {"id": i, "text": c["chunk_text"]}
            for i, c in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        # FlashRank returns list of dicts with 'id', 'text', 'score' keys.
        # 'score' is np.float32 — convert to Python float for JSON safety.
        score_map: dict[int, float] = {
            int(r["id"]): float(r["score"]) for r in results
        }

        # Update relevance_score on each chunk
        reranked: List[Chunk] = []
        for i, chunk in enumerate(chunks):
            reranked.append({
                **chunk,
                "relevance_score": score_map.get(i, chunk["relevance_score"]),
            })

        # Sort descending by relevance_score
        reranked.sort(key=lambda c: c["relevance_score"], reverse=True)
        return reranked


# ---------------------------------------------------------------------------
# Passthrough implementation
# ---------------------------------------------------------------------------

class PassthroughRanker(RerankerInterface):
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reranker() -> RerankerInterface:
    """Return FlashRanker if reranker_enabled is true in config, else PassthroughRanker."""
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    enabled = config.get("retrieval", {}).get("reranker_enabled", False)
    if enabled:
        return FlashRanker()
    return PassthroughRanker()


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.reranker
# ---------------------------------------------------------------------------

def test_reranker():
    # ── Test 1: FlashRanker reranks and updates scores ────────────
    chunks: List[Chunk] = [
        {
            "chunk_text": "The capital of France is Paris, known for the Eiffel Tower.",
            "source_pdf": "geography.pdf",
            "page_number": 1,
            "relevance_score": 0.1,
        },
        {
            "chunk_text": "Python is a popular programming language used in data science.",
            "source_pdf": "tech.pdf",
            "page_number": 3,
            "relevance_score": 0.9,
        },
        {
            "chunk_text": "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            "source_pdf": "history.pdf",
            "page_number": 7,
            "relevance_score": 0.5,
        },
    ]

    ranker = FlashRanker()
    result = ranker.rerank(query="What is the capital of France?", chunks=chunks)

    # Same number of chunks returned
    assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    # All original chunk_text values are present
    original_texts = {c["chunk_text"] for c in chunks}
    result_texts = {c["chunk_text"] for c in result}
    assert original_texts == result_texts, "Reranked chunks must contain the same texts"

    # Scores are updated (at least one should differ from original)
    original_scores = [c["relevance_score"] for c in chunks]
    result_scores = [c["relevance_score"] for c in result]
    assert original_scores != result_scores, "Scores should be updated by reranking"

    # Result is sorted descending by relevance_score
    for i in range(len(result) - 1):
        assert result[i]["relevance_score"] >= result[i + 1]["relevance_score"], \
            f"Results not sorted descending: {result[i]['relevance_score']} < {result[i + 1]['relevance_score']}"

    print(f"PASS: FlashRanker reranked 3 chunks, scores updated and sorted descending")
    for c in result:
        print(f"  score={c['relevance_score']:.4f}  text={c['chunk_text'][:60]}...")

    # ── Test 2: FlashRanker handles empty list ────────────────────
    assert ranker.rerank("anything", []) == [], "Empty input must return empty list"
    print("PASS: FlashRanker handles empty chunk list")

    # ── Test 3: PassthroughRanker returns chunks unchanged ────────
    passthrough = PassthroughRanker()
    result_pt = passthrough.rerank("any query", chunks)
    assert result_pt is chunks, "PassthroughRanker must return the same list object"
    print("PASS: PassthroughRanker returns chunks unchanged")

    # ── Test 4: get_reranker reads config ─────────────────────────
    reranker = get_reranker()
    # Config currently has reranker_enabled: true
    assert isinstance(reranker, FlashRanker), \
        f"Expected FlashRanker when enabled, got {type(reranker).__name__}"
    print("PASS: get_reranker returns FlashRanker when reranker_enabled is true")

    # ── Test 5: relevant chunks score above 0.1 after reranking ──
    policy_chunks: List[Chunk] = [
        {
            "chunk_text": "Clearance Level C employees are entitled to Economy class on all flights. "
                          "Business Class upgrade requires VP approval and flights exceeding 8 hours.",
            "source_pdf": "travel_policy.pdf",
            "page_number": 2,
            "relevance_score": 0.3,
        },
        {
            "chunk_text": "The hotel nightly allowance for domestic travel is 150 USD per night.",
            "source_pdf": "travel_policy.pdf",
            "page_number": 5,
            "relevance_score": 0.8,
        },
        {
            "chunk_text": "Per diem rates: Tier 1 cities 75 USD, Tier 2 cities 50 USD per day.",
            "source_pdf": "travel_policy.pdf",
            "page_number": 6,
            "relevance_score": 0.7,
        },
    ]

    query = "Can a clearance C employee fly Business Class?"
    reranked = ranker.rerank(query=query, chunks=policy_chunks)

    # The first chunk directly answers the question — must score well above 0.1
    top_chunk = reranked[0]
    assert "Clearance Level C" in top_chunk["chunk_text"], \
        f"Top chunk should be about clearance C, got: {top_chunk['chunk_text'][:60]}"
    assert top_chunk["relevance_score"] > 0.1, \
        f"Relevant chunk must score above 0.1, got {top_chunk['relevance_score']:.6f}"
    print(f"PASS: relevant chunk scores {top_chunk['relevance_score']:.4f} (above 0.1 threshold)")

    # Irrelevant chunks should score much lower than the top chunk
    assert reranked[0]["relevance_score"] > reranked[-1]["relevance_score"], \
        "Top chunk must score higher than least relevant chunk"
    print(f"PASS: score ordering correct — top={reranked[0]['relevance_score']:.4f}, "
          f"bottom={reranked[-1]['relevance_score']:.6f}")

    # ── Test 6: score field is correctly read from FlashRank output ─
    from flashrank import RerankRequest as RR
    raw_passages = [{"id": 0, "text": "Paris is the capital of France."}]
    raw_results = ranker._ranker.rerank(RR(query="capital of France", passages=raw_passages))
    raw_score = raw_results[0]["score"]
    assert hasattr(raw_score, "__float__"), f"FlashRank score must be numeric, got {type(raw_score)}"
    converted = float(raw_score)
    assert converted > 0.5, f"Directly relevant passage must score > 0.5, got {converted:.6f}"
    print(f"PASS: FlashRank raw 'score' field reads correctly: {converted:.4f}")

    print("\nPASS: all reranker tests passed")


if __name__ == "__main__":
    test_reranker()

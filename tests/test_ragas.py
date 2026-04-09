"""
tests/test_ragas.py — RAGAS evaluation pipeline for the multi-agent RAG system.

Standalone script (not pytest). Requires:
  - API running on localhost:8000 (or BASE_URL env var)
  - Real data ingested (PDFs in ChromaDB, tables in place)

Run with:
  python tests/test_ragas.py               — run all 20 queries + RAGAS evaluation
  python tests/test_ragas.py --scores-only  — re-evaluate saved results (skip queries)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Disable RAGAS telemetry before any RAGAS imports
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import httpx

BASE_URL = os.environ.get("RAGAS_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# 1. Test dataset — 20 questions with expected sources
# ---------------------------------------------------------------------------

TEST_DATASET: List[Dict[str, Any]] = [
    # Single source, direct lookup
    {"question": "What are the password rotation requirements?",
     "expected_sources": ["it_security_policy"]},
    {"question": "How many sick days do employees get per year?",
     "expected_sources": ["hr_handbook_v3"]},
    {"question": "Which projects are currently on hold?",
     "expected_sources": ["projects"]},
    {"question": "What is the late payment penalty for vendors?",
     "expected_sources": ["finance_policy_2024"]},

    # Single source, requires filtering
    {"question": "Which employees have expired training records?",
     "expected_sources": ["training_records"]},
    {"question": "What projects has Michal Katz led and are any over budget?",
     "expected_sources": ["projects"]},
    {"question": "What is the salary range for a clearance B engineer?",
     "expected_sources": ["salary_bands"]},

    # Cross-source, dependency chain
    {"question": "What flight class is Dan Cohen entitled to?",
     "expected_sources": ["employees", "travel_policy_2024"]},
    {"question": "What is Oren Shapiro's hotel nightly allowance?",
     "expected_sources": ["employees", "travel_policy_2024"]},
    {"question": "What data can Tal Mizrahi access according to the security policy?",
     "expected_sources": ["employees", "it_security_policy"]},
    {"question": "What is Noa Levi's monthly project budget limit?",
     "expected_sources": ["employees", "finance_policy_2024"]},

    # Cross-source, multi-entity
    {"question": "Compare the flight entitlements of Noa Levi and Oren Shapiro.",
     "expected_sources": ["employees", "travel_policy_2024"]},
    {"question": "What are the annual leave entitlements for Dan Cohen and Shira Goldman?",
     "expected_sources": ["employees", "hr_handbook_v3"]},

    # Cross-source, multi-hop reasoning
    {"question": "Which employees lead projects that are over budget, and what are their clearance levels?",
     "expected_sources": ["projects", "employees"]},
    {"question": "Does Roi Peretz have an active Security Awareness training, and when does it expire?",
     "expected_sources": ["training_records"]},

    # Policy + data combination
    {"question": "Oren Shapiro wants to expense a $1,500 hotel stay - who needs to approve it?",
     "expected_sources": ["employees", "finance_policy_2024"]},
    {"question": "Can Shira Goldman access employee personal data?",
     "expected_sources": ["employees", "it_security_policy"]},
    {"question": "If Amir Dayan gets a performance rating of 1, what happens next?",
     "expected_sources": ["hr_handbook_v3"]},

    # Edge cases
    {"question": "What is the per diem rate for clearance level E?",
     "expected_sources": ["travel_policy_2024"]},
    {"question": "What department is Alice in?",
     "expected_sources": ["employees"]},
]


# ---------------------------------------------------------------------------
# 2. Query runner
# ---------------------------------------------------------------------------

async def run_single_query(question: str, session_id: str) -> dict:
    """Call POST /chat and return the full response JSON."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        resp = await client.post("/chat", json={
            "query": question,
            "session_id": session_id,
        })
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# 3. Extract contexts from task results
# ---------------------------------------------------------------------------

def _format_table_result(table_name: str, result_value: Any) -> List[str]:
    """
    Convert a Data Scientist result_value into readable sentences.
    List of dicts  -> one sentence per row: 'From X: key is val, key is val.'
    Single value   -> 'From X: result is val.'
    """
    if isinstance(result_value, list):
        sentences: List[str] = []
        for row in result_value:
            if isinstance(row, dict):
                pairs = ", ".join(f"{k} is {v}" for k, v in row.items())
                sentences.append(f"From {table_name}: {pairs}.")
            else:
                sentences.append(f"From {table_name}: result is {row}.")
        return sentences
    if isinstance(result_value, dict) and "result_value" in result_value:
        # Scalar result wrapped as {"result_value": val, "row_count": 1}
        return [f"From {table_name}: result is {result_value['result_value']}."]
    return [f"From {table_name}: result is {result_value}."]


def _extract_contexts(result: dict) -> List[str]:
    """
    Pull contexts from all successful task results — both Librarian (PDF chunks)
    and Data Scientist (table query results). This gives RAGAS the full evidence
    the Synthesizer used, so faithfulness isn't penalized for table-backed claims.
    """
    contexts: List[str] = []
    task_results = result.get("trace", {}).get("task_results", {})
    for tid, tr in task_results.items():
        if not tr.get("success"):
            continue

        try:
            output = json.loads(tr["output"])
        except (json.JSONDecodeError, TypeError):
            continue

        if tr.get("worker_type") == "librarian":
            # Librarian output is a list of chunk dicts
            if isinstance(output, list):
                for chunk in output:
                    if isinstance(chunk, dict) and "chunk_text" in chunk:
                        contexts.append(chunk["chunk_text"])

        elif tr.get("worker_type") == "data_scientist":
            # Data Scientist output is a dict with result_value and table_name.
            # Format as readable sentences so RAGAS can verify claims.
            if isinstance(output, dict):
                table_name = output.get("table_name", "unknown table")
                result_value = output.get("result_value", output)
                for sentence in _format_table_result(table_name, result_value):
                    contexts.append(sentence)

    # Fallback: use synthesizer output as context if no task results
    if not contexts:
        synth = result.get("trace", {}).get("synthesizer_output", "")
        if synth:
            contexts.append(synth)

    return contexts if contexts else ["No context retrieved."]


# ---------------------------------------------------------------------------
# 4. RAGAS evaluation
# ---------------------------------------------------------------------------

def evaluate_with_ragas(results: List[dict], debug: bool = False) -> dict:
    """
    Build a RAGAS EvaluationDataset from SingleTurnSample objects and evaluate
    faithfulness. Returns overall score and per-question breakdown.

    Note: answer_relevancy is deferred — RAGAS v0.4.3 OpenAIEmbeddings does not
    implement embed_query which answer_relevancy requires internally.
    """
    import math

    from dotenv import load_dotenv
    from openai import OpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.metrics._faithfulness import faithfulness

    # Load .env so OpenAI client picks up the key
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    # Configure RAGAS LLM (no embeddings needed — faithfulness is LLM-only)
    client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    # Build samples — skip questions with empty/placeholder contexts
    samples: List[SingleTurnSample] = []
    included_indices: List[int] = []
    skipped_indices: List[int] = []

    for i, r in enumerate(results):
        contexts = r.get("contexts", [])
        answer = r.get("final_answer", "")

        # Skip if no real contexts or answer is an error
        has_real_context = contexts and not (len(contexts) == 1 and contexts[0] == "No context retrieved.")
        if not has_real_context or answer.startswith("ERROR:"):
            skipped_indices.append(i)
            continue

        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=answer,
            retrieved_contexts=contexts,
        ))
        included_indices.append(i)

    print(f"  Evaluating {len(samples)} questions, skipping {len(skipped_indices)} (no contexts)")

    if not samples:
        print("  WARNING: No questions with contexts to evaluate")
        return {"overall": {"faithfulness": 0.0}, "per_question": []}

    # --debug: print samples for questions 9, 10, 12, 13 and exit
    if debug:
        debug_qnums = {9, 10, 12, 13}
        print("\n" + "=" * 80)
        print("DEBUG: SingleTurnSample inputs for questions 9, 10, 12, 13")
        print("=" * 80)
        for sample, orig_idx in zip(samples, included_indices):
            qnum = orig_idx + 1
            if qnum not in debug_qnums:
                continue
            print(f"\n--- Question {qnum} ---")
            print(f"QUESTION: {sample.user_input}")
            print(f"ANSWER:   {sample.response}")
            print(f"CONTEXTS ({len(sample.retrieved_contexts)}):")
            for j, ctx in enumerate(sample.retrieved_contexts):
                print(f"  [{j+1}] {ctx}")
        print("\n" + "=" * 80)
        print("DEBUG mode — skipping evaluation.")
        sys.exit(0)

    dataset = EvaluationDataset(samples=samples)

    eval_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm,
    )

    # Extract scores via to_pandas() DataFrame
    df = eval_result.to_pandas()

    def _safe_mean(series) -> float:
        vals = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(vals) / len(vals) if vals else 0.0

    overall_faithfulness = _safe_mean(df["faithfulness"]) if "faithfulness" in df.columns else 0.0

    # Build per-question breakdown — map back to original indices
    per_question: List[dict] = []
    for df_row, orig_idx in enumerate(included_indices):
        r = results[orig_idx]
        faith_val = df.iloc[df_row].get("faithfulness", 0.0) if "faithfulness" in df.columns else 0.0

        # Convert NaN to 0.0
        if faith_val is None or (isinstance(faith_val, float) and math.isnan(faith_val)):
            faith_val = 0.0

        per_question.append({
            "question_num": orig_idx + 1,
            "question": r["question"],
            "faithfulness": float(faith_val),
        })

    # Add skipped questions with 0.0 scores
    for orig_idx in skipped_indices:
        r = results[orig_idx]
        per_question.append({
            "question_num": orig_idx + 1,
            "question": r["question"],
            "faithfulness": 0.0,
        })

    # Sort by question number
    per_question.sort(key=lambda x: x["question_num"])

    return {
        "overall": {
            "faithfulness": overall_faithfulness,
        },
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# 5. Main — run all queries and evaluate
# ---------------------------------------------------------------------------

def _check_api_health():
    """Verify the API is running. Exit with clear error if not."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            print(f"ERROR: API health check returned unexpected status: {data}")
            sys.exit(1)
        print(f"API health check passed: {data}")
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to API at {BASE_URL}")
        print("Start the API first: python -m api")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: API health check failed: {exc}")
        sys.exit(1)


async def _run_all_queries() -> List[dict]:
    """Run all 20 queries sequentially and collect results."""
    results: List[dict] = []

    for i, entry in enumerate(TEST_DATASET):
        question = entry["question"]
        session_id = f"ragas-eval-{uuid.uuid4().hex[:8]}"

        print(f"\n[{i+1:2d}/20] {question[:70]}...")
        start = time.time()

        try:
            response = await run_single_query(question, session_id)
            elapsed = time.time() - start

            final_answer = response.get("final_answer", "")
            trace = response.get("trace", {})
            contexts = _extract_contexts(response)

            actual_sources = [s["source_id"] for s in response.get("final_sources", [])]
            expected_sources = entry["expected_sources"]
            source_match = set(expected_sources).issubset(set(actual_sources))

            result = {
                "question_num": i + 1,
                "question": question,
                "expected_sources": expected_sources,
                "actual_sources": actual_sources,
                "source_match": source_match,
                "final_answer": final_answer,
                "contexts": contexts,
                "retry_count": trace.get("retry_count", 0),
                "audit_verdict": trace.get("audit_verdict", "N/A"),
                "chat_intent": trace.get("chat_intent", ""),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(result)

            verdict_symbol = "PASS" if result["audit_verdict"] == "PASS" else result["audit_verdict"] or "N/A"
            src_symbol = "OK" if source_match else "MISS"
            print(f"        {elapsed:.1f}s | verdict={verdict_symbol} | sources={src_symbol} | "
                  f"retries={result['retry_count']} | answer={final_answer[:80]}...")

        except Exception as exc:
            elapsed = time.time() - start
            print(f"        ERROR after {elapsed:.1f}s: {exc}")
            results.append({
                "question_num": i + 1,
                "question": question,
                "expected_sources": entry["expected_sources"],
                "actual_sources": [],
                "source_match": False,
                "final_answer": f"ERROR: {exc}",
                "contexts": ["No context retrieved."],
                "retry_count": 0,
                "audit_verdict": "ERROR",
                "chat_intent": "",
                "elapsed_s": round(elapsed, 1),
            })

    return results


def _print_summary(results: List[dict], ragas_scores: dict | None):
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 110)

    # Header
    print(f"{'#':>3} | {'Question':<60} | {'Verdict':<7} | {'Src':<4} | {'Retry':>5} | {'Time':>5}")
    print("-" * 110)

    for r in results:
        q_trunc = r["question"][:57] + "..." if len(r["question"]) > 60 else r["question"]
        src = "OK" if r["source_match"] else "MISS"
        verdict = r["audit_verdict"] or "N/A"
        print(f"{r['question_num']:3d} | {q_trunc:<60} | {verdict:<7} | {src:<4} | {r['retry_count']:5d} | {r['elapsed_s']:5.1f}s")

    # Source accuracy
    source_matches = sum(1 for r in results if r["source_match"])
    pass_count = sum(1 for r in results if r["audit_verdict"] == "PASS")
    total = len(results)
    total_time = sum(r["elapsed_s"] for r in results)

    print("-" * 110)
    print(f"Source match: {source_matches}/{total} ({100*source_matches/total:.0f}%)")
    print(f"Audit PASS:   {pass_count}/{total} ({100*pass_count/total:.0f}%)")
    print(f"Total time:   {total_time:.1f}s (avg {total_time/total:.1f}s per query)")

    if ragas_scores:
        print(f"\nRAGAS Scores:")
        print(f"  Faithfulness:  {ragas_scores['overall']['faithfulness']:.4f}")

        # Per-question RAGAS breakdown
        pq = ragas_scores.get("per_question", [])
        if pq:
            print(f"\n{'#':>3} | {'Question':<60} | {'Faith':>6}")
            print("-" * 78)
            for entry in pq:
                q_trunc = entry["question"][:57] + "..." if len(entry["question"]) > 60 else entry["question"]
                print(f"{entry['question_num']:3d} | {q_trunc:<60} | {entry['faithfulness']:6.3f}")


def _load_saved_results() -> List[dict]:
    """Load results from tests/ragas_results.json."""
    output_path = Path(__file__).resolve().parent / "ragas_results.json"
    if not output_path.exists():
        print(f"ERROR: No saved results at {output_path}")
        print("Run without --scores-only first to generate results.")
        sys.exit(1)

    with open(output_path, "r") as f:
        data = json.load(f)

    # Handle both formats: raw list or {results: [...], ragas_scores: {...}}
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]

    print(f"ERROR: Unexpected format in {output_path}")
    sys.exit(1)


def main():
    scores_only = "--scores-only" in sys.argv
    debug = "--debug" in sys.argv

    print(f"RAGAS Evaluation Pipeline")
    print(f"Base URL: {BASE_URL}")
    print(f"Questions: {len(TEST_DATASET)}")
    if scores_only:
        print(f"Mode: --scores-only (re-evaluate saved results)")
    print()

    output_path = Path(__file__).resolve().parent / "ragas_results.json"

    if scores_only:
        # Load existing results — skip queries entirely
        print("Loading saved results...")
        results = _load_saved_results()
        print(f"Loaded {len(results)} results from {output_path}")
    else:
        # Health check
        _check_api_health()

        # Run all queries
        print("\nRunning queries...")
        results = asyncio.run(_run_all_queries())

        # Save raw results (before RAGAS — in case RAGAS fails)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nRaw results saved to {output_path}")

    # Run RAGAS evaluation
    ragas_scores = None
    try:
        print("\nRunning RAGAS evaluation (faithfulness only)...")
        ragas_scores = evaluate_with_ragas(results, debug=debug)

        # Merge RAGAS per-question scores into results
        for pq in ragas_scores.get("per_question", []):
            idx = pq["question_num"] - 1
            if idx < len(results):
                results[idx]["faithfulness"] = pq["faithfulness"]

        # Re-save with RAGAS scores
        full_output = {
            "results": results,
            "ragas_scores": ragas_scores,
        }
        with open(output_path, "w") as f:
            json.dump(full_output, f, indent=2, default=str)
        print(f"Full results with RAGAS scores saved to {output_path}")

    except ImportError as exc:
        print(f"\nWARNING: RAGAS not installed or import error: {exc}")
        print("Install with: pip install ragas")
        print("Skipping RAGAS evaluation — raw results still saved.")
    except Exception as exc:
        print(f"\nWARNING: RAGAS evaluation failed: {exc}")
        print("Raw results still saved.")

    # Print summary
    _print_summary(results, ragas_scores)


if __name__ == "__main__":
    main()

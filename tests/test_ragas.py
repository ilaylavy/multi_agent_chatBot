"""
tests/test_ragas.py — RAGAS evaluation pipeline for the multi-agent RAG system.

Standalone script (not pytest). Requires:
  - API running on localhost:8000 (or BASE_URL env var)
  - Real data ingested (PDFs in ChromaDB, tables in place)

Run with:
  python tests/test_ragas.py                        — run all 18 queries + evaluate 1x
  python tests/test_ragas.py --multi-eval            — run queries + evaluate 3x sequentially (stable scores)
  python tests/test_ragas.py --scores-only           — re-evaluate saved results 3x sequentially
  python tests/test_ragas.py --questions 6,10,15     — re-run specific questions, merge, evaluate
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Disable RAGAS telemetry before any RAGAS imports
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import httpx

BASE_URL = os.environ.get("RAGAS_BASE_URL", "http://localhost:8000")

METRICS_LIST = ["faithfulness", "context_precision", "context_recall"]

# RAGAS column names differ from our metric names
_RAGAS_COL_MAP = {
    "faithfulness": "faithfulness",
    "context_precision": "llm_context_precision_without_reference",
    "context_recall": "context_recall",
}

THRESHOLDS = {
    "faithfulness": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}

# ---------------------------------------------------------------------------
# 1. Test dataset — 18 questions with golden answers, expected sources, tiers
# ---------------------------------------------------------------------------

TEST_DATASET: List[Dict[str, Any]] = [
    # ── Tier 1: Simple Lookup ──────────────────────────────────────────────
    {
        "question": "What are the password rotation requirements?",
        "reference_answer": (
            "Passwords must be changed every 90 days. The system prompts the user "
            "7 days before expiration. After expiration, the account is locked until "
            "a new password is set via the self-service portal or IT support."
        ),
        "expected_sources": ["it_security_policy"],
        "difficulty": "simple_lookup",
    },
    {
        "question": "How many sick days do employees get per year?",
        "reference_answer": (
            "All employees receive 10 sick days per year regardless of clearance level. "
            "A medical certificate is required for absences of 3 or more consecutive days. "
            "Unused sick leave does not carry over and is not compensated."
        ),
        "expected_sources": ["hr_handbook_v3"],
        "difficulty": "simple_lookup",
    },
    {
        "question": "What is the late payment penalty for vendors?",
        "reference_answer": (
            "A penalty of 1.5% per month is applied to any invoice not paid within "
            "the agreed payment terms. The penalty is calculated on the outstanding "
            "balance from the due date until full payment is received."
        ),
        "expected_sources": ["finance_policy_2024"],
        "difficulty": "simple_lookup",
    },
    {
        "question": "What are the payment terms for preferred vendors?",
        "reference_answer": (
            "Preferred vendors have Net 15 payment terms. Preferred vendor status is "
            "granted to vendors with a minimum 2-year relationship and annual spend "
            "exceeding $50,000. They receive priority invoice processing."
        ),
        "expected_sources": ["finance_policy_2024"],
        "difficulty": "simple_lookup",
    },
    {
        "question": "What is the salary range for a clearance level B engineer?",
        "reference_answer": (
            "The salary range for a clearance level B engineer is $110,000 to $155,000 USD."
        ),
        "expected_sources": ["salary_bands"],
        "difficulty": "simple_lookup",
    },

    # ── Tier 2: Filtering ──────────────────────────────────────────────────
    {
        "question": "Which projects are currently on hold?",
        "reference_answer": (
            "Two projects are currently on hold: Revenue Dashboard (project_id 7, "
            "Finance department, budget $55,000, spent $12,000) and Data Lake Setup "
            "(project_id 10, Engineering department, budget $150,000, spent $45,000)."
        ),
        "expected_sources": ["projects"],
        "difficulty": "filtering",
    },
    {
        "question": "Which employees have expired training records?",
        "reference_answer": (
            "Four employees have expired training records: Dan Cohen (employee_id 2, "
            "Security Awareness expired 2024-11-05), Oren Shapiro (employee_id 4, "
            "Emergency Procedures expired 2024-08-10), Tal Mizrahi (employee_id 6, "
            "Data Privacy expired 2024-06-15), and Amir Dayan (employee_id 9, "
            "Financial Compliance expired 2024-09-25)."
        ),
        "expected_sources": ["training_records"],
        "difficulty": "filtering",
    },
    {
        "question": "What are Hila Stern's training records and scores?",
        "reference_answer": (
            "Hila Stern (employee_id 10) has three completed training records: "
            "Security Awareness (score 96, completed 2024-01-05, expires 2025-01-05), "
            "Leadership Fundamentals (score 98, completed 2024-04-18, expires 2025-04-18), "
            "and Data Privacy (score 93, completed 2024-07-30, expires 2025-07-30)."
        ),
        "expected_sources": ["training_records"],
        "difficulty": "filtering",
    },
    {
        "question": "Which completed projects went over budget?",
        "reference_answer": (
            "Three completed projects went over budget: Annual Audit Automation "
            "(budget $45,000, spent $48,000, over by $3,000), CRM Integration "
            "(budget $80,000, spent $82,500, over by $2,500), and API Gateway Upgrade "
            "(budget $70,000, spent $70,500, over by $500)."
        ),
        "expected_sources": ["projects"],
        "difficulty": "filtering",
    },

    # ── Tier 3: Cross-Source ───────────────────────────────────────────────
    {
        "question": "What flight class is Dan Cohen entitled to?",
        "reference_answer": (
            "Dan Cohen has clearance level B. Employees with clearance level B are "
            "entitled to Business Class on flights exceeding 4 hours, and Economy or "
            "Economy Plus on shorter flights."
        ),
        "expected_sources": ["employees", "travel_policy_2024"],
        "difficulty": "cross_source",
    },
    {
        "question": "What is Oren Shapiro's hotel nightly allowance?",
        "reference_answer": (
            "Oren Shapiro has clearance level C. Employees with clearance level C "
            "are entitled to up to $180 per night for hotel accommodation."
        ),
        "expected_sources": ["employees", "travel_policy_2024"],
        "difficulty": "cross_source",
    },
    {
        "question": "What data can Tal Mizrahi access according to the security policy?",
        "reference_answer": (
            "Tal Mizrahi has clearance level D. According to the data access matrix, "
            "clearance level D has: Read Only access to Customer Data, Full Access to "
            "Internal Documents and Public Data, and No Access to Employee Personal "
            "Data and Financial Records."
        ),
        "expected_sources": ["employees", "it_security_policy"],
        "difficulty": "cross_source",
    },
    {
        "question": "What is Noa Levi's monthly project budget limit?",
        "reference_answer": (
            "Noa Levi is in the Engineering department with clearance level A. "
            "Her monthly project budget limit is $50,000."
        ),
        "expected_sources": ["employees", "finance_policy_2024"],
        "difficulty": "cross_source",
    },
    {
        "question": "Can Shira Goldman access employee personal data?",
        "reference_answer": (
            "Shira Goldman has clearance level C. According to the data access matrix, "
            "clearance level C has No Access to Employee Personal Data. Therefore, "
            "Shira Goldman cannot access employee personal data."
        ),
        "expected_sources": ["employees", "it_security_policy"],
        "difficulty": "cross_source",
    },

    # ── Tier 4: Multi-Hop Reasoning ───────────────────────────────────────
    {
        "question": "Compare the annual leave entitlements of Dan Cohen and Shira Goldman.",
        "reference_answer": (
            "Dan Cohen has clearance level B and is entitled to 22 days of annual "
            "leave per year. Shira Goldman has clearance level C and is entitled to "
            "18 days of annual leave per year. Dan Cohen gets 4 more days than "
            "Shira Goldman."
        ),
        "expected_sources": ["employees", "hr_handbook_v3"],
        "difficulty": "multi_hop",
    },
    {
        "question": "Which employees lead projects that are over budget, and what are their clearance levels?",
        "reference_answer": (
            "Three projects are over budget. Michal Katz (employee_id 5, clearance B) "
            "leads Annual Audit Automation (budget $45,000, spent $48,000) and CRM "
            "Integration (budget $80,000, spent $82,500). Roi Peretz (employee_id 7, "
            "clearance B) leads API Gateway Upgrade (budget $70,000, spent $70,500)."
        ),
        "expected_sources": ["projects", "employees"],
        "difficulty": "multi_hop",
    },
    {
        "question": "Oren Shapiro wants to expense a $1,500 hotel stay - who needs to approve it?",
        "reference_answer": (
            "A $1,500 expense falls in the $500-$2,000 range, which requires Manager "
            "approval plus Department Head sign-off. The turnaround time is 5 business days."
        ),
        "expected_sources": ["employees", "finance_policy_2024"],
        "difficulty": "multi_hop",
    },
    {
        "question": "If Amir Dayan receives a performance rating of 1, what happens next?",
        "reference_answer": (
            "A rating of 1 means Unsatisfactory. Any employee receiving a rating below "
            "2 is placed on a Performance Improvement Plan (PIP). The PIP lasts 90 days "
            "and includes specific, measurable goals agreed upon by the employee and "
            "manager. Progress is reviewed at 30-day intervals. Failure to meet PIP "
            "goals may result in reassignment, demotion, or termination."
        ),
        "expected_sources": ["hr_handbook_v3"],
        "difficulty": "multi_hop",
    },
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
        return [f"From {table_name}: result is {result_value['result_value']}."]
    return [f"From {table_name}: result is {result_value}."]


def _extract_contexts(result: dict) -> List[str]:
    """
    Pull contexts from all successful task results — both Librarian (PDF chunks)
    and Data Scientist (table query results). This gives RAGAS the full evidence
    the Synthesizer used.

    No fallback to synthesizer output — that is generated text, not evidence.
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
            # New format: dict with "selected_chunks"; legacy: plain list
            chunk_list = output
            if isinstance(output, dict) and "selected_chunks" in output:
                chunk_list = output["selected_chunks"]
            if isinstance(chunk_list, list):
                for chunk in chunk_list:
                    if isinstance(chunk, dict) and "chunk_text" in chunk:
                        contexts.append(chunk["chunk_text"])

        elif tr.get("worker_type") == "data_scientist":
            if isinstance(output, dict):
                table_name = output.get("table_name", "unknown table")
                result_value = output.get("result_value", output)
                for sentence in _format_table_result(table_name, result_value):
                    contexts.append(sentence)

    return contexts if contexts else ["No context retrieved."]


# ---------------------------------------------------------------------------
# 3b. Build debug trace for ragas_results.json
# ---------------------------------------------------------------------------

_CHUNK_TEXT_LIMIT = 500


def _truncate_chunks(chunks: list) -> list:
    """Truncate chunk_text fields in a list of chunk dicts."""
    for chunk in chunks:
        if isinstance(chunk, dict) and "chunk_text" in chunk:
            text = chunk["chunk_text"]
            if len(text) > _CHUNK_TEXT_LIMIT:
                chunk["chunk_text"] = text[:_CHUNK_TEXT_LIMIT] + "..."
    return chunks


def _build_debug_trace(trace: dict) -> dict:
    """Build a size-controlled debug trace from the API trace response."""
    parsed_task_results = {}
    for tid, tr in trace.get("task_results", {}).items():
        entry = {
            "worker_type": tr.get("worker_type"),
            "success": tr.get("success"),
            "error": tr.get("error"),
        }
        try:
            output = json.loads(tr["output"]) if isinstance(tr.get("output"), str) else tr.get("output")
        except (json.JSONDecodeError, TypeError):
            output = tr.get("output")

        if isinstance(output, dict):
            if "selected_chunks" in output:
                _truncate_chunks(output["selected_chunks"])
            if "chunks" in output:
                _truncate_chunks(output["chunks"])

        entry["output"] = output
        parsed_task_results[tid] = entry

    planner_reasoning = trace.get("planner_reasoning")
    if isinstance(planner_reasoning, str):
        try:
            planner_reasoning = json.loads(planner_reasoning)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "chat_intent": trace.get("chat_intent", ""),
        "rewritten_query": trace.get("rewritten_query", ""),
        "query_sent_to_planner": trace.get("query_sent_to_planner"),
        "planner_reasoning": planner_reasoning,
        "plan": trace.get("plan"),
        "task_results": parsed_task_results,
        "synthesizer_output": trace.get("synthesizer_output", ""),
        "audit_verdict": trace.get("audit_verdict"),
        "audit_notes": trace.get("audit_notes"),
        "retry_count": trace.get("retry_count", 0),
        "retry_history": trace.get("retry_history", []),
        "step_timings": trace.get("step_timings", {}),
        "chat_formatted_response": trace.get("chat_formatted_response"),
    }


# ---------------------------------------------------------------------------
# 4. RAGAS evaluation — all four metrics
# ---------------------------------------------------------------------------

def evaluate_with_ragas(results: List[dict]) -> dict:
    """
    Build a RAGAS EvaluationDataset and evaluate with four metrics:
    faithfulness, factual_correctness, context_precision, context_recall.
    Returns overall scores and per-question breakdown.
    """
    from dotenv import load_dotenv
    from openai import OpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
    from ragas.metrics._context_recall import LLMContextRecall

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=client, temperature=0.0)

    metrics_map = {
        "faithfulness": Faithfulness(),
        "context_precision": LLMContextPrecisionWithoutReference(),
        "context_recall": LLMContextRecall(),
    }

    samples: List[SingleTurnSample] = []
    included_indices: List[int] = []
    skipped: List[dict] = []

    for i, r in enumerate(results):
        if r.get("status") == "error":
            skipped.append({"index": i, "reason": "system_error", "question": r["question"]})
            continue

        contexts = r.get("contexts", [])
        has_context = contexts and not (len(contexts) == 1 and contexts[0] == "No context retrieved.")
        if not has_context:
            skipped.append({"index": i, "reason": "no_context", "question": r["question"]})
            continue

        answer = r.get("final_answer", "")
        if answer.startswith("ERROR:"):
            skipped.append({"index": i, "reason": "error_response", "question": r["question"]})
            continue

        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=r.get("reference_answer", ""),
        ))
        included_indices.append(i)

    print(f"  Evaluating {len(samples)} questions, skipping {len(skipped)}")

    if not samples:
        print("  WARNING: No questions with contexts to evaluate")
        return {"overall": {m: 0.0 for m in METRICS_LIST}, "per_question": [], "skipped": skipped}

    dataset = EvaluationDataset(samples=samples)

    def _safe_val(series_val) -> float:
        if series_val is None or (isinstance(series_val, float) and math.isnan(series_val)):
            return 0.0
        return float(series_val)

    def _safe_mean(vals: List[float]) -> float:
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else 0.0

    # Run each metric separately to avoid deadlocks
    all_dfs = {}
    for metric_name, metric_obj in metrics_map.items():
        ragas_col = _RAGAS_COL_MAP[metric_name]
        print(f"    Running {metric_name}...", flush=True)
        eval_result = evaluate(dataset=dataset, metrics=[metric_obj], llm=llm)
        df = eval_result.to_pandas()
        if ragas_col in df.columns:
            all_dfs[metric_name] = [_safe_val(df.iloc[row].get(ragas_col, 0.0)) for row in range(len(df))]
        else:
            all_dfs[metric_name] = [0.0] * len(samples)
        print(f"    {metric_name} done — avg {_safe_mean(all_dfs[metric_name]):.3f}")

    # Build per-question scores
    per_question: List[dict] = []
    metric_totals: Dict[str, List[float]] = {m: [] for m in METRICS_LIST}

    for df_row, orig_idx in enumerate(included_indices):
        r = results[orig_idx]
        scores: Dict[str, float] = {}
        for m in METRICS_LIST:
            scores[m] = all_dfs[m][df_row]
            metric_totals[m].append(scores[m])

        per_question.append({
            "question_num": r["question_num"],
            "question": r["question"],
            "difficulty": r.get("difficulty", ""),
            **scores,
        })

    for skip in skipped:
        orig_idx = skip["index"]
        r = results[orig_idx]
        per_question.append({
            "question_num": r["question_num"],
            "question": r["question"],
            "difficulty": r.get("difficulty", ""),
            "status": skip["reason"],
            **{m: 0.0 for m in METRICS_LIST},
        })

    per_question.sort(key=lambda x: x["question_num"])

    overall = {m: _safe_mean(metric_totals[m]) for m in METRICS_LIST}

    return {
        "overall": overall,
        "per_question": per_question,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# 5. Multi-run averaging
# ---------------------------------------------------------------------------

def _evaluate_multi_run(results: List[dict], n_runs: int = 3) -> dict:
    """
    Run RAGAS evaluation n_runs times and return averaged scores with stddev.
    """
    all_scores: Dict[int, Dict[str, List[float]]] = {}

    print(f"  Running {n_runs} RAGAS evaluations sequentially...")
    run_results = {}
    for run_num in range(1, n_runs + 1):
        run_results[run_num] = evaluate_with_ragas(results)
        print(f"  --- RAGAS run {run_num}/{n_runs} complete ---")

    # Collect all skipped info from any run
    all_skipped = run_results.get(1, {}).get("skipped", [])

    for run_num in sorted(run_results):
        for pq in run_results[run_num].get("per_question", []):
            qnum = pq["question_num"]
            if qnum not in all_scores:
                all_scores[qnum] = {m: [] for m in METRICS_LIST}
            for m in METRICS_LIST:
                all_scores[qnum][m].append(pq.get(m, 0.0))

    # Build per-question averages
    per_question: List[dict] = []
    metric_avgs: Dict[str, List[float]] = {m: [] for m in METRICS_LIST}

    for qnum in sorted(all_scores.keys()):
        q_text = ""
        difficulty = ""
        for r in results:
            if r.get("question_num") == qnum:
                q_text = r["question"]
                difficulty = r.get("difficulty", "")
                break

        entry: dict = {
            "question_num": qnum,
            "question": q_text,
            "difficulty": difficulty,
        }

        for m in METRICS_LIST:
            scores = all_scores[qnum][m]
            avg = sum(scores) / len(scores) if scores else 0.0
            stddev = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores)) if scores else 0.0
            entry[m] = avg
            entry[f"{m}_stddev"] = stddev
            entry[f"{m}_runs"] = scores
            metric_avgs[m].append(avg)

        per_question.append(entry)

    overall: Dict[str, Any] = {}
    for m in METRICS_LIST:
        vals = metric_avgs[m]
        avg = sum(vals) / len(vals) if vals else 0.0
        stddev = math.sqrt(sum((v - avg) ** 2 for v in vals) / len(vals)) if vals else 0.0
        overall[m] = avg
        overall[f"{m}_stddev"] = stddev

    return {
        "overall": overall,
        "per_question": per_question,
        "skipped": all_skipped,
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# 6. Main — run queries and evaluate
# ---------------------------------------------------------------------------

def _check_api_health():
    """Verify the API is running."""
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


async def _run_all_queries(question_nums: set[int] | None = None) -> List[dict]:
    """Run queries sequentially. If question_nums is set, run only those."""
    results: List[dict] = []
    total = len(question_nums) if question_nums else len(TEST_DATASET)

    for i, entry in enumerate(TEST_DATASET):
        qnum = i + 1
        if question_nums and qnum not in question_nums:
            continue
        question = entry["question"]
        session_id = f"ragas-eval-{uuid.uuid4().hex[:8]}"

        print(f"\n[{qnum:2d}/{len(TEST_DATASET)}] {question[:70]}...")
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
                "question_num": qnum,
                "question": question,
                "reference_answer": entry["reference_answer"],
                "difficulty": entry["difficulty"],
                "expected_sources": expected_sources,
                "actual_sources": actual_sources,
                "source_match": source_match,
                "status": "success",
                "final_answer": final_answer,
                "contexts": contexts,
                "retry_count": trace.get("retry_count", 0),
                "audit_verdict": trace.get("audit_verdict", "N/A"),
                "chat_intent": trace.get("chat_intent", ""),
                "elapsed_s": round(elapsed, 1),
                "trace": _build_debug_trace(trace),
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
                "question_num": qnum,
                "question": question,
                "reference_answer": entry["reference_answer"],
                "difficulty": entry["difficulty"],
                "expected_sources": entry["expected_sources"],
                "actual_sources": [],
                "source_match": False,
                "status": "error",
                "final_answer": f"ERROR: {exc}",
                "contexts": ["No context retrieved."],
                "retry_count": 0,
                "audit_verdict": "ERROR",
                "chat_intent": "",
                "elapsed_s": round(elapsed, 1),
                "trace": None,
            })

    return results


def _print_summary(results: List[dict], ragas_scores: dict | None):
    """Print a formatted summary table."""
    print("\n" + "=" * 120)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 120)

    n_runs = ragas_scores.get("n_runs") if ragas_scores else None
    has_multi = n_runs is not None

    # ── Per-question table ─────────────────────────────────────────────
    header = (f"{'#':>3} | {'Question':<50} | {'Tier':<13} | {'Src':<4} | "
              f"{'Faith':>6} | {'Fact':>6} | {'CtxP':>6} | {'CtxR':>6}")
    print(header)
    print("-" * 120)

    pq_map: Dict[int, dict] = {}
    if ragas_scores:
        for pq in ragas_scores.get("per_question", []):
            pq_map[pq["question_num"]] = pq

    for r in results:
        q_trunc = r["question"][:47] + "..." if len(r["question"]) > 50 else r["question"]
        src = "OK" if r["source_match"] else "MISS"
        difficulty = r.get("difficulty", "")[:13]

        scores = pq_map.get(r["question_num"], {})
        faith = scores.get("faithfulness", 0.0)
        fact = scores.get("factual_correctness", 0.0)
        ctx_p = scores.get("context_precision", 0.0)
        ctx_r = scores.get("context_recall", 0.0)

        status = r.get("status", "success")
        if status == "error":
            print(f"{r['question_num']:3d} | {q_trunc:<50} | {difficulty:<13} | {src:<4} | {'ERROR':>6} | {'ERROR':>6} | {'ERROR':>6} | {'ERROR':>6}")
        else:
            print(f"{r['question_num']:3d} | {q_trunc:<50} | {difficulty:<13} | {src:<4} | {faith:6.3f} | {fact:6.3f} | {ctx_p:6.3f} | {ctx_r:6.3f}")

    # ── Summary stats ─────────────────────────────────────────────────
    source_matches = sum(1 for r in results if r["source_match"])
    pass_count = sum(1 for r in results if r["audit_verdict"] == "PASS")
    error_count = sum(1 for r in results if r.get("status") == "error")
    total = len(results)
    total_time = sum(r["elapsed_s"] for r in results)

    print("-" * 120)
    print(f"Source match: {source_matches}/{total} ({100*source_matches/total:.0f}%)")
    print(f"Audit PASS:   {pass_count}/{total} ({100*pass_count/total:.0f}%)")
    if error_count:
        print(f"Errors:       {error_count}/{total} (excluded from RAGAS scores)")
    print(f"Total time:   {total_time:.1f}s (avg {total_time/total:.1f}s per query)")

    if ragas_scores:
        overall = ragas_scores["overall"]
        print(f"\n{'Metric':<25} | {'Score':>7} | {'StdDev':>7} | {'Threshold':>9} | {'Result':>6}")
        print("-" * 70)
        for m in METRICS_LIST:
            score = overall.get(m, 0.0)
            stddev = overall.get(f"{m}_stddev", 0.0)
            threshold = THRESHOLDS[m]
            passed = score >= threshold
            label = "PASS" if passed else "FAIL"
            if has_multi:
                print(f"{m:<25} | {score:7.4f} | {stddev:7.4f} | {threshold:9.2f} | {label:>6}")
            else:
                print(f"{m:<25} | {score:7.4f} | {'  N/A':>7} | {threshold:9.2f} | {label:>6}")

        # ── Per-difficulty breakdown ──────────────────────────────────
        tiers = ["simple_lookup", "filtering", "cross_source", "multi_hop"]
        tier_scores: Dict[str, Dict[str, List[float]]] = {t: {m: [] for m in METRICS_LIST} for t in tiers}
        for pq in ragas_scores.get("per_question", []):
            tier = pq.get("difficulty", "")
            if tier in tier_scores and pq.get("status") != "system_error":
                for m in METRICS_LIST:
                    tier_scores[tier][m].append(pq.get(m, 0.0))

        print(f"\n{'Difficulty':<15}", end="")
        for m in METRICS_LIST:
            print(f" | {m[:8]:>8}", end="")
        print(f" |  {'N':>3}")
        print("-" * 70)
        for tier in tiers:
            n = len(tier_scores[tier][METRICS_LIST[0]])
            print(f"{tier:<15}", end="")
            for m in METRICS_LIST:
                vals = tier_scores[tier][m]
                avg = sum(vals) / len(vals) if vals else 0.0
                print(f" | {avg:8.3f}", end="")
            print(f" |  {n:3d}")

    # ── Skipped / errors ──────────────────────────────────────────────
    if ragas_scores and ragas_scores.get("skipped"):
        print(f"\nSkipped questions:")
        for s in ragas_scores["skipped"]:
            print(f"  Q{s['index']+1}: {s['reason']} — {s['question'][:60]}")


def _load_saved_results() -> List[dict]:
    """Load results from tests/ragas_results.json."""
    output_path = Path(__file__).resolve().parent / "ragas_results.json"
    if not output_path.exists():
        print(f"ERROR: No saved results at {output_path}")
        print("Run without --scores-only first to generate results.")
        sys.exit(1)

    with open(output_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]

    print(f"ERROR: Unexpected format in {output_path}")
    sys.exit(1)


def _parse_questions_arg() -> set[int] | None:
    """Parse --questions 6,10,15 from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == "--questions" and i + 1 < len(sys.argv):
            try:
                nums = {int(n.strip()) for n in sys.argv[i + 1].split(",")}
                invalid = {n for n in nums if n < 1 or n > len(TEST_DATASET)}
                if invalid:
                    print(f"ERROR: Invalid question numbers: {invalid}. Valid range: 1-{len(TEST_DATASET)}")
                    sys.exit(1)
                return nums
            except ValueError:
                print(f"ERROR: --questions expects comma-separated integers, got: {sys.argv[i + 1]}")
                sys.exit(1)
    return None


def main():
    scores_only = "--scores-only" in sys.argv
    multi_eval = "--multi-eval" in sys.argv
    question_nums = _parse_questions_arg()

    print(f"RAGAS Evaluation Pipeline")
    print(f"Base URL: {BASE_URL}")
    print(f"Questions: {len(TEST_DATASET)}")
    print(f"Metrics: {', '.join(METRICS_LIST)}")
    if scores_only:
        print(f"Mode: --scores-only (re-evaluate saved results)")
    if multi_eval:
        print(f"Mode: --multi-eval (3 sequential evaluation runs for stable averaging)")
    if question_nums:
        print(f"Mode: --questions {','.join(str(n) for n in sorted(question_nums))}")
    print()

    output_path = Path(__file__).resolve().parent / "ragas_results.json"

    if scores_only and not question_nums:
        print("Loading saved results...")
        results = _load_saved_results()
        print(f"Loaded {len(results)} results from {output_path}")
    elif question_nums:
        _check_api_health()
        print(f"\nRunning {len(question_nums)} selected queries...")
        new_results = asyncio.run(_run_all_queries(question_nums))

        if output_path.exists():
            existing = _load_saved_results()
            results_map: dict[int, dict] = {r["question_num"]: r for r in existing}
        else:
            results_map = {}

        for r in new_results:
            results_map[r["question_num"]] = r
        results = [results_map[n] for n in sorted(results_map.keys())]

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nMerged results saved to {output_path} ({len(results)} total, {len(new_results)} updated)")
    else:
        _check_api_health()
        print("\nRunning queries...")
        results = asyncio.run(_run_all_queries())

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nRaw results saved to {output_path}")

    # Run RAGAS evaluation
    ragas_scores = None
    try:
        if multi_eval:
            print("\nRunning RAGAS evaluation x3 sequentially (all 4 metrics)...")
            ragas_scores = _evaluate_multi_run(results, n_runs=3)
        else:
            print("\nRunning RAGAS evaluation (1 run, all 4 metrics)...")
            ragas_scores = evaluate_with_ragas(results)

        # Merge RAGAS per-question scores into results
        for pq in ragas_scores.get("per_question", []):
            idx = pq["question_num"] - 1
            if idx < len(results):
                for m in METRICS_LIST:
                    results[idx][m] = pq.get(m, 0.0)

        # Save full output
        full_output = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_url": BASE_URL,
                "n_questions": len(TEST_DATASET),
                "n_evaluated": len(results) - len(ragas_scores.get("skipped", [])),
                "n_errors": sum(1 for r in results if r.get("status") == "error"),
                "n_eval_runs": ragas_scores.get("n_runs", 1),
                "ragas_version": "0.4.3",
                "judge_model": "gpt-4o-mini",
            },
            "thresholds": THRESHOLDS,
            "overall": ragas_scores["overall"],
            "by_difficulty": _compute_by_difficulty(ragas_scores),
            "results": results,
            "skipped": ragas_scores.get("skipped", []),
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
        import traceback
        traceback.print_exc()
        print("Raw results still saved.")

    _print_summary(results, ragas_scores)


def _compute_by_difficulty(ragas_scores: dict) -> dict:
    """Compute average scores per difficulty tier."""
    tiers = ["simple_lookup", "filtering", "cross_source", "multi_hop"]
    tier_scores: Dict[str, Dict[str, List[float]]] = {t: {m: [] for m in METRICS_LIST} for t in tiers}

    for pq in ragas_scores.get("per_question", []):
        tier = pq.get("difficulty", "")
        if tier in tier_scores and pq.get("status") not in ("system_error", "no_context", "error_response"):
            for m in METRICS_LIST:
                tier_scores[tier][m].append(pq.get(m, 0.0))

    result = {}
    for tier in tiers:
        result[tier] = {}
        for m in METRICS_LIST:
            vals = tier_scores[tier][m]
            result[tier][m] = sum(vals) / len(vals) if vals else 0.0
        result[tier]["n"] = len(tier_scores[tier][METRICS_LIST[0]])
    return result


if __name__ == "__main__":
    main()

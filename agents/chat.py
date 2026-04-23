"""
agents/chat.py — Chat Agent node (The Face).

LangGraph node: async chat_node(state) -> dict

Four execution paths:

  RETRY EXHAUSTION  — final_answer empty + retry_count >= max_attempts.
                      Returns a failure message (LLM-composed partial answer
                      when synthesizer_output exists, else the config default).

  FAST GREETING     — obvious greetings match a regex; no LLM call.
                      Returns a canned DIRECT response.

  REASON AND ROUTE  — two LLM calls. First a scope filter (gpt-4.1-nano,
                      data_context aware) decides whether the message is
                      in-scope. Out-of-scope short-circuits with a friendly
                      reply. In-scope proceeds to the reasoning call (no
                      data_context) which produces DIRECT, CLARIFY, or PLAN
                      with a self-contained rewritten query and extracts
                      stable user facts into session_context ambiently.

  FORMAT AND DELIVER — final_answer populated by pipeline (audit verdict
                      PASS). LLM reformats the verified answer. Post-format
                      safety check rejects polarity inversions.

View    : original_query, conversation_history, final_answer, final_sources,
          chat_intent, rewritten_query
Returns : { conversation_history, chat_intent, rewritten_query, chat_reasoning }          on PLAN
          { conversation_history, chat_intent, final_answer, chat_reasoning }              on DIRECT / CLARIFY from reasoning
          { conversation_history, chat_intent, final_answer }                              on out-of-scope short-circuit
          { conversation_history, final_answer }                                            on FORMAT or EXHAUSTION
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from core.data_context import get_data_context
from core.llm_config import _load_config, get_llm
from core.parse import parse_llm_json
from core.prompt_capture import capture as capture_prompt
from core.scope_result import set_scope_result
from core.session_context import get_session_context, update_session_context
from core.state import AgentState, Message

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_retry_cfg       = _load_config()["retry"]
_FAILURE_MESSAGE: str = _retry_cfg["failure_message"]
_MAX_ATTEMPTS:    int = _retry_cfg["max_attempts"]

# Reasoning call gets a generous history window — it must reconstruct intent
# across clarification cycles and keep session facts consistent. Configurable
# via config.yaml chat.max_reasoning_history; default 12.
_chat_cfg = _load_config().get("chat", {})
MAX_REASONING_HISTORY: int = _chat_cfg.get("max_reasoning_history", 12)


# ---------------------------------------------------------------------------
# Prompt templates — scope filter (first call)
# ---------------------------------------------------------------------------

_SCOPE_SYSTEM_PROMPT = """\
You are a scope filter. You receive a user message and a description of what
data this system covers. Your only job: should this message be processed
further, or is it clearly outside the system's scope?

Return in_scope for:
  - Any question that relates to the topics described in DATA CONTEXT
  - Any message where the user provides information about themselves
  - Any message that includes a data question, even if mixed with other
    content
  - Any statement or question you are unsure about

Return out_of_scope for:
  - Questions clearly and unambiguously outside every topic described in
    DATA CONTEXT — judge only by comparing the message to DATA CONTEXT,
    not by your own knowledge of what topics exist
  - Pure chit-chat with no personal content and no data question

When in doubt, always return in_scope. A false in_scope is cheap. A false
out_of_scope means the system ignores the user.

Respond with ONLY JSON:
{
  "scope": "in_scope" | "out_of_scope",
  "response": "short friendly reply when out_of_scope, empty string when in_scope"
}
"""

_SCOPE_USER_TEMPLATE = """\
DATA CONTEXT:
{data_context}

USER MESSAGE:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — reasoning call (second call)
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT = """\
You are the entry agent of a data-answering system. You turn human
conversation into a single clear question that a downstream system with no
conversation memory can answer.

You see conversation history and session context. You do NOT see any
description of what data exists — you know nothing about the data. Your
only concern is the conversation.

STEP 1 — UNDERSTAND AND EXTRACT
  What does the user want?
  Scan their message for any self-identifying facts they volunteer and
  emit them in session_context_update under stable keys (user_name,
  department, role). Corrections overwrite prior values. Do this on
  every turn regardless of the decision you reach.

STEP 2 — DECIDE
  Pick exactly one:

  PLAN — the user is asking a question that requires looking up data.
  Write output as a single natural question that stands completely alone.
  Resolve every reference from conversation history and session context.
  Preserve every part of a multi-part question. Do not mention sources,
  tables, records, documents, or files. Do not use procedural language.
  Write it the way a person would ask the question.

  CLARIFY — the user is asking a data question but you cannot identify
  who or what it is about. The only valid reason: an unresolved "my" or
  "I" with no name in session context, or a pronoun with no clear
  referent. Write output as one short question targeting exactly what
  is missing. Never clarify about scope, detail level, or preferences.
  If the previous assistant turn was a clarification, treat the user's
  response as an answer and move to PLAN — never clarify twice in a row.

  DIRECT — anything other than a data question: greetings, identity
  statements, conversation, statements without a question, corrections
  to personal info. Write output as a short response.

STEP 3 — REWRITE (PLAN only)
  Output PLAN with rewritten_query: ONE clean natural question, the way a
  person would ask it. Resolve every implicit reference using the
  conversation history and SESSION CONTEXT. If the user asked about more
  than one thing, preserve every part. The downstream Planner receives no
  conversation history, so the question must stand completely on its own.

  The query is natural language. Do not mention sources, tables, records,
  documents, or files. Do not use procedural wording such as "retrieve",
  "provide", "lookup", "if available", or "otherwise". Do not instruct the
  system what to do on failure. Write the question the way a user would
  type it.

Additional rules
  - Post-clarification confirmations ("yes", "right", "correct") must
    reconstruct the original concrete question using the clarified
    information. Never echo the clarifying question back as the
    rewritten_query.
  - You never tell the user that something was not found or does not exist —
    you do not know what the data holds.
  - you can only respond with **one** of (rewritten_query, clarifying_question, direct_response) depending on the decision.
Respond with ONLY JSON:
{
  "reasoning": {
    "user_intent":         "one sentence",
    "decision_rationale":  "one sentence"
  },
  "decision":                "PLAN|CLARIFY|DIRECT",
  "rewritten_query":         "",
  "clarifying_question":     "",
  "direct_response":         "",
  "session_context_update":  {}
}
"""

_REASONING_USER_TEMPLATE = """\
{session_context_block}{history_block}CURRENT USER MESSAGE:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — format and deliver (unchanged)
# ---------------------------------------------------------------------------

_DELIVER_SYSTEM_PROMPT = """\
You are a copyeditor, not an author. Every name, number, category, date, and
conclusion in the VERIFIED ANSWER must survive in your output. You may change
tone, length, and phrasing — never facts. If the verified answer states a
positive finding (a name, a value, a rule), your output must state that same
positive finding. Never invert polarity: do not output "not found", "does not
exist", "no record", "no such X", or any negation unless the verified answer
itself contains that same negation.

Rewrite the VERIFIED ANSWER as a direct, concise response to the user's
question. Use the RECENT CONVERSATION block only to understand what the user
was asking about — never to add facts the verified answer did not contain.

Rules:
  - Lead with the key fact.
  - Remove source attribution phrases like "per the [source name]",
    "according to [document name]", "per Section N of [document]" —
    sources are shown separately in the UI.
  - Maximum 2 sentences for simple factual answers.
  - Keep the answer concise but never drop conditions, exceptions, thresholds,
    or approval requirements — these are critical for policy questions. If the
    answer contains a rule with an exception such as "permitted only when X" or
    "requires approval for Y", always include that exception in the final answer.
  - Do not add information not in the answer. Never add advice, suggestions, or
    recommendations not directly stated in the answer.
  - Never refer to source documents by name in the formatted answer.
  - Do not reveal internal system details.
  - Do not hedge or add uncertainty.
  - If the answer explains that data was not found, tell the user clearly and
    specifically what was not found — for example: "There is no [entity type]
    named [name] in the system. Did you mean a different name?"
"""

_DELIVER_USER_TEMPLATE = """\
{history_block}USER QUESTION (resolved from conversation context):
{resolved_query}

VERIFIED ANSWER (rewrite for the user):
{final_answer}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def chat_view(state: AgentState) -> dict:
    """
    Returns only the fields the Chat Agent's LLM prompts may see.
    Never exposes plan, task_results, audit_result, retry_count, retry_notes,
    or chat_reasoning (written by this agent — must not feed back into its
    own prompt).

    rewritten_query is included so the formatter can use the self-contained
    query (set on the PLAN path) when composing the deliver prompt.
    """
    return {
        "original_query":       state["original_query"],
        "conversation_history": state["conversation_history"],
        "final_answer":         state["final_answer"],
        "final_sources":        state["final_sources"],
        "chat_intent":          state.get("chat_intent", ""),
        "rewritten_query":      state.get("rewritten_query", ""),
    }


# ---------------------------------------------------------------------------
# Internal helpers — prompt formatting
# ---------------------------------------------------------------------------

def _format_sources_block(final_sources: list) -> str:
    if not final_sources:
        return "  (none)"
    lines = []
    for src in final_sources:
        src_id   = src["source_id"]
        src_type = src["source_type"]
        label    = src["label"]
        lines.append(f"  - {src_id} [{src_type}] — {label}")
    return "\n".join(lines)


def _format_history_block(history: list, max_messages: int = MAX_REASONING_HISTORY) -> str:
    """Return the last `max_messages` messages formatted for the reasoning prompt."""
    recent = history[-max_messages:] if len(history) > max_messages else history
    if not recent:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_recent_for_deliver(history: list, last_n: int = 2) -> str:
    """Return the last `last_n` messages formatted for the deliver prompt."""
    recent = history[-last_n:] if len(history) > last_n else history
    if not recent:
        return ""
    lines = ["RECENT CONVERSATION:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_session_context_block(ctx: dict) -> str:
    """Render the session context dict for the reasoning prompt."""
    if not ctx:
        return "SESSION CONTEXT: (empty — nothing known about this user yet)\n\n"
    lines = ["SESSION CONTEXT (stable facts from prior turns):"]
    for k, v in ctx.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    return "\n".join(lines) + "\n"


# Negation phrases the formatter must never introduce on its own.
_NEGATION_PATTERNS: tuple[str, ...] = (
    "not found",
    "no employee",
    "no record",
    "does not exist",
    "no such",
    "could not find",
    "no results",
    "isn't in",
    "is not in",
)


def _contains_negation(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _NEGATION_PATTERNS)


def _append_messages(
    history: list,
    query: str,
    answer: str | None = None,
) -> list[Message]:
    """Return a new history list with query (and optionally answer) appended."""
    updated = list(history)
    if not updated or updated[-1].get("content") != query:
        updated.append(Message(role="user", content=query))
    if answer is not None:
        updated.append(Message(role="assistant", content=answer))
    return updated


# ---------------------------------------------------------------------------
# Reasoning output helpers
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"PLAN", "CLARIFY", "DIRECT"}


def _fallback_reasoning(original_query: str, raw: str | None = None) -> dict:
    """Return a safe reasoning payload when parsing fails.

    The Planner always gets *something*, so a malformed LLM response cannot
    block the pipeline entirely — it degrades to a straight PLAN with the
    user's original query.
    """
    if raw:
        logger.warning("Reasoning call produced unparseable output; falling back to PLAN. Raw: %s", raw[:400])
    return {
        "reasoning": {
            "user_intent":         "fallback — reasoning call failed to return parseable JSON",
            "decision_rationale":  "Parse failure — default to PLAN with original query.",
        },
        "decision":               "PLAN",
        "rewritten_query":        original_query,
        "clarifying_question":    "",
        "direct_response":        "",
        "session_context_update": {},
    }


def _normalize_reasoning(data: dict, original_query: str) -> dict:
    """Ensure the reasoning dict has every expected field with safe defaults."""
    reasoning = data.get("reasoning") or {}
    decision  = (data.get("decision") or "").upper().strip()
    if decision not in _VALID_DECISIONS:
        logger.warning("Reasoning returned invalid decision %r — coercing to PLAN", decision)
        decision = "PLAN"

    rewritten = (data.get("rewritten_query") or "").strip()
    if decision == "PLAN" and not rewritten:
        rewritten = original_query

    return {
        "reasoning": {
            "user_intent":         reasoning.get("user_intent", "") or "",
            "decision_rationale":  reasoning.get("decision_rationale", "") or "",
        },
        "decision":               decision,
        "rewritten_query":        rewritten,
        "clarifying_question":   (data.get("clarifying_question") or "").strip(),
        "direct_response":       (data.get("direct_response") or "").strip(),
        "session_context_update": data.get("session_context_update") or {},
    }


def _serialize_reasoning(payload: dict) -> str:
    """Render the reasoning dict as a stable JSON string for the trace."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fast pre-classification — skip LLM for obvious greetings
# ---------------------------------------------------------------------------

_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye)[\s!.,?]*$",
    re.IGNORECASE,
)

_CANNED_RESPONSES: dict[str, str] = {
    "hi":        "Hello! Ask me anything about your company data.",
    "hello":     "Hello! Ask me anything about your company data.",
    "hey":       "Hey! Ask me anything about your company data.",
    "thanks":    "You're welcome! Let me know if you have more questions.",
    "thank you": "You're welcome! Let me know if you have more questions.",
    "bye":       "Goodbye! Feel free to start a new session any time.",
    "goodbye":   "Goodbye! Feel free to start a new session any time.",
}

_MAX_FAST_LENGTH = 20


def _try_fast_classify(query: str) -> str | None:
    """Return a canned response if the query is an obvious greeting, else None."""
    if len(query) > _MAX_FAST_LENGTH:
        return None
    m = _GREETING_PATTERN.match(query.strip())
    if not m:
        return None
    key = m.group(1).lower()
    return _CANNED_RESPONSES.get(key)


# ---------------------------------------------------------------------------
# Out-of-scope fallback
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_FALLBACK = "I can help with questions about your company data."


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def chat_node(state: AgentState) -> dict:
    view = chat_view(state)

    final_answer = view["final_answer"]
    retry_count  = state["retry_count"]
    session_id   = state.get("session_id", "?")

    # ── Path 1: RETRY EXHAUSTION ──────────────────────────────────
    if not final_answer and retry_count >= _MAX_ATTEMPTS:
        synthesizer_output = state.get("synthesizer_output", "")
        if synthesizer_output:
            llm = get_llm("chat")
            exhaust_system = (
                "The system could not fully verify an answer. "
                f"Here is what was found: {synthesizer_output}. "
                "Write a honest 1-2 sentence response telling the user "
                "what happened and what was found, even if incomplete. Be direct."
            )
            capture_prompt(
                session_id, 0,
                "chat", "exhaust",
                exhaust_system, view["original_query"],
            )
            exhaust_response = await llm.ainvoke([
                {"role": "system", "content": exhaust_system},
                {"role": "user",   "content": view["original_query"]},
            ])
            answer = exhaust_response.content.strip()
        else:
            answer = _FAILURE_MESSAGE

        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            answer,
        )
        return {
            "conversation_history": updated_history,
            "final_answer":         answer,
        }

    # ── Path 2: REASON AND ROUTE ──────────────────────────────────
    if not final_answer:
        # Fast path — skip both LLM calls for obvious greetings
        fast_response = _try_fast_classify(view["original_query"])
        if fast_response is not None:
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
                fast_response,
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "DIRECT",
                "final_answer":         fast_response,
            }

        # Step A: scope filter call — sees DATA CONTEXT, not conversation.
        scope_user_message = _SCOPE_USER_TEMPLATE.format(
            data_context=get_data_context(),
            original_query=view["original_query"],
        )
        scope_llm = get_llm("chat_scope")
        capture_prompt(
            session_id, 0,
            "chat", "scope",
            _SCOPE_SYSTEM_PROMPT, scope_user_message,
        )
        scope_response = await scope_llm.ainvoke([
            {"role": "system", "content": _SCOPE_SYSTEM_PROMPT},
            {"role": "user",   "content": scope_user_message},
        ])

        scope_verdict = "in_scope"
        scope_reply   = ""
        try:
            scope_data   = parse_llm_json(scope_response.content)
            raw_scope    = (scope_data.get("scope") or "").strip().lower()
            scope_verdict = raw_scope if raw_scope in ("in_scope", "out_of_scope") else "in_scope"
            scope_reply   = (scope_data.get("response") or "").strip()
        except ValueError as exc:
            logger.warning(
                "[%s] scope parse failed — defaulting to in_scope. Raw: %s",
                session_id, str(scope_response.content)[:200],
            )

        # Persist the scope outcome so api.py can surface it in the trace.
        # Written only on paths that actually made the scope call — the fast
        # greeting short-circuits above and therefore leaves scope_result unset
        # (api.py clears the store at the start of every /chat request).
        set_scope_result(session_id, scope_verdict, scope_reply)

        logger.debug(
            "[%s] Chat scope — verdict=%s has_reply=%s",
            session_id, scope_verdict, bool(scope_reply),
        )

        if scope_verdict == "out_of_scope":
            response_text = scope_reply or _OUT_OF_SCOPE_FALLBACK
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
                response_text,
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "DIRECT",
                "final_answer":         response_text,
            }

        # Step B: reasoning call — sees session context + history, NOT data_context.
        session_context_block = _format_session_context_block(get_session_context(session_id))
        history_block         = _format_history_block(view["conversation_history"])

        user_message = _REASONING_USER_TEMPLATE.format(
            session_context_block=session_context_block,
            history_block=history_block,
            original_query=view["original_query"],
        )

        llm      = get_llm("chat")
        capture_prompt(
            session_id, 0,
            "chat", "reasoning",
            _REASONING_SYSTEM_PROMPT, user_message,
        )
        response = await llm.ainvoke([
            {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ])

        try:
            data = parse_llm_json(response.content)
            payload = _normalize_reasoning(data, view["original_query"])
        except ValueError as exc:
            logger.warning("[%s] reasoning parse failed: %s", session_id, exc)
            payload = _fallback_reasoning(view["original_query"], raw=str(response.content))

        # Ambient: apply session_context_update on every path, independent of decision.
        context_update = payload.get("session_context_update") or {}
        if context_update:
            update_session_context(session_id, context_update)

        chat_reasoning_json = _serialize_reasoning(payload)
        decision = payload["decision"]

        logger.debug(
            "[%s] Chat reasoning — decision=%s update_keys=%s",
            session_id, decision, list(context_update.keys()),
        )

        if decision == "PLAN":
            rewritten = payload["rewritten_query"] or view["original_query"]
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "PLAN",
                "rewritten_query":      rewritten,
                "chat_reasoning":       chat_reasoning_json,
            }

        if decision == "CLARIFY":
            response_text = payload["clarifying_question"] or "Could you rephrase your question?"
        else:  # DIRECT
            response_text = payload["direct_response"] or "Hello! Ask me anything about your company data."

        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            response_text,
        )
        return {
            "conversation_history": updated_history,
            "chat_intent":          decision,
            "final_answer":         response_text,
            "chat_reasoning":       chat_reasoning_json,
        }

    # ── Path 3: FORMAT AND DELIVER ────────────────────────────────
    audit = state.get("audit_result") or {}
    if audit.get("verdict") != "PASS":
        logger.warning(
            "chat_node reached delivery with final_answer set but audit verdict is %r — "
            "treating as exhaustion",
            audit.get("verdict"),
        )
        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            _FAILURE_MESSAGE,
        )
        return {
            "conversation_history": updated_history,
            "final_answer":         _FAILURE_MESSAGE,
        }

    # Feed the formatter the self-contained query (rewritten_query) plus a small
    # window of recent history. Without this, terse follow-ups reach the
    # formatter as raw fragments and the LLM may invert polarity trying to
    # reconcile the fragment with a concrete answer.
    resolved_query = view["rewritten_query"] or view["original_query"]
    history_block  = _format_recent_for_deliver(view["conversation_history"], last_n=2)

    user_message = _DELIVER_USER_TEMPLATE.format(
        history_block=history_block,
        resolved_query=resolved_query,
        final_answer=final_answer,
    )

    llm      = get_llm("chat")
    capture_prompt(
        session_id, 0,
        "chat", "formatter",
        _DELIVER_SYSTEM_PROMPT, user_message,
    )
    response = await llm.ainvoke([
        {"role": "system", "content": _DELIVER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    formatted_answer = response.content.strip()

    # Factual-preservation safety check — reject formatter polarity inversions.
    if _contains_negation(formatted_answer) and not _contains_negation(final_answer):
        logger.warning(
            "[%s] Formatter introduced negation absent from audited draft — "
            "rejecting formatted output and delivering the audited draft verbatim. "
            "Draft: %r Formatted: %r",
            session_id,
            final_answer,
            formatted_answer,
        )
        formatted_answer = final_answer

    updated_history = _append_messages(
        view["conversation_history"],
        view["original_query"],
        formatted_answer,
    )

    return {
        "conversation_history": updated_history,
        "final_answer":         formatted_answer,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.chat
# ---------------------------------------------------------------------------

def test_chat():
    from unittest.mock import AsyncMock, MagicMock, patch

    from core.scope_result import clear_scope_result, get_scope_result
    from core.session_context import clear_session_context
    from tests.fixtures import CHAT_AGENT_STATE

    patch_target        = f"{__name__}.get_llm"
    data_ctx_patch      = f"{__name__}.get_data_context"
    fake_data_context   = "The system contains the following data:\n- Example Source (record): description"

    # ── Shared minimal state skeleton ────────────────────────────
    base_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "chat_intent":          "",
        "rewritten_query":      "",
        "chat_reasoning":       "",
        "plan":                 [],
        "manifest_context":     "",
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

    def _reasoning_payload(
        decision,
        rewritten_query="",
        direct_response="",
        clarifying_question="",
        session_context_update=None,
        user_intent="",
    ):
        import json as _json
        return _json.dumps({
            "reasoning": {
                "user_intent":        user_intent or f"user requested {decision.lower()}",
                "decision_rationale": f"selected {decision}",
            },
            "decision":               decision,
            "rewritten_query":        rewritten_query,
            "clarifying_question":    clarifying_question,
            "direct_response":        direct_response,
            "session_context_update": session_context_update or {},
        })

    def _scope_payload(scope: str, response: str = "") -> str:
        import json as _json
        return _json.dumps({"scope": scope, "response": response})

    def _mock_llm_from_content(content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=resp)
        return llm

    # In-scope scope reply, caller supplies the reasoning JSON.
    # Returns (selector_fn, scope_llm_mock, chat_llm_mock) for fine-grained assertions.
    def _mock_llms(reasoning_json: str, scope_json: str | None = None):
        scope_llm = _mock_llm_from_content(
            scope_json or _scope_payload("in_scope", "")
        )
        chat_llm = _mock_llm_from_content(reasoning_json)

        def selector(agent_name):
            return scope_llm if agent_name == "chat_scope" else chat_llm

        return selector, scope_llm, chat_llm

    # Every reasoning-path test runs with data_context mocked — otherwise the
    # module tries to load the real manifest on disk.
    def _run(state: dict, selector, sid: str = "test-session-001") -> dict:
        state = {**state, "session_id": sid}
        with patch(data_ctx_patch, return_value=fake_data_context), \
             patch(patch_target, side_effect=selector):
            return asyncio.run(chat_node(state))

    # Convenience: run with a single-mock selector (same LLM for every agent).
    # Used by paths that never hit the scope call (exhaustion, delivery, fast greet).
    def _run_single(state: dict, llm, sid: str = "test-session-001") -> dict:
        def selector(_agent_name):
            return llm
        return _run(state, selector, sid)

    # Ensure a clean session_context + scope_result between tests
    def _reset_sid(sid: str = "test-session-001") -> None:
        clear_session_context(sid)
        clear_scope_result(sid)

    # ── Test 1: chat_view exposes only the permitted fields ───────
    view = chat_view(CHAT_AGENT_STATE)
    assert set(view.keys()) == {
        "original_query", "conversation_history", "final_answer", "final_sources",
        "chat_intent", "rewritten_query",
    }
    for forbidden in ("plan", "task_results", "audit_result", "retry_count",
                      "retry_notes", "synthesizer_output", "chat_reasoning"):
        assert forbidden not in view, f"chat_view leaked {forbidden}"
    print("PASS: chat_view exposes only permitted fields; hides chat_reasoning from its own prompt")

    # ── Test 2: DIRECT — reasoning call answers; both LLMs invoked ─
    _reset_sid()
    direct_response = "Nice to hear from you. Ask me anything about your data."
    direct_json = _reasoning_payload("DIRECT", direct_response=direct_response)

    direct_state = {**base_state, "original_query": "I'm just saying hi from Engineering"}
    selector, scope_llm, chat_llm = _mock_llms(direct_json)
    result_direct = _run(direct_state, selector)

    assert result_direct["chat_intent"]   == "DIRECT"
    assert result_direct["final_answer"]  == direct_response
    assert "chat_reasoning" in result_direct, "DIRECT from reasoning must write chat_reasoning"
    assert "rewritten_query" not in result_direct, "DIRECT must not write rewritten_query"
    scope_llm.ainvoke.assert_awaited_once()
    chat_llm.ainvoke.assert_awaited_once()
    # Scope call persists its verdict in the scope_result store for the trace.
    sr = get_scope_result("test-session-001")
    assert sr == {"scope": "in_scope", "response": ""}, \
        f"scope_result must record in_scope for this path; got {sr!r}"
    hist = result_direct["conversation_history"]
    assert hist[-1] == {"role": "assistant", "content": direct_response}
    print("PASS: DIRECT from reasoning writes final_answer + chat_reasoning; scope_result records in_scope verdict")

    # ── Test 3: CLARIFY ───────────────────────────────────────────
    _reset_sid()
    clar_question = "Which Noa do you mean — Noa Levi or Noa Levy?"
    clar_json = _reasoning_payload(
        "CLARIFY",
        clarifying_question=clar_question,
    )
    selector, _, _ = _mock_llms(clar_json)
    result_clar = _run({**base_state, "original_query": "Can Noa fly Business Class?"}, selector)
    assert result_clar["chat_intent"]  == "CLARIFY"
    assert result_clar["final_answer"] == clar_question
    assert "chat_reasoning" in result_clar
    print("PASS: CLARIFY returns clarifying_question as final_answer")

    # ── Test 4: PLAN — sets rewritten_query, no final_answer ──────
    _reset_sid()
    rewritten = "What is Dan Cohen's flight class entitlement?"
    plan_json = _reasoning_payload(
        "PLAN",
        rewritten_query=rewritten,
    )
    plan_state = {
        **base_state,
        "original_query": "What about Dan?",
        "conversation_history": [
            {"role": "user",      "content": "Can Noa fly Business Class?"},
            {"role": "assistant", "content": "Yes, Noa holds clearance level A."},
        ],
    }
    selector, _, _ = _mock_llms(plan_json)
    result_plan = _run(plan_state, selector)
    assert result_plan["chat_intent"]     == "PLAN"
    assert result_plan["rewritten_query"] == rewritten
    assert "chat_reasoning" in result_plan
    assert "final_answer" not in result_plan, "PLAN must not set final_answer"
    print("PASS: PLAN writes rewritten_query + chat_reasoning, leaves final_answer empty")

    # ── Test 5: reasoning parse failure falls back to PLAN ────────
    _reset_sid()
    selector, _, _ = _mock_llms("not valid json at all")
    result_bad = _run(base_state, selector)
    assert result_bad["chat_intent"]     == "PLAN"
    assert result_bad["rewritten_query"] == base_state["original_query"]
    assert "chat_reasoning" in result_bad
    print("PASS: reasoning parse failure falls back to PLAN with original_query")

    # ── Test 6: session_context_update persists across turns ──────
    _reset_sid()
    id_json = _reasoning_payload(
        "DIRECT",
        direct_response="Nice to meet you, Dan. Ask me anything about your data.",
        session_context_update={"user_name": "Dan", "department": "Engineering"},
    )
    selector, _, _ = _mock_llms(id_json)
    _run({**base_state, "original_query": "my name is Dan, I'm in Engineering"}, selector)
    ctx_after = get_session_context("test-session-001")
    assert ctx_after.get("user_name")  == "Dan"
    assert ctx_after.get("department") == "Engineering"
    print("PASS: identity-only message persists user_name + department in session_context")

    # ── Test 6b: session_context correction overwrites on next turn ──
    correction_json = _reasoning_payload(
        "DIRECT",
        direct_response="Got it, Noa.",
        session_context_update={"user_name": "Noa"},
    )
    selector, _, _ = _mock_llms(correction_json)
    _run({**base_state, "original_query": "actually I'm Noa, not Dan"}, selector)
    ctx_corrected = get_session_context("test-session-001")
    assert ctx_corrected.get("user_name")  == "Noa",         "correction must overwrite user_name"
    assert ctx_corrected.get("department") == "Engineering", "unchanged keys must survive"
    print("PASS: session_context correction overwrites the collision key")

    # ── Test 7a: retry exhaustion, no synthesizer → failure message, no LLM ──
    _reset_sid()
    mock_llm_exhaust = MagicMock(); mock_llm_exhaust.ainvoke = AsyncMock()
    exh_state = {**base_state, "retry_count": _MAX_ATTEMPTS, "synthesizer_output": ""}
    result_exh = _run_single(exh_state, mock_llm_exhaust)
    mock_llm_exhaust.ainvoke.assert_not_called()
    assert result_exh["final_answer"] == _FAILURE_MESSAGE
    # Exhaustion path does not run the scope call in this chat_node invocation.
    # (The entry invocation from earlier in the /chat run might have — that
    # value would remain in the store. Here we asserted _reset_sid first, so
    # the store starts empty and remains empty.)
    assert get_scope_result("test-session-001") is None, \
        "exhaustion path must not write scope_result"
    print("PASS: exhaustion without synthesizer_output returns _FAILURE_MESSAGE, no LLM call, scope_result untouched")

    # ── Test 7b: retry exhaustion with synthesizer_output → LLM composes partial ──
    _reset_sid()
    partial = "I could not fully verify, but records suggest Noa has clearance A."
    partial_resp = MagicMock(); partial_resp.content = f"  {partial}  "
    partial_llm  = MagicMock(); partial_llm.ainvoke = AsyncMock(return_value=partial_resp)
    exh_state2 = {
        **base_state,
        "retry_count":        _MAX_ATTEMPTS,
        "synthesizer_output": "Noa holds clearance level A per the employees table.",
    }
    result_exh2 = _run_single(exh_state2, partial_llm)
    assert result_exh2["final_answer"] == partial
    print("PASS: exhaustion with synthesizer_output calls LLM and returns partial answer")

    # ── Test 8: fast greeting — no LLM call at all ────────────────
    _reset_sid()
    fast_llm = MagicMock(); fast_llm.ainvoke = AsyncMock()
    for greeting in ("hi", "hello", "Hey!", "thanks", "bye"):
        result_fast = _run_single({**base_state, "original_query": greeting}, fast_llm)
        assert result_fast["chat_intent"]  == "DIRECT"
        assert result_fast["final_answer"]
        assert "chat_reasoning" not in result_fast, "fast path should not emit reasoning"
        # Fast greeting short-circuits before the scope call — the scope_result
        # store must stay empty so the trace shows scope_result=None downstream.
        assert get_scope_result("test-session-001") is None, \
            "fast greeting must not write scope_result"
    fast_llm.ainvoke.assert_not_called()
    print("PASS: fast greeting path returns DIRECT without any LLM call and leaves scope_result unset")

    # ── Test 9: delivery path — audit PASS — uses rewritten_query + recent history ──
    _reset_sid()
    audited_draft = "Yes. Noa holds clearance level A, which entitles her to Business Class on flights over 4 hours."
    formatted     = "Noa has clearance level A and is entitled to Business Class on flights over 4 hours."
    deliver_resp  = MagicMock(); deliver_resp.content = f"  {formatted}  "
    captured: list = []
    async def capture(messages):
        captured.extend(messages)
        return deliver_resp
    deliver_llm = MagicMock(); deliver_llm.ainvoke = capture

    deliver_state = {
        **base_state,
        "original_query":   "Can she fly Business?",
        "rewritten_query":  "Can Noa fly Business Class?",
        "conversation_history": [
            {"role": "user",      "content": "Who is Noa?"},
            {"role": "assistant", "content": "Noa is a senior engineer."},
        ],
        "final_answer":    audited_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
    }
    result_deliver = _run_single(deliver_state, deliver_llm)
    assert set(result_deliver.keys()) == {"conversation_history", "final_answer"}
    assert result_deliver["final_answer"] == formatted
    system_msg = next(m["content"] for m in captured if m["role"] == "system")
    user_msg   = next(m["content"] for m in captured if m["role"] == "user")
    assert "copyeditor" in system_msg.lower()
    assert "never invert polarity" in system_msg.lower()
    # Formatter must use rewritten_query, not original_query
    assert "Can Noa fly Business Class?" in user_msg
    assert "RECENT CONVERSATION" in user_msg
    # Delivery path does not fire the scope call. The store was reset above,
    # so it must still be empty after this invocation.
    assert get_scope_result("test-session-001") is None, \
        "delivery path must not write scope_result"
    print("PASS: delivery uses rewritten_query + recent history; copyeditor constraints present; scope_result untouched")

    # ── Test 10: formatter polarity safety check ─────────────────
    _reset_sid()
    positive_draft = "Noa Levi's salary is in the $85,000 - $110,000 range based on the employees table."
    inverted       = "There is no employee named Noa Levi in the system."
    inv_resp = MagicMock(); inv_resp.content = inverted
    inv_llm  = MagicMock(); inv_llm.ainvoke = AsyncMock(return_value=inv_resp)
    inv_state = {
        **base_state,
        "final_answer":    positive_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is Noa Levi's salary?",
    }
    result_inv = _run_single(inv_state, inv_llm)
    assert result_inv["final_answer"] == positive_draft, \
        "Safety check must revert to audited draft on polarity inversion"
    print("PASS: formatter polarity inversion triggers safety check")

    # Legitimate negation (draft already negates) → formatted output accepted
    negative_draft     = "No employee named Noa Levie was found in the records."
    negative_formatted = "There is no employee named Noa Levie."
    neg_resp = MagicMock(); neg_resp.content = negative_formatted
    neg_llm  = MagicMock(); neg_llm.ainvoke = AsyncMock(return_value=neg_resp)
    neg_state = {
        **base_state,
        "final_answer":    negative_draft,
        "audit_result":    {"verdict": "PASS", "notes": ""},
        "rewritten_query": "What is Noa Levie's salary?",
    }
    result_neg = _run_single(neg_state, neg_llm)
    assert result_neg["final_answer"] == negative_formatted
    print("PASS: legitimate negation in audited draft delivers formatted output")

    # ── Test 11: delivery rejects when audit verdict != PASS ─────
    _reset_sid()
    no_llm = MagicMock(); no_llm.ainvoke = AsyncMock()
    bad_state = {
        **base_state,
        "final_answer":  "Some answer that should not be delivered.",
        "audit_result":  {"verdict": "FAIL", "notes": "Rejected."},
    }
    result_bad_audit = _run_single(bad_state, no_llm)
    no_llm.ainvoke.assert_not_called()
    assert result_bad_audit["final_answer"] == _FAILURE_MESSAGE
    print("PASS: delivery with non-PASS audit returns _FAILURE_MESSAGE, no LLM call")

    # ── Test 12: scope prompt carries DATA CONTEXT; reasoning prompt does NOT ──
    _reset_sid()
    seen_scope_msgs: list = []
    seen_chat_msgs:  list = []

    scope_resp = MagicMock(); scope_resp.content = _scope_payload("in_scope", "")
    chat_resp  = MagicMock(); chat_resp.content  = _reasoning_payload("PLAN", rewritten_query="What is X?")

    async def capture_scope(messages):
        seen_scope_msgs.extend(messages)
        return scope_resp
    async def capture_chat(messages):
        seen_chat_msgs.extend(messages)
        return chat_resp

    scope_llm_capture = MagicMock(); scope_llm_capture.ainvoke = capture_scope
    chat_llm_capture  = MagicMock(); chat_llm_capture.ainvoke  = capture_chat

    def selector_capture(agent_name):
        return scope_llm_capture if agent_name == "chat_scope" else chat_llm_capture

    # Pre-populate session context
    update_session_context("test-session-001", {"user_name": "Taylor"})

    seen_state = {
        **base_state,
        "original_query": "what is X?",
        "conversation_history": [{"role": "user", "content": "earlier message"}],
    }
    _run(seen_state, selector_capture)

    # Scope call should carry data_context and user message
    scope_system = next(m["content"] for m in seen_scope_msgs if m["role"] == "system")
    scope_user   = next(m["content"] for m in seen_scope_msgs if m["role"] == "user")
    assert "scope filter"    in scope_system.lower(), "scope system prompt missing 'scope filter'"
    assert "DATA CONTEXT:"   in scope_user,           "scope user prompt must include DATA CONTEXT block"
    assert fake_data_context.splitlines()[0] in scope_user, \
        "scope user prompt must carry the data_context paragraph"
    assert "USER MESSAGE:"   in scope_user,           "scope user prompt must include USER MESSAGE block"

    # Reasoning call should carry session_context + history, but NOT data_context
    chat_system = next(m["content"] for m in seen_chat_msgs if m["role"] == "system")
    chat_user   = next(m["content"] for m in seen_chat_msgs if m["role"] == "user")
    assert "DATA CONTEXT"         not in chat_user, \
        "reasoning prompt must NOT include DATA CONTEXT block"
    assert "SESSION CONTEXT"      in chat_user, "reasoning prompt must include SESSION CONTEXT block"
    assert "Taylor"               in chat_user, "SESSION CONTEXT must carry prior user_name"
    assert "CONVERSATION HISTORY" in chat_user, "reasoning prompt must include history"
    assert "CURRENT USER MESSAGE" in chat_user
    # System prompt must not carry removed tokens.
    for removed in ("gaps", "DATA CONTEXT", '"gaps"'):
        assert removed not in chat_system, \
            f"reasoning system prompt still references removed token {removed!r}"
    # New step labels must be present.
    for present in ("STEP 1", "STEP 2", "STEP 3"):
        assert present in chat_system, \
            f"reasoning system prompt missing step label {present!r}"
    print("PASS: scope prompt has DATA CONTEXT; reasoning prompt has session+history but no data_context")

    # ── Test 13: history truncation — last MAX_REASONING_HISTORY entries only ──
    _reset_sid()
    big_history = []
    for i in range(20):
        big_history.append({"role": "user",      "content": f"user-msg-{i}"})
        big_history.append({"role": "assistant", "content": f"asst-msg-{i}"})  # 40 messages total

    selector, _, _ = _mock_llms(_reasoning_payload("PLAN", rewritten_query="What?"))
    # Wrap selector to capture the reasoning-call user prompt
    captured_prompts: list = []
    def selector_cap(agent_name):
        llm = selector(agent_name)
        if agent_name != "chat_scope":
            original = llm.ainvoke
            async def cap(messages):
                captured_prompts.extend(messages)
                return await original(messages)
            llm.ainvoke = cap
        return llm

    _run({**base_state, "conversation_history": big_history, "original_query": "What?"}, selector_cap)
    user_prompt = next(m["content"] for m in captured_prompts if m["role"] == "user")
    # MAX_REASONING_HISTORY = 12 → last 12 messages. We had 40 → last 12 are
    # user-msg-14, asst-msg-14, ..., asst-msg-19. user-msg-0 MUST NOT appear,
    # asst-msg-19 MUST.
    assert "user-msg-0"   not in user_prompt
    assert "user-msg-13"  not in user_prompt
    assert "user-msg-14"  in user_prompt
    assert "asst-msg-19"  in user_prompt
    print(f"PASS: reasoning call truncates history to last {MAX_REASONING_HISTORY} messages")

    # ── Test 14: post-clarification "yes" → PLAN rewrites full intent ──
    _reset_sid()
    yes_json = _reasoning_payload(
        "PLAN",
        rewritten_query="What is the salary range for Noa Levy?",
    )
    yes_state = {
        **base_state,
        "original_query": "yes",
        "conversation_history": [
            {"role": "user",      "content": "What is the salary range for Noa Levi?"},
            {"role": "assistant", "content": "I could not find an employee named Noa Levi. Did you mean a similar name?"},
            {"role": "user",      "content": "maybe noa levy"},
            {"role": "assistant", "content": "Did you mean Noa Levy (not Levi)?"},
        ],
    }
    selector, _, _ = _mock_llms(yes_json)
    result_yes = _run(yes_state, selector)
    assert result_yes["chat_intent"] == "PLAN"
    assert "Noa Levy" in result_yes["rewritten_query"]
    assert "salary"   in result_yes["rewritten_query"].lower()
    print("PASS: post-clarification 'yes' routes as PLAN with reconstructed intent")

    # ── Test 15: identity + question in one message → PLAN + context update ──
    _reset_sid()
    combo_json = _reasoning_payload(
        "PLAN",
        rewritten_query="What is Dan's (Engineering) salary?",
        session_context_update={"user_name": "Dan", "department": "Engineering"},
    )
    combo_state = {**base_state, "original_query": "I'm Dan from Engineering, what's my salary?"}
    selector, _, _ = _mock_llms(combo_json)
    result_combo = _run(combo_state, selector)
    assert result_combo["chat_intent"]     == "PLAN"
    assert result_combo["rewritten_query"] == "What is Dan's (Engineering) salary?"
    ctx = get_session_context("test-session-001")
    assert ctx["user_name"]  == "Dan"
    assert ctx["department"] == "Engineering"
    print("PASS: identity + question emits PLAN + session_context_update together")

    # ── Test 16: delivery preserves chat_reasoning set on prior turn ──
    # Delivery path returns only {conversation_history, final_answer}, not
    # chat_reasoning. chat_reasoning is a snapshot of the entry-turn decision
    # and must not be overwritten by the delivery path on a retry loop.
    _reset_sid()
    plain_draft     = "Dan Cohen earns $95,000 annually based on the employees table."
    plain_formatted = "Dan Cohen makes $95,000 per year."
    plain_resp = MagicMock(); plain_resp.content = plain_formatted
    plain_llm  = MagicMock(); plain_llm.ainvoke = AsyncMock(return_value=plain_resp)
    plain_state = {
        **base_state,
        "chat_reasoning":  '{"previous": "reasoning"}',
        "final_answer":    plain_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is Dan Cohen's salary?",
    }
    result_plain = _run_single(plain_state, plain_llm)
    assert "chat_reasoning" not in result_plain, \
        "delivery path must not overwrite chat_reasoning"
    print("PASS: delivery path returns only {conversation_history, final_answer}")

    # ── Test 17: _normalize_reasoning / _fallback_reasoning emit trimmed schema ──
    _reset_sid()
    normalized = _normalize_reasoning({
        "reasoning": {
            "user_intent":        "look up a fact",
            "decision_rationale": "clear lookup",
        },
        "decision":               "PLAN",
        "rewritten_query":        "What is Dan Cohen's salary?",
        "clarifying_question":    "",
        "direct_response":        "",
        "session_context_update": {},
    }, "original")
    assert set(normalized["reasoning"].keys()) == {"user_intent", "decision_rationale"}, \
        f"normalized reasoning has unexpected keys: {normalized['reasoning'].keys()}"

    fb = _fallback_reasoning("orig q")
    assert set(fb["reasoning"].keys()) == {"user_intent", "decision_rationale"}, \
        f"fallback reasoning has unexpected keys: {fb['reasoning'].keys()}"
    assert fb["decision"] == "PLAN"
    assert fb["rewritten_query"] == "orig q"
    print("PASS: reasoning normalize/fallback emit only {user_intent, decision_rationale} — no gaps")

    # Defensive: inputs that still carry a 'gaps' key (from stale LLM output or
    # a serialized payload from before the schema change) must be accepted and
    # normalized to the trimmed shape without exploding.
    legacy = _normalize_reasoning({
        "reasoning": {
            "user_intent":        "legacy caller still emits gaps",
            "gaps":               ["something"],
            "decision_rationale": "ok",
        },
        "decision":               "PLAN",
        "rewritten_query":        "What?",
        "clarifying_question":    "",
        "direct_response":        "",
        "session_context_update": {},
    }, "original")
    assert "gaps" not in legacy["reasoning"], \
        "normalize must drop legacy gaps field from inbound reasoning"
    print("PASS: _normalize_reasoning drops a legacy 'gaps' key if present")

    # ── Test 18: PLAN rewritten_query is natural language, no procedural words ──
    _reset_sid()
    banned_tokens = (
        "source", "table", "record", "document", "file",
        "retrieve", "provide", "lookup", "if available", "otherwise",
    )
    clean_queries_seen: list[str] = []
    for q in (
        "What is Dan Cohen's flight class entitlement?",
        "What is the salary range for Noa Levy?",
        "What is Dan's (Engineering) salary?",
        "What is Dan Cohen's salary?",
        "What is X?",
    ):
        clean_queries_seen.append(q)

    for q in clean_queries_seen:
        low = q.lower()
        for banned in banned_tokens:
            assert banned not in low, (
                f"test-authored rewritten_query {q!r} contains banned token "
                f"{banned!r} — update the fixture to a natural-language question"
            )
    print("PASS: all test-authored PLAN rewrites avoid source/procedural language")

    # ── Test 19: self-reference resolved from SESSION CONTEXT → PLAN ──
    _reset_sid()
    update_session_context("test-session-001", {"user_name": "Dan"})
    self_ref_rewrite = "What is Dan's salary range?"
    self_ref_json = _reasoning_payload("PLAN", rewritten_query=self_ref_rewrite)
    self_ref_state = {**base_state, "original_query": "what's my salary range?"}
    selector, _, _ = _mock_llms(self_ref_json)
    result_self_ref = _run(self_ref_state, selector)
    assert result_self_ref["chat_intent"] == "PLAN"
    assert "Dan" in result_self_ref["rewritten_query"], \
        "PLAN rewrite must resolve 'my' using SESSION CONTEXT user_name"
    print("PASS: self-reference resolvable from SESSION CONTEXT routes as PLAN with resolved name")

    # ── Test 20: self-reference with empty SESSION CONTEXT → CLARIFY ──
    _reset_sid()
    unresolved_question = "What is your name?"
    unresolved_json = _reasoning_payload(
        "CLARIFY",
        clarifying_question=unresolved_question,
    )
    unresolved_state = {**base_state, "original_query": "what's my salary range?"}
    selector, _, _ = _mock_llms(unresolved_json)
    result_unresolved = _run(unresolved_state, selector)
    assert result_unresolved["chat_intent"]  == "CLARIFY"
    assert result_unresolved["final_answer"] == unresolved_question
    print("PASS: self-reference with empty SESSION CONTEXT routes as CLARIFY asking for identity")

    # ── Test 21: multi-part question → PLAN preserves every part ──
    _reset_sid()
    multi_rewrite = "What is Dan Cohen's role and who is his manager?"
    multi_json = _reasoning_payload("PLAN", rewritten_query=multi_rewrite)
    multi_state = {
        **base_state,
        "original_query": "What is Dan Cohen's role and who does he report to?",
    }
    selector, _, _ = _mock_llms(multi_json)
    result_multi = _run(multi_state, selector)
    assert result_multi["chat_intent"] == "PLAN"
    assert "role"    in result_multi["rewritten_query"].lower()
    assert "manager" in result_multi["rewritten_query"].lower()
    print("PASS: multi-part question routes as PLAN preserving every part")

    # ── Test 22: out_of_scope short-circuits — reasoning LLM never called ──
    _reset_sid()
    oos_reply = "I only answer questions about the ingested company data."
    scope_llm_oos = _mock_llm_from_content(_scope_payload("out_of_scope", oos_reply))
    reasoning_llm_oos = MagicMock(); reasoning_llm_oos.ainvoke = AsyncMock()

    def selector_oos(agent_name):
        return scope_llm_oos if agent_name == "chat_scope" else reasoning_llm_oos

    oos_state = {**base_state, "original_query": "Tell me a joke about cats."}
    result_oos = _run(oos_state, selector_oos)
    assert result_oos["chat_intent"]  == "DIRECT"
    assert result_oos["final_answer"] == oos_reply
    assert "chat_reasoning" not in result_oos, \
        "out_of_scope short-circuit must not emit chat_reasoning"
    scope_llm_oos.ainvoke.assert_awaited_once()
    reasoning_llm_oos.ainvoke.assert_not_called()
    # Scope call fired and recorded verdict + friendly reply in scope_result.
    sr_oos = get_scope_result("test-session-001")
    assert sr_oos == {"scope": "out_of_scope", "response": oos_reply}, \
        f"scope_result must record out_of_scope verdict + reply; got {sr_oos!r}"
    # History gets user + assistant appended
    hist = result_oos["conversation_history"]
    assert hist[-2] == {"role": "user", "content": "Tell me a joke about cats."}
    assert hist[-1] == {"role": "assistant", "content": oos_reply}
    print("PASS: out_of_scope short-circuits with friendly reply; scope_result captures verdict + response")

    # ── Test 23: out_of_scope with empty 'response' → fallback string ──
    _reset_sid()
    scope_llm_empty = _mock_llm_from_content(_scope_payload("out_of_scope", ""))
    reasoning_llm_empty = MagicMock(); reasoning_llm_empty.ainvoke = AsyncMock()

    def selector_empty(agent_name):
        return scope_llm_empty if agent_name == "chat_scope" else reasoning_llm_empty

    empty_state = {**base_state, "original_query": "chit chat"}
    result_empty = _run(empty_state, selector_empty)
    assert result_empty["chat_intent"]  == "DIRECT"
    assert result_empty["final_answer"] == _OUT_OF_SCOPE_FALLBACK
    reasoning_llm_empty.ainvoke.assert_not_called()
    print("PASS: out_of_scope with empty response falls back to _OUT_OF_SCOPE_FALLBACK")

    # ── Test 24: scope parse failure defaults to in_scope → reasoning call runs ──
    _reset_sid()
    bad_scope_llm = _mock_llm_from_content("not json at all")
    good_reasoning_json = _reasoning_payload("PLAN", rewritten_query="What is the policy?")
    good_reasoning_llm = _mock_llm_from_content(good_reasoning_json)

    def selector_bad_scope(agent_name):
        return bad_scope_llm if agent_name == "chat_scope" else good_reasoning_llm

    bad_scope_state = {**base_state, "original_query": "What is the policy?"}
    result_bad_scope = _run(bad_scope_state, selector_bad_scope)
    assert result_bad_scope["chat_intent"] == "PLAN"
    good_reasoning_llm.ainvoke.assert_awaited_once()
    print("PASS: scope parse failure defaults to in_scope; reasoning call still runs")

    _reset_sid()
    print("\nPASS: all chat tests passed")


if __name__ == "__main__":
    test_chat()

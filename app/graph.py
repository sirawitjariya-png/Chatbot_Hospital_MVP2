"""Skill-based LangGraph workflow — no RAG, no vector DB.
"""Skill-based LangGraph workflow — no RAG, no vector DB.

    START → classify
        ├─ off_topic  → fixed reply → END
        ├─ smalltalk  → LLM reply   → END
        └─ treatment / general
               └─ load_files → check_info
                      ├─ no data  → fixed reply → END
                      └─ has data → format_answer → END

2 LLM calls per normal question (classify + answer).
1 LLM call for smalltalk. 0 for off_topic / no_data.

AI session gate (Step 5):
  Activate : user sends "สอบถามเอไอ" or "Ask AI"  (case-insensitive)
  Deactivate: user sends "ปิดเอไอ"  or "Close AI" (case-insensitive)
  While inactive the graph is never invoked — no LLM API calls are made.
  Every answer appends a one-line reminder of how to close the AI.
"""
import logging
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import llm, smalltalk as _smalltalk_fn, off_topic, no_data, _is_thai
from .skills import classify_question, load_files, format_answer
from .tracer import write_trace

log = logging.getLogger(__name__)

_MAX_HISTORY = 10
_user_history: dict[str, list] = {}

# ---------------------------------------------------------------------------
# AI session state
# ---------------------------------------------------------------------------

_ai_active: dict[str, bool] = {}

_AI_TRIGGER  = {"สอบถามเอไอ", "ask ai"}
_AI_CLOSE    = {"ปิดเอไอ", "close ai"}

_ACTIVATE_TH = (
    "เปิดใช้งานเอไอเรียบร้อยแล้วค่ะ 😊 ยินดีตอบคำถามทุกเรื่องเกี่ยวกับทันตกรรมและบริการของศูนย์ฯ ค่ะ\n\n"
    "พิมพ์ \"ปิดเอไอ\" หรือ \"Close AI\" หากต้องการหยุดการใช้งานเอไอ"
)
_ACTIVATE_EN = (
    "AI assistant is now active! 😊 Feel free to ask me anything about dental care or our services.\n\n"
    "Type \"ปิดเอไอ\" or \"Close AI\" to stop the AI assistant."
)

_DEACTIVATE_TH = "ปิดการใช้งานเอไอเรียบร้อยแล้วค่ะ ขอบคุณที่ใช้บริการนะคะ 🙏"
_DEACTIVATE_EN = "AI assistant has been closed. Thank you for using our service! 🙏"

_FOOTER_TH = "\n\n---\nพิมพ์ \"ปิดเอไอ\" หรือ \"Close AI\" หากต้องการหยุดการใช้งานเอไอ"
_FOOTER_EN  = "\n\n---\nType \"ปิดเอไอ\" or \"Close AI\" if you want to stop the AI assistant."

# Patterns that look like a close attempt but aren't exact
# EN: message starts with "close" / TH: message starts with "ปิด"
_CLOSE_LIKE_EN = re.compile(r"^close\b", re.IGNORECASE)
_CLOSE_LIKE_TH = re.compile(r"^ปิด")

_CLOSE_CONFIRM_TH = (
    "ต้องการปิดการใช้งานเอไอใช่ไหมคะ?\n"
    "กรุณาพิมพ์ \"ปิดเอไอ\" หรือ \"Close AI\" ให้ถูกต้องเพื่อยืนยันนะคะ"
)
_CLOSE_CONFIRM_EN = (
    "Did you want to stop the AI assistant?\n"
    "Please type \"ปิดเอไอ\" or \"Close AI\" exactly to confirm."
)


def _looks_like_close(normalized: str, original: str) -> bool:
    """Return True when the message resembles a close command but is not an exact match."""
    return bool(
        _CLOSE_LIKE_EN.match(normalized) or
        _CLOSE_LIKE_TH.match(original.strip())
    )


class State(TypedDict, total=False):
    question: str
    user_id: str
    route: str
    files: list
    content: dict
    has_data: bool
    files: list
    content: dict
    has_data: bool
    answer: str
    history: list
    trace: Annotated[list, operator.add]


# ---- nodes ------------------------------------------------------------------

def _classify_node(state: State) -> dict:
    try:
        result = classify_question(state["question"], state.get("history", []), llm)
        return {
            "route": result["route"],
            "files": result["files"],
            "trace": [result["trace_entry"]],
        }
    except Exception as e:
        log.error("classify_node unexpected error: %s", e)
        return {"route": "general", "files": [], "trace": [{"node": "classify", "route": "general", "files": [], "error": str(e)}]}
# ---- nodes ------------------------------------------------------------------

def _classify_node(state: State) -> dict:
    try:
        result = classify_question(state["question"], state.get("history", []), llm)
        return {
            "route": result["route"],
            "files": result["files"],
            "trace": [result["trace_entry"]],
        }
    except Exception as e:
        log.error("classify_node unexpected error: %s", e)
        return {"route": "general", "files": [], "trace": [{"node": "classify", "route": "general", "files": [], "error": str(e)}]}


def _smalltalk_node(state: State) -> dict:
    try:
        answer = _smalltalk_fn(state["question"], state.get("history", []))
        return {"answer": answer, "trace": [{"node": "smalltalk"}]}
    except Exception as e:
        log.error("smalltalk_node unexpected error: %s", e)
        from .agents import _is_thai
        fallback = "สวัสดีค่ะ มีอะไรให้ช่วยบ้างคะ" if _is_thai(state.get("question", "")) else "Hello! How may I assist you?"
        return {"answer": fallback, "trace": [{"node": "smalltalk"}]}


def _off_topic_node(state: State) -> dict:
    return {"answer": off_topic(state.get("question", "")), "trace": [{"node": "off_topic"}]}


def _load_files_node(state: State) -> dict:
    try:
        result = load_files(state.get("files", []))
        return {
            "content": result["content"],
            "has_data": result["has_data"],
            "trace": [result["trace_entry"]],
        }
    except Exception as e:
        log.error("load_files_node unexpected error: %s", e)
        return {"content": {}, "has_data": False, "trace": [{"node": "read_files", "files": [], "has_data": False, "error": str(e)}]}


def _no_data_node(state: State) -> dict:
    return {"answer": no_data(state.get("question", "")), "trace": [{"node": "no_data"}]}


def _format_answer_node(state: State) -> dict:
    try:
        answer = format_answer(
            state["question"],
            state.get("content", {}),
            state.get("history", []),
            llm,
        )
        return {"answer": answer, "trace": [{"node": "answer", "draft": answer[:80]}]}
    except Exception as e:
        log.error("format_answer_node unexpected error: %s", e)
        return {"answer": no_data(state.get("question", "")), "trace": [{"node": "answer", "error": str(e)}]}
def _smalltalk_node(state: State) -> dict:
    try:
        answer = _smalltalk_fn(state["question"], state.get("history", []))
        return {"answer": answer, "trace": [{"node": "smalltalk"}]}
    except Exception as e:
        log.error("smalltalk_node unexpected error: %s", e)
        from .agents import _is_thai
        fallback = "สวัสดีค่ะ มีอะไรให้ช่วยบ้างคะ" if _is_thai(state.get("question", "")) else "Hello! How may I assist you?"
        return {"answer": fallback, "trace": [{"node": "smalltalk"}]}


def _off_topic_node(state: State) -> dict:
    return {"answer": off_topic(state.get("question", "")), "trace": [{"node": "off_topic"}]}


def _load_files_node(state: State) -> dict:
    try:
        result = load_files(state.get("files", []))
        return {
            "content": result["content"],
            "has_data": result["has_data"],
            "trace": [result["trace_entry"]],
        }
    except Exception as e:
        log.error("load_files_node unexpected error: %s", e)
        return {"content": {}, "has_data": False, "trace": [{"node": "read_files", "files": [], "has_data": False, "error": str(e)}]}


def _no_data_node(state: State) -> dict:
    return {"answer": no_data(state.get("question", "")), "trace": [{"node": "no_data"}]}


def _format_answer_node(state: State) -> dict:
    try:
        answer = format_answer(
            state["question"],
            state.get("content", {}),
            state.get("history", []),
            llm,
        )
        return {"answer": answer, "trace": [{"node": "answer", "draft": answer[:80]}]}
    except Exception as e:
        log.error("format_answer_node unexpected error: %s", e)
        return {"answer": no_data(state.get("question", "")), "trace": [{"node": "answer", "error": str(e)}]}


# ---- routing ----------------------------------------------------------------

def _route_after_classify(state: State) -> str:
    return state.get("route", "general")

def _route_after_classify(state: State) -> str:
    return state.get("route", "general")


def _route_after_load(state: State) -> str:
    return "ok" if state.get("has_data") else "no_data"
def _route_after_load(state: State) -> str:
    return "ok" if state.get("has_data") else "no_data"


# ---- graph ------------------------------------------------------------------

def _build_graph():
# ---- graph ------------------------------------------------------------------

def _build_graph():
    g = StateGraph(State)

    g.add_node("classify",      _classify_node)
    g.add_node("smalltalk",     _smalltalk_node)
    g.add_node("off_topic",     _off_topic_node)
    g.add_node("load_files",    _load_files_node)
    g.add_node("no_data",       _no_data_node)
    g.add_node("format_answer", _format_answer_node)

    g.add_edge(START, "classify")
    g.add_node("classify",      _classify_node)
    g.add_node("smalltalk",     _smalltalk_node)
    g.add_node("off_topic",     _off_topic_node)
    g.add_node("load_files",    _load_files_node)
    g.add_node("no_data",       _no_data_node)
    g.add_node("format_answer", _format_answer_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "off_topic": "off_topic",
            "smalltalk": "smalltalk",
            "treatment": "load_files",
            "general":   "load_files",
        },
    )
        "classify",
        _route_after_classify,
        {
            "off_topic": "off_topic",
            "smalltalk": "smalltalk",
            "treatment": "load_files",
            "general":   "load_files",
        },
    )
    g.add_conditional_edges(
        "load_files",
        _route_after_load,
        {"ok": "format_answer", "no_data": "no_data"},
    )

        "load_files",
        _route_after_load,
        {"ok": "format_answer", "no_data": "no_data"},
    )

    g.add_edge("smalltalk",     END)
    g.add_edge("off_topic",     END)
    g.add_edge("no_data",       END)
    g.add_edge("format_answer", END)
    g.add_edge("format_answer", END)

    return g.compile()


_graph = _build_graph()
_graph = _build_graph()

_FALLBACK_TH = (
    "ขออภัยค่ะ ระบบเกิดข้อผิดพลาดชั่วคราว "
    "กรุณาลองใหม่อีกครั้ง หรือติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ โดยตรงค่ะ"
)
_FALLBACK_EN = (
_FALLBACK_TH = (
    "ขออภัยค่ะ ระบบเกิดข้อผิดพลาดชั่วคราว "
    "กรุณาลองใหม่อีกครั้ง หรือติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ โดยตรงค่ะ"
)
_FALLBACK_EN = (
    "Sorry, a temporary error occurred. Please try again or contact "
    "Walailuk University Dentist Hospital in Bangkok directly."
    "Walailuk University Dentist Hospital in Bangkok directly."
)


def ask(question: str, user_id: str = "cli") -> str | None:
    normalized = question.strip().lower()
    thai = _is_thai(question)

    # --- activation trigger ---
    if normalized in _AI_TRIGGER:
        _ai_active[user_id] = True
        _user_history.pop(user_id, None)  # fresh history for each AI session
        log.info("AI session opened for user %s", user_id)
        return _ACTIVATE_TH if thai else _ACTIVATE_EN

    # --- deactivation trigger (exact match only) ---
    if normalized in _AI_CLOSE:
        _ai_active.pop(user_id, None)
        _user_history.pop(user_id, None)
        log.info("AI session closed for user %s", user_id)
        return _DEACTIVATE_TH if thai else _DEACTIVATE_EN

    # --- looks like a close attempt but not exact — ask for confirmation ---
    if _ai_active.get(user_id, False) and _looks_like_close(normalized, question):
        log.info("Close-like message from %s (not exact): %r", user_id, question)
        return _CLOSE_CONFIRM_TH if thai else _CLOSE_CONFIRM_EN

    # --- gate: AI must be active to proceed — return None so admin can reply manually ---
    if not _ai_active.get(user_id, False):
        return None

    # --- normal graph invocation ---
    history = _user_history.get(user_id, [])
    try:
        result = _graph.invoke({
            "question": question,
            "user_id":  user_id,
            "history":  history,
            "trace":    [],
            "user_id":  user_id,
            "history":  history,
            "trace":    [],
        })
        answer_text = result["answer"]
        # append stop-AI reminder to every answer
        answer_text += _FOOTER_TH if thai else _FOOTER_EN

        _user_history[user_id] = (history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer_text},
        ])[-_MAX_HISTORY:]
        try:
            write_trace(
                user_id=result.get("user_id", user_id),
                question=question,
                trace=result.get("trace", []),
                final_answer=answer_text,
            )
        except Exception as e:
            log.warning("write_trace failed: %s", e)
        return answer_text
    except Exception as e:
        log.error("Graph error for user %s: %s", user_id, e)
        fallback = _FALLBACK_TH if thai else _FALLBACK_EN
        return fallback + (_FOOTER_TH if thai else _FOOTER_EN)

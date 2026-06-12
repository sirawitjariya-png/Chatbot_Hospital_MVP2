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
"""
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import llm, smalltalk as _smalltalk_fn, off_topic, no_data
from .skills import classify_question, load_files, format_answer
from .tracer import write_trace

log = logging.getLogger(__name__)

_MAX_HISTORY = 10
_user_history: dict[str, list] = {}


class State(TypedDict, total=False):
    question: str
    user_id: str
    route: str
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


def _route_after_load(state: State) -> str:
    return "ok" if state.get("has_data") else "no_data"


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
    g.add_conditional_edges(
        "load_files",
        _route_after_load,
        {"ok": "format_answer", "no_data": "no_data"},
    )

    g.add_edge("smalltalk",     END)
    g.add_edge("off_topic",     END)
    g.add_edge("no_data",       END)
    g.add_edge("format_answer", END)

    return g.compile()


_graph = _build_graph()

_FALLBACK_TH = (
    "ขออภัยค่ะ ระบบเกิดข้อผิดพลาดชั่วคราว "
    "กรุณาลองใหม่อีกครั้ง หรือติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ โดยตรงค่ะ"
)
_FALLBACK_EN = (
    "Sorry, a temporary error occurred. Please try again or contact "
    "Walailuk University Dentist Hospital in Bangkok directly."
)


def ask(question: str, user_id: str = "cli") -> str:
    history = _user_history.get(user_id, [])
    try:
        result = _graph.invoke({
            "question": question,
            "user_id":  user_id,
            "history":  history,
            "trace":    [],
        })
        answer_text = result["answer"]
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
        from .agents import _is_thai
        return _FALLBACK_TH if _is_thai(question) else _FALLBACK_EN

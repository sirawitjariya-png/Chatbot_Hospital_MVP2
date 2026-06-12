"""CRAG-style LangGraph workflow — two-collection sequential RAG:

    supervisor → smalltalk / off_topic → END
              ↘ retrieve_chat → grade_chat
                    relevant       → draft_answer → END
                    not relevant   → retrieve_general → grade_general
                                         relevant     → draft_answer → END
                                         not relevant → web_search → answer_web → END

Worst case: supervisor + grade_chat + grade_general + web_search + answer = 5 LLM calls.
Typical happy path: supervisor + grade_chat + answer = 3 LLM calls.
"""
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import (
    supervisor,
    grade_chat,
    grade_general,
    draft_answer,
    web_search,
    no_data,
    smalltalk,
    off_topic,
)
from .rag import retrieve_chat, retrieve_general
from .tracer import write_trace

log = logging.getLogger(__name__)

_MAX_HISTORY = 10
_user_history: dict[str, list] = {}


class State(TypedDict, total=False):
    question: str
    user_id: str
    route: str
    stage: str
    retrieved: list
    context: str
    chat_relevant: bool
    general_relevant: bool
    draft: str
    answer: str
    history: list
    trace: Annotated[list, operator.add]


# ---- retrieve nodes ---------------------------------------------------------
def _retrieve_chat_node(state):
    hits = retrieve_chat(state["question"], k=4)
    context = "\n\n".join(h["text"] for h in hits) if hits else "(no relevant context found)"
    return {
        "retrieved": hits,
        "context": context,
        "stage": "chat",
        "trace": [{"node": "retrieve_chat", "chunks": len(hits)}],
    }


def _retrieve_general_node(state):
    hits = retrieve_general(state["question"], k=5)
    context = "\n\n".join(h["text"] for h in hits) if hits else "(no relevant context found)"
    return {
        "retrieved": hits,
        "context": context,
        "stage": "general",
        "trace": [{"node": "retrieve_general", "chunks": len(hits)}],
    }


# ---- routing ----------------------------------------------------------------
def _route_after_chat(state) -> str:
    return "ok" if state.get("chat_relevant") else "general"


def _route_after_general(state) -> str:
    return "ok" if state.get("general_relevant") else "web"


# ---- graph build ------------------------------------------------------------
def build_graph():
    g = StateGraph(State)

    g.add_node("supervisor",       supervisor)
    g.add_node("retrieve_chat",    _retrieve_chat_node)
    g.add_node("grade_chat",       grade_chat)
    g.add_node("retrieve_general", _retrieve_general_node)
    g.add_node("grade_general",    grade_general)
    g.add_node("draft_answer",     draft_answer)
    g.add_node("web_search",       web_search)
    g.add_node("answer_web",       draft_answer)
    g.add_node("smalltalk",        smalltalk)
    g.add_node("off_topic",        off_topic)
    g.add_node("no_data",          no_data)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["route"],
        {"rag": "retrieve_chat", "smalltalk": "smalltalk", "off_topic": "off_topic"},
    )

    # Stage 1 — แชท AI collection
    g.add_edge("retrieve_chat", "grade_chat")
    g.add_conditional_edges(
        "grade_chat",
        _route_after_chat,
        {"ok": "draft_answer", "general": "retrieve_general"},
    )

    # Stage 2 — general collection
    g.add_edge("retrieve_general", "grade_general")
    g.add_conditional_edges(
        "grade_general",
        _route_after_general,
        {"ok": "draft_answer", "web": "web_search"},
    )

    # Stage 3 — web fallback
    g.add_edge("web_search", "answer_web")

    g.add_edge("draft_answer",  END)
    g.add_edge("answer_web",    END)
    g.add_edge("smalltalk",     END)
    g.add_edge("off_topic",     END)
    g.add_edge("no_data",       END)

    return g.compile()


_graph = build_graph()


_FALLBACK = (
    "ขออภัย ระบบเกิดข้อผิดพลาดชั่วคราว กรุณาลองใหม่อีกครั้ง "
    "หรือติดต่อโรงพยาบาลวลัยลักษณ์โดยตรงค่ะ\n\n"
    "Sorry, a temporary error occurred. Please try again or contact "
    "Walailuk Hospital directly."
)


def ask(question: str, user_id: str = "cli") -> str:
    history = _user_history.get(user_id, [])
    try:
        result = _graph.invoke({
            "question": question,
            "user_id": user_id,
            "history": history,
            "trace": [],
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
        return _FALLBACK

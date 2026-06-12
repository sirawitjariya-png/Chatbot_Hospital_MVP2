"""Supervisor + RAG + smalltalk + grade_chunks + fallback nodes.

Model split:
  - ROUTER_MODEL: supervisor, grade_chunks, query expansion (cheap)
  - ANSWER_MODEL: answer, reflect (user-facing — upgrade this)
"""
import logging
from openai import OpenAI

from .config import (
    ROUTER_MODEL,
    ANSWER_MODEL,
    OPENAI_TIMEOUT_S,
    TAVILY_API_KEY,
)

log = logging.getLogger(__name__)
client = OpenAI()

# Tavily client — only instantiated if a key is present
_tavily = None
if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=TAVILY_API_KEY)
    except Exception as e:
        log.warning("Tavily disabled: %s", e)


def llm(messages, *, model: str = ANSWER_MODEL, **kw):
    """Thin OpenAI wrapper. Pass model=ROUTER_MODEL for routing/judging calls."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=OPENAI_TIMEOUT_S,  # why: prevents stuck calls from holding a Cloud Run instance
        **kw,
    ).choices[0].message.content


# ---------------------------------------------------------------------------
def supervisor(state):
    """Route: 'rag' | 'smalltalk' | 'off_topic'."""
    out = llm(
        [
            {
                "role": "system",
                "content": (
                    "You route hospital chatbot questions. Reply with ONE WORD only.\n"
                    "- 'rag' — the user asks about hospital facts: hours, services, doctors, "
                    "appointments, departments, prices, location, contact, symptoms, or medical treatment.\n"
                    "- 'smalltalk' — the user is greeting, saying thanks, or making chitchat.\n"
                    "- 'off_topic' — the question has nothing to do with the hospital, doctors, "
                    "or medical symptoms (e.g. cooking, sports, politics, general knowledge)."
                ),
            },
            *state.get("history", []),
            {"role": "user", "content": state["question"]},
        ],
        model=ROUTER_MODEL,
        temperature=0,
    )
    route = out.strip().lower()
    if route not in ("rag", "smalltalk", "off_topic"):
        route = "rag"
    return {"route": route, "trace": [{"node": "supervisor", "route": route}]}


# ---------------------------------------------------------------------------
def _grade(state, relevance_key: str, node_name: str) -> dict:
    chunks = state.get("retrieved", [])
    if not chunks:
        return {
            relevance_key: False,
            "trace": [{"node": node_name, "relevant": False, "reason": "no chunks"}],
        }
    joined = "\n\n---\n".join(c["text"][:400] for c in chunks)
    prompt = (
        "You are grading retrieved hospital documents for relevance.\n\n"
        f"QUESTION: {state['question']}\n\n"
        f"RETRIEVED CHUNKS:\n{joined}\n\n"
        "Do these chunks contain useful, specific information that can answer the question? "
        "Reply with ONE WORD: 'yes' or 'no'."
    )
    out = llm([{"role": "user", "content": prompt}], model=ROUTER_MODEL, temperature=0).strip().lower()
    relevant = out.startswith("yes")
    return {
        relevance_key: relevant,
        "trace": [{"node": node_name, "relevant": relevant, "n_chunks": len(chunks)}],
    }


def grade_chat(state):
    """Grade chunks from แชท AI collection."""
    return _grade(state, "chat_relevant", "grade_chat")


def grade_general(state):
    """Grade chunks from general collection."""
    return _grade(state, "general_relevant", "grade_general")


# ---------------------------------------------------------------------------
def draft_answer(state):
    """Generate the user-facing draft answer from state['context']."""
    is_first = len(state.get("history", [])) == 0
    greeting = (
        "This is the user's very first message. Begin your reply with a warm greeting: "
        "'สวัสดีค่ะ ยินดีต้อนรับสู่โรงพยาบาลวลัยลักษณ์ พร้อมให้บริการคุณค่ะ 😊 / "
        "Hello! Welcome to Walailuk Hospital, we are happy to assist you.' "
        "Then answer the question.\n\n"
        if is_first else ""
    )
    system = (
        "You are a warm, professional, and caring assistant for Walailuk Hospital. "
        "Answer the QUESTION using ONLY the CONTEXT below. "
        "Write in a humanized, polite, and empathetic tone — as if speaking to a patient in person. "
        "If the context is insufficient, politely say so and kindly suggest contacting the hospital directly.\n\n"
        f"{greeting}"
        f"CONTEXT:\n{state['context']}"
    )
    draft = llm(
        [
            {"role": "system", "content": system},
            *state.get("history", []),
            {"role": "user", "content": state["question"]},
        ],
        model=ANSWER_MODEL,
    )
    stage = state.get("stage", "")
    return {"draft": draft, "answer": draft, "trace": [{"node": "answer", "stage": stage, "draft": draft}]}


# ---------------------------------------------------------------------------
def _is_thai(text: str) -> bool:
    return any("฀" <= ch <= "๿" for ch in text)


def web_search(state):
    """Tavily web search fallback. Returns (context, stage='web')."""
    if not _tavily:
        return {
            "context": "(web search unavailable)",
            "stage": "web",
            "trace": [{"node": "web_search", "query": "", "results_count": 0}],
        }

    question = state["question"]
    prefix = "รพ วลัยลักษณ์ " if _is_thai(question) else "Walailuk"
    query = f"{prefix} {question}"
    try:
        results = _tavily.search(query, max_results=3)
        snippets = [r["content"] for r in results.get("results", [])]
    except Exception as e:
        log.warning("Tavily search failed: %s", e)
        snippets = []

    web_context = "\n\n".join(snippets) if snippets else "(no web results found)"
    return {
        "context": f"[Web search results]\n{web_context}",
        "stage": "web",
        "trace": [{"node": "web_search", "query": query, "results_count": len(snippets)}],
    }


# ---------------------------------------------------------------------------
def reflect(state):
    """Critique + polish the draft. Gated by ENABLE_REFLECTION at graph-build time."""
    prompt = (
        "You are a senior quality reviewer for Walailuk Hospital's chatbot.\n\n"
        "QUESTION: {question}\n\n"
        "CONTEXT (verified facts only):\n{context}\n\n"
        "DRAFT ANSWER:\n{draft}\n\n"
        "Review the draft against these three criteria:\n"
        "1. RELIABILITY — Every fact must be grounded in the CONTEXT. "
        "Remove or soften any claim that is not directly supported.\n"
        "2. POLITENESS — The tone must be warm, respectful, and professional, "
        "suitable for a patient or visitor of the hospital.\n"
        "3. LOGIC — The answer must directly and clearly address the question "
        "without contradictions or irrelevant content.\n\n"
        "If the draft meets all three criteria — reply with ONLY the draft unchanged.\n"
        "If it needs improvement — rewrite it to meet all criteria. "
        "Reply with ONLY the final answer, no commentary or explanation."
    ).format(
        question=state["question"],
        context=state.get("context", ""),
        draft=state["draft"],
    )
    final = llm([{"role": "user", "content": prompt}], model=ANSWER_MODEL)
    changed = final.strip() != state["draft"].strip()
    return {"answer": final, "trace": [{"node": "reflect", "changed": changed}]}


# ---------------------------------------------------------------------------
def no_data(_state):
    reply = (
        "ขออภัย ไม่พบข้อมูลเพียงพอในระบบสำหรับคำถามนี้ "
        "กรุณาติดต่อโรงพยาบาลวลัยลักษณ์โดยตรงเพื่อรับข้อมูลที่ถูกต้องครับ/ค่ะ\n\n"
        "Sorry, I couldn't find enough information to answer your question. "
        "Please contact Walailuk Hospital directly for accurate assistance."
    )
    return {"answer": reply, "trace": [{"node": "no_data"}]}


def off_topic(_state):
    reply = (
        "ขออภัยค่ะ คำถามของคุณไม่เกี่ยวข้องกับบริการของโรงพยาบาลวลัยลักษณ์ "
        "ระบบนี้ให้บริการเฉพาะข้อมูลด้านการแพทย์และบริการของโรงพยาบาลเท่านั้น "
        "หากมีคำถามเกี่ยวกับการรักษา แพทย์ หรือบริการของเรา ยินดีให้บริการเสมอค่ะ 🙏\n\n"
        "Sorry, your question is not related to Walailuk Hospital's services. "
        "This chatbot is designed to answer questions about our hospital, doctors, and medical services only. "
        "Please feel free to ask anything related to our healthcare services."
    )
    return {"answer": reply, "trace": [{"node": "off_topic"}]}


def smalltalk(state):
    reply = llm(
        [
            {
                "role": "system",
                "content": (
                    "You are a friendly hospital assistant. "
                    "Reply briefly (under 30 words). If the user just greeted you, "
                    "greet back and offer to answer questions about the hospital."
                ),
            },
            *state.get("history", []),
            {"role": "user", "content": state["question"]},
        ],
        model=ANSWER_MODEL,
    )
    return {"answer": reply, "trace": [{"node": "smalltalk"}]}

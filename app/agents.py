"""Core LLM helpers + static reply functions.

Model split:
  ROUTER_MODEL  — classify_question (cheap routing)
  ANSWER_MODEL  — format_answer, smalltalk (user-facing quality)

Language detection:
  _is_thai(text) — True when text contains Thai Unicode characters.
  Every user-facing function uses this to pick Thai or English copy.
"""
import logging
from openai import OpenAI

from .config import ROUTER_MODEL, ANSWER_MODEL, OPENAI_TIMEOUT_S

log = logging.getLogger(__name__)
_client = OpenAI()


# ---------------------------------------------------------------------------
# Language helper
# ---------------------------------------------------------------------------

def _is_thai(text: str) -> bool:
    """Return True if text contains at least one Thai character (U+0E00–U+0E7F)."""
    return any("฀" <= ch <= "๿" for ch in text)


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

def llm(messages: list, *, use_router: bool = False, **kw) -> str:
    """OpenAI chat wrapper.

    Pass use_router=True for cheap routing/classification calls.
    Raises on failure — callers must handle.
    """
    model = ROUTER_MODEL if use_router else ANSWER_MODEL
    try:
        return _client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=OPENAI_TIMEOUT_S,
            **kw,
        ).choices[0].message.content
    except Exception as e:
        log.error("LLM call failed (model=%s): %s", model, e)
        raise


# ---------------------------------------------------------------------------
# Smalltalk — two language variants
# ---------------------------------------------------------------------------

_SMALLTALK_TH = (
    "คุณคือเจ้าหน้าที่ให้บริการของ ศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ พูดจาสุภาพ เป็นมืออาชีพ และให้ความช่วยเหลืออย่างเต็มที่\n"
    "สำคัญมาก: ตอบเป็นภาษาไทยเท่านั้น ห้ามมีคำภาษาอังกฤษในคำตอบ ไม่เกิน 40 คำ\n"
    "ถ้าผู้ใช้ทักทาย กรุณาทักทายกลับอย่างสุภาพและเสนอให้ความช่วยเหลือด้านทันตกรรมหรือบริการของศูนย์ฯ"
)

_SMALLTALK_EN = (
    "You are a professional staff member at Walailuk University Dentist Hospital in Bangkok — courteous, attentive, and dedicated to providing excellent service.\n"
    "IMPORTANT: Reply in English ONLY — no Thai words, no mixed language. Under 40 words.\n"
    "If greeted, respond warmly and offer assistance with dental treatments or hospital services."
)

_SMALLTALK_FIRST_TH = (
    "นี่คือข้อความแรกของการสนทนา เริ่มด้วย "
    '"สวัสดีค่ะ ยินดีต้อนรับสู่ศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพค่ะ มีอะไรให้ดิฉันช่วยเหลือได้บ้างคะ" '
    "แล้วค่อยตอบตามธรรมชาติ\n\n"
)

_SMALLTALK_FIRST_EN = (
    "This is the first message of the conversation. "
    'Start with "Welcome to Walailuk University Dentist Hospital in Bangkok. How may I assist you today?" '
    "then continue naturally.\n\n"
)


def smalltalk(question: str, history: list) -> str:
    """Brief friendly reply for greetings / chitchat."""
    thai = _is_thai(question)
    is_first = len(history) == 0

    if thai:
        system = (_SMALLTALK_FIRST_TH if is_first else "") + _SMALLTALK_TH
        fallback = "สวัสดีค่ะ ยินดีต้อนรับสู่ศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพค่ะ มีอะไรให้ดิฉันช่วยเหลือได้บ้างคะ"
    else:
        system = (_SMALLTALK_FIRST_EN if is_first else "") + _SMALLTALK_EN
        fallback = "Welcome to Walailuk University Dentist Hospital in Bangkok. How may I assist you today?"

    try:
        return llm(
            [
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": question},
            ],
            use_router=False,
            max_tokens=120,
        )
    except Exception as e:
        log.error("smalltalk LLM failed: %s", e)
        return fallback


# ---------------------------------------------------------------------------
# Static replies — language-aware
# ---------------------------------------------------------------------------

def off_topic(question: str = "") -> str:
    if _is_thai(question):
        return (
            "ขออภัยค่ะ คำถามดังกล่าวอยู่นอกขอบเขตการให้บริการของศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพค่ะ "
            "ศูนย์ฯ ให้บริการตอบคำถามเฉพาะด้านทันตกรรมและบริการของโรงพยาบาลเท่านั้นค่ะ "
            "หากมีข้อสงสัยเกี่ยวกับการรักษาหรือบริการของเรา ยินดีให้ความช่วยเหลือค่ะ"
        )
    return (
        "I'm sorry, that question falls outside the scope of our services. "
        "Walailuk University Dentist Hospital in Bangkok provides information regarding dental treatments and hospital services only. "
        "Please feel free to ask if you have any inquiries in those areas."
    )


def no_data(question: str = "") -> str:
    if _is_thai(question):
        return (
            "ขออภัยค่ะ ขณะนี้ระบบยังไม่มีข้อมูลในส่วนนี้ค่ะ "
            "กรุณาติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ โดยตรง "
            "เพื่อรับข้อมูลที่ถูกต้องและครบถ้วนจากเจ้าหน้าที่ค่ะ"
        )
    return (
        "We apologize, but the requested information is not currently available in our system. "
        "Please contact Walailuk University Dentist Hospital in Bangkok directly "
        "for accurate and comprehensive assistance from our staff."
    )


def fallback_reply(question: str = "") -> str:
    """Used when the entire graph crashes unexpectedly."""
    if _is_thai(question):
        return (
            "ขออภัยค่ะ ขณะนี้ระบบมีปัญหาชั่วคราว "
            "กรุณาลองใหม่อีกครั้ง หรือติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ โดยตรงค่ะ"
        )
    return (
        "We apologize for the inconvenience. A temporary issue has occurred. "
        "Please try again shortly, or contact Walailuk University Dentist Hospital in Bangkok directly."
    )

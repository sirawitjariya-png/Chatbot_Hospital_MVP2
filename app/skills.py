"""Skill-based hospital chatbot agent — see app/SKILLS.md for full spec.

Skills (sequential pipeline):
  1. classify_question  — route + identify relevant treatment files  [ROUTER_MODEL]
  2. load_files         — read DOCX files from data/raw/             [no LLM]
  3. check_information  — verify content is non-empty                [no LLM]
  4. format_answer      — polish into user-facing reply              [ANSWER_MODEL]

Short-circuits:
  off_topic  → fixed bilingual reply
  smalltalk  → brief LLM reply
  no data    → fixed bilingual reply
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

try:
    from docx import Document
except ImportError as _e:
    raise ImportError("python-docx required: pip install python-docx") from _e

from .config import RAW_DIR
from .agents import _is_thai

log = logging.getLogger(__name__)

# Max files the classifier may request (prevents reading every file on vague questions)
_MAX_TREATMENT_FILES = 3

# Truncate context fed to LLM to stay well within token limits (~6 000 tokens ≈ 24 000 chars)
_MAX_CONTEXT_CHARS = 24_000

# ---------------------------------------------------------------------------
# File catalog — mirrors the 12 files in data/raw/
# ---------------------------------------------------------------------------
FILE_CATALOG: dict[int, str] = {
    1:  "ขูดหินปูนและเกลารากฟัน (Scaling & Root Planing)",
    2:  "อุดฟัน (Dental Filling)",
    3:  "คลองรากฟัน (Root Canal Treatment)",
    4:  "ครอบฟันและสะพานฟัน (Crown & Bridge)",
    5:  "ฟันปลอมถอดได้ (Removable Denture)",
    6:  "วีเนียร์ (Dental Veneer)",
    7:  "จัดฟัน (Orthodontics / Braces)",
    8:  "ถอนฟันและผ่าฟันคุด (Tooth Extraction & Wisdom Tooth)",
    9:  "ผ่าตัดตกแต่งกระดูกหรือเนื้อเยื่ออ่อน (Oral Surgery)",
    10: "รากฟันเทียม (Dental Implants)",
    11: "ทันตกรรมเด็ก (Pediatric Dentistry)",
    12: "ราคาและข้อมูลทั่วไปเกี่ยวกับโรงพยาบาล (Hospital Info & Pricing)",
}

_CATALOG_TEXT = "\n".join(f"{k}. {v}" for k, v in FILE_CATALOG.items())

# Pre-compiled regex to strip markdown code fences from LLM JSON output
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Directory index — built once, avoids re-scanning on every file lookup
# ---------------------------------------------------------------------------
_docx_index: dict[int, Path] | None = None


def _build_index() -> dict[int, Path]:
    """Scan RAW_DIR once and map file number → Path."""
    index: dict[int, Path] = {}
    try:
        for p in Path(RAW_DIR).iterdir():
            if p.suffix != ".docx" or p.name.startswith("~$"):
                continue
            # filename starts with "<number>."
            prefix = p.name.split(".")[0]
            try:
                num = int(prefix)
                index[num] = p
            except ValueError:
                pass
    except Exception as e:
        log.warning("Failed to build docx index from RAW_DIR: %s", e)
    return index


def _get_index() -> dict[int, Path]:
    global _docx_index
    if _docx_index is None:
        _docx_index = _build_index()
    return _docx_index


def _find_docx(number: int) -> Path | None:
    return _get_index().get(number)


def _read_docx(path: Path) -> str:
    """Extract plain text from a DOCX file."""
    try:
        doc = Document(str(path))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines)
    except Exception as e:
        log.warning("Read %s failed: %s", path, e)
        return ""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap JSON in."""
    return _FENCE_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Skill 1 — classify_question
# ---------------------------------------------------------------------------

# Only pass the last 2 history messages to the classifier — it needs just
# enough context to understand a follow-up question, not the full history.
_CLASSIFIER_HISTORY_LIMIT = 2

_CLASSIFY_SYSTEM = (
    "You are a routing classifier for a hospital dental chatbot (Walailuk University Dentist Hospital in Bangkok).\n"
    "You are a routing classifier for a hospital dental chatbot (Walailuk University Dentist Hospital in Bangkok).\n"
    "Your ONLY job is to output a single JSON object — no explanation, no markdown fences.\n\n"
    "Available treatment files (numbers 1–11):\n"
    f"{_CATALOG_TEXT}\n\n"
    "OUTPUT FORMAT (strict JSON, nothing else):\n"
    '{"route": "<route>", "files": [<numbers>]}\n\n'
    "ROUTE RULES:\n"
    '- "treatment": the question is about a specific dental procedure '
    "(symptoms, steps, risks, aftercare, OR price of that specific procedure). "
    "Set files = list of the relevant treatment numbers (1–11, max 3).\n"
    '- "general": hospital info, overall price list, location, hours, contact, '
    "insurance, appointment booking, or a question about the hospital in general. "
    "Set files = [].\n"
    '- "smalltalk": greeting, farewell, thanks, or casual chat with no medical content. '
    "Set files = [].\n"
    '- "off_topic": entirely unrelated to hospitals, dental care, or medicine. '
    "Set files = [].\n\n"
    "The user may ask in Thai OR English — classify by meaning, not language.\n\n"
    "EXAMPLES (Thai):\n"
    '  Q: "ขูดหินปูนเจ็บไหม"                   → {"route":"treatment","files":[1]}\n'
    '  Q: "ราคาขูดหินปูนเท่าไร"                → {"route":"treatment","files":[1]}\n'
    '  Q: "โรงพยาบาลเปิดกี่โมง"               → {"route":"general","files":[]}\n'
    '  Q: "ราคาทั้งหมดมีอะไรบ้าง"              → {"route":"general","files":[]}\n'
    "The user may ask in Thai OR English — classify by meaning, not language.\n\n"
    "EXAMPLES (Thai):\n"
    '  Q: "ขูดหินปูนเจ็บไหม"                   → {"route":"treatment","files":[1]}\n'
    '  Q: "ราคาขูดหินปูนเท่าไร"                → {"route":"treatment","files":[1]}\n'
    '  Q: "โรงพยาบาลเปิดกี่โมง"               → {"route":"general","files":[]}\n'
    '  Q: "ราคาทั้งหมดมีอะไรบ้าง"              → {"route":"general","files":[]}\n'
    '  Q: "จัดฟันกับรากเทียม ราคาต่างกันยังไง" → {"route":"treatment","files":[7,10]}\n'
    '  Q: "สวัสดีครับ"                         → {"route":"smalltalk","files":[]}\n'
    '  Q: "ใครชนะบอลเมื่อคืน"                 → {"route":"off_topic","files":[]}\n\n'
    "EXAMPLES (English):\n"
    '  Q: "Does scaling hurt?"                  → {"route":"treatment","files":[1]}\n'
    '  Q: "How much does scaling cost?"         → {"route":"treatment","files":[1]}\n'
    '  Q: "What are the hospital opening hours?"→ {"route":"general","files":[]}\n'
    '  Q: "What is the full price list?"        → {"route":"general","files":[]}\n'
    '  Q: "Compare braces vs implants price"   → {"route":"treatment","files":[7,10]}\n'
    '  Q: "Hello"                               → {"route":"smalltalk","files":[]}\n'
    '  Q: "Who won the game last night?"        → {"route":"off_topic","files":[]}\n\n'
    '  Q: "สวัสดีครับ"                         → {"route":"smalltalk","files":[]}\n'
    '  Q: "ใครชนะบอลเมื่อคืน"                 → {"route":"off_topic","files":[]}\n\n'
    "EXAMPLES (English):\n"
    '  Q: "Does scaling hurt?"                  → {"route":"treatment","files":[1]}\n'
    '  Q: "How much does scaling cost?"         → {"route":"treatment","files":[1]}\n'
    '  Q: "What are the hospital opening hours?"→ {"route":"general","files":[]}\n'
    '  Q: "What is the full price list?"        → {"route":"general","files":[]}\n'
    '  Q: "Compare braces vs implants price"   → {"route":"treatment","files":[7,10]}\n'
    '  Q: "Hello"                               → {"route":"smalltalk","files":[]}\n'
    '  Q: "Who won the game last night?"        → {"route":"off_topic","files":[]}\n\n'
    "IMPORTANT: If the question asks about price or details of a SPECIFIC treatment, "
    'always use "treatment" (not "general"), and include that treatment\'s file number.'
)


def classify_question(question: str, history: list, llm_fn) -> dict:
    """Route question and identify relevant treatment files (Skill 1).

    Returns:
        {route: str, files: list[int], trace_entry: dict}
    route: "off_topic" | "smalltalk" | "treatment" | "general"
    files: 1–11 numbers (empty for general/smalltalk/off_topic)
    """
    recent_history = history[-_CLASSIFIER_HISTORY_LIMIT:] if history else []
    route = "general"
    files: list[int] = []
    try:
        raw = llm_fn(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                *recent_history,
                {"role": "user", "content": question},
            ],
            use_router=True,
        )
        data = json.loads(_strip_fences(raw))
        parsed_route = str(data.get("route", "general")).lower().strip()
        if parsed_route in ("off_topic", "smalltalk", "treatment", "general"):
            route = parsed_route

        raw_files = data.get("files") or []
        safe_files: list[int] = []
        for f in raw_files:
            try:
                n = int(f)
                if 1 <= n <= 11:
                    safe_files.append(n)
            except (ValueError, TypeError):
                pass
        files = safe_files[:_MAX_TREATMENT_FILES]

    except Exception as e:
        log.warning("classify_question failed (%s) — defaulting to general", e)

    trace_entry = {"node": "classify", "route": route, "files": files}
    return {"route": route, "files": files, "trace_entry": trace_entry}


# ---------------------------------------------------------------------------
# Skill 2 & 3 — load_files + check_information
# ---------------------------------------------------------------------------

def load_files(file_numbers: list[int]) -> dict:
    """Read treatment DOCX files. File 12 is always included (Skill 2 & 3).

    Returns:
        {content: dict[int, str], has_data: bool, trace_entry: dict}
    """
    numbers = sorted(set(file_numbers + [12]))
    content: dict[int, str] = {}
    file_trace: list[dict] = []

    for num in numbers:
        path = _find_docx(num)
        if path is None:
            log.warning("File %d not found in RAW_DIR", num)
            file_trace.append({"file": num, "found": False})
            continue
        text = _read_docx(path)
        entry: dict = {"file": num, "found": True, "chars": len(text), "name": path.name}
        if text:
            content[num] = text
        else:
            entry["readable"] = False
        file_trace.append(entry)

    has_data = bool(content) and any(len(v) > 50 for v in content.values())
    trace_entry = {"node": "read_files", "files": file_trace, "has_data": has_data}
    return {"content": content, "has_data": has_data, "trace_entry": trace_entry}


# ---------------------------------------------------------------------------
# Skill 4 — format_answer  (two separate prompts by language)
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM_TH = (
    "คุณคือผู้ช่วยของศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ พูดจาเป็นกันเอง อบอุ่น และสุภาพ "
    "คุณคือผู้ช่วยของศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ พูดจาเป็นกันเอง อบอุ่น และสุภาพ "
    "เหมือนพนักงานต้อนรับที่ใส่ใจคนไข้จริงๆ ไม่ใช่หุ่นยนต์\n\n"
    "สำคัญมาก: ตอบเป็นภาษาไทยเท่านั้น ห้ามมีคำภาษาอังกฤษในคำตอบ\n"
    "ใช้เฉพาะข้อมูลจาก CONTEXT ด้านล่างเท่านั้น ห้ามเดาหรือเพิ่มข้อมูลที่ไม่มีใน CONTEXT\n\n"
    "สำคัญมาก: ตอบเป็นภาษาไทยเท่านั้น ห้ามมีคำภาษาอังกฤษในคำตอบ\n"
    "ใช้เฉพาะข้อมูลจาก CONTEXT ด้านล่างเท่านั้น ห้ามเดาหรือเพิ่มข้อมูลที่ไม่มีใน CONTEXT\n\n"
    "แนวทางการตอบ:\n"
    "- ตอบเฉพาะสิ่งที่ผู้ใช้ถามเท่านั้น ห้ามเพิ่มข้อมูลอื่นที่ไม่ได้ถาม\n"
    "- ประโยคสั้น กระชับ อ่านง่าย เป็นกันเอง แต่ยังสุภาพ\n"
    "- ใช้ bullet points หรือรายการหมายเลขถ้ามีหลายข้อหรือมีขั้นตอน\n"
    "- ถ้ามีราคาใน context ระบุให้ชัดเจนทุกครั้ง (ห้ามเดาราคา)\n"
    "- ถ้าข้อมูลใน context ไม่พอ บอกตรงๆ อย่างสุภาพ "
    "แล้วแนะนำให้ติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพโดยตรง (โทร/LINE/มาด้วยตัวเอง)\n"
    "- ไม่พูดซ้ำๆ และไม่ยืดเยื้อโดยไม่จำเป็น\n"
    "- ห้ามให้ข้อมูลที่ผู้ใช้ไม่ได้ถาม เช่น คำแนะนำก่อน/หลังการรักษา ราคา หรือข้อมูลติดต่อ หากไม่ได้ถามถึง\n\n"
    "Please left the topic format unchanged. (not **,*,` added)\n"
    "Don't provide information under ** สำหรับข้อมูลละเอียด ** topic unless users ask for details"
)

_ANSWER_SYSTEM_EN = (
    "You're the friendly assistant at Walailuk University Dentist Hospital in Bangkok — warm, easy-going, and genuinely helpful. "
    "You're the friendly assistant at Walailuk University Dentist Hospital in Bangkok — warm, easy-going, and genuinely helpful. "
    "Think of yourself as a front desk person who actually cares, not a scripted robot.\n\n"
    "IMPORTANT: Reply in English ONLY — no Thai words, no mixed language. "
    "The source documents in CONTEXT may be written in Thai; translate the relevant information into English in your reply.\n\n"
    "Use ONLY the information in the CONTEXT section below — don't make up facts, prices, or advice.\n\n"
    "IMPORTANT: Reply in English ONLY — no Thai words, no mixed language. "
    "The source documents in CONTEXT may be written in Thai; translate the relevant information into English in your reply.\n\n"
    "Use ONLY the information in the CONTEXT section below — don't make up facts, prices, or advice.\n\n"
    "How to respond:\n"
    "- Answer ONLY what the user asked — do not volunteer extra info they didn't request\n"
    "- Keep it conversational and friendly, but still professional\n"
    "- Short sentences beat long ones every time\n"
    "- Use bullet points or numbered lists for steps, risks, or multiple items\n"
    "- Only include prices, pre/post-care tips, or contact info if the user asked for them\n"
    "- If the context doesn't cover the question, be upfront and suggest contacting "
    "Walailuk University Dentist Hospital in Bangkok directly (call/LINE/walk-in)\n"
    "- Don't repeat yourself or pad the answer\n\n"
    "Please left the topic format unchanged. (not **,*,` added)\n"
    "Don't provide information under ** สำหรับข้อมูลละเอียด ** topic unless users ask for details"
)

_FIRST_MSG_TH = (
    "นี่คือข้อความแรกของการสนทนา เริ่มด้วย "
    '"สวัสดีค่ะ ยินดีต้อนรับสู่ศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพ 😊" '
    "แล้วค่อยตอบคำถามต่อเลย\n\n"
)

_FIRST_MSG_EN = (
    "This is the very first message of the conversation. "
    'Start with "Hey there! Welcome to Walailuk University Dentist Hospital in Bangkok 😊" '
    "then answer their question naturally.\n\n"
)


def format_answer(question: str, content: dict[int, str], history: list, llm_fn) -> str:
    """Generate a polished, structured answer from loaded file content (Skill 4).

    Detects question language and applies the matching Thai or English prompt.
    Never raises — falls back to a fixed no-data reply on LLM error.
    """
    thai = _is_thai(question)

    # Build context sections, truncating to stay within token budget
    sections: list[str] = []
    total_chars = 0
    for num in sorted(content.keys()):
        label = FILE_CATALOG.get(num, f"File {num}")
        text = content[num]
        remaining = _MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        chunk = text[:remaining]
        sections.append(f"--- {label} ---\n{chunk}")
        total_chars += len(chunk)

    combined = "\n\n".join(sections)
    is_first = len(history) == 0

    if thai:
        base_system = _ANSWER_SYSTEM_TH
        first_msg = _FIRST_MSG_TH if is_first else ""
        error_reply = (
            "ขอโทษนะคะ ข้อมูลในระบบยังไม่ครอบคลุมเรื่องนี้ค่ะ "
            "ลองติดต่อศูนย์ทันตกรรม ม.วลัยลักษณ์ กรุงเทพโดยตรงได้เลยนะคะ 🙏"
        )
    else:
        base_system = _ANSWER_SYSTEM_EN
        first_msg = _FIRST_MSG_EN if is_first else ""
        error_reply = (
            "Sorry, I don't have that info in my system right now! "
            "Your best bet is to reach out to Walailuk University Dentist Hospital in Bangkok directly. 🙏"
        )

    system = f"{first_msg}{base_system}\n\nCONTEXT:\n{combined}"

    try:
        answer = llm_fn(
            [
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": question},
            ],
            use_router=False,
            max_tokens=2048,
        )
    except Exception as e:
        log.error("format_answer LLM call failed: %s", e)
        answer = error_reply
    return answer

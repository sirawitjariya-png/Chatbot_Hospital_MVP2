# Hospital Chatbot — Skill Agent Specification

## Agent: HospitalSkillAgent

One agent with a sequential skill pipeline. No vector database.
Reads `.docx` files directly from `data/raw/`.

---

## Available Treatment Files

| # | Thai | English |
|---|------|---------|
| 1 | ขูดหินปูนและเกลารากฟัน | Scaling & Root Planing |
| 2 | อุดฟัน | Dental Filling |
| 3 | คลองรากฟัน | Root Canal Treatment |
| 4 | ครอบฟันและสะพานฟัน | Crown & Bridge |
| 5 | ฟันปลอมถอดได้ | Removable Denture |
| 6 | วีเนียร์ | Dental Veneer |
| 7 | จัดฟัน | Orthodontics / Braces |
| 8 | ถอนฟันและผ่าฟันคุด | Tooth Extraction & Wisdom Tooth |
| 9 | ผ่าตัดตกแต่งกระดูกหรือเนื้อเยื่ออ่อน | Oral Surgery |
| 10 | รากฟันเทียม | Dental Implants |
| 11 | ทันตกรรมเด็ก | Pediatric Dentistry |
| 12 | ราคาและข้อมูลทั่วไปเกี่ยวกับโรงพยาบาล | Hospital Info & Pricing *(always read)* |

---

## Skill Pipeline

### Skill 1 — classify_question
**Model:** ROUTER_MODEL (cheap)  
**Input:** question + history  
**Output:** `{route, files}`

Routes:
- `treatment` — question about a specific dental treatment → `files` = list of 1–11
- `general` — hospital info, location, hours, pricing, appointment → `files` = []
- `smalltalk` — greeting, thanks, casual chat → short-circuit to smalltalk reply
- `off_topic` — unrelated to hospital/dental → short-circuit to off_topic reply

Tracer label: `CLASSIFY`

---

### Skill 2 — load_files
**No LLM call**  
**Input:** treatment file numbers from Skill 1  
**Output:** `{file_num: full_text}` dict

Rules:
- File 12 is **always** included
- Files 1–11 are added based on `files` from classification
- Reads `.docx` files via `python-docx`

Tracer label: `READ_FILES`

---

### Skill 3 — check_information
**No LLM call**  
**Input:** content dict from Skill 2  
**Output:** boolean

Returns `True` if any file content > 50 chars was loaded.  
If `False` → short-circuit to no_data reply.

Tracer label: embedded in `READ_FILES` result

---

### Skill 4 — format_answer
**Model:** ANSWER_MODEL (quality)  
**Input:** question + content dict + history  
**Output:** formatted answer string

Format rules:
- Use bullet points or numbered lists where appropriate
- Include relevant prices if in context
- Mention pre/post treatment advice if relevant
- Warm, empathetic tone for patients
- If first message: add welcome greeting

Tracer label: `ANSWER`

---

## Short-circuit Paths

```
off_topic  → fixed bilingual reply (no file reads, no LLM)
smalltalk  → LLM brief reply (no file reads)
no_data    → fixed bilingual reply
```

## Full Path

```
classify → load_files → check_info → format_answer
```

---

## Trace Format

Each skill appends to `state["trace"]`:

```json
{"node": "classify",    "route": "treatment", "files": [3, 12]}
{"node": "read_files",  "files": [{"file": 3, "found": true, "chars": 4200, "name": "3.xxx.docx"}, ...]}
{"node": "answer",      "draft": "..."}
{"node": "smalltalk"}
{"node": "off_topic"}
{"node": "no_data"}
```

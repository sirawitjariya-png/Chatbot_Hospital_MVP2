# Hospital Chatbot — MVP2

A bilingual (Thai + English) skill-based chatbot for **Walailuk Hospital**.
Patients ask questions about treatments, prices, hours, appointments, and more.
The bot classifies the question, reads the relevant `.docx` files directly, and answers — no vector database required.

Channels: **LINE** · **Facebook Messenger** · **REST API** · **CLI terminal**

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| **LLM (classify)** | `gpt-4o-mini` | Cheap — routes question + picks files |
| **LLM (answer)** | `gpt-4.1` | Quality user-facing reply |
| **Workflow** | LangGraph | Stateful skill pipeline with conditional edges |
| **Doc reader** | `python-docx` | Reads `.docx` files directly — no embedding needed |
| **API** | FastAPI + Uvicorn | LINE/Facebook webhooks wired |

---

## Architecture — Skill Pipeline

```
User Question
     │
     ▼
┌────────────┐
│  Skill 1   │  classify_question  [gpt-4o-mini]
│  CLASSIFY  │  → route + treatment file numbers
└────┬───────┘
     │
  ┌──┴─────────────────┐
  │                    │
off_topic           smalltalk ──► brief LLM reply
  │
  ▼
fixed reply      treatment / general
                      │
                      ▼
              ┌───────────────┐
              │   Skill 2+3   │  load_files  [no LLM]
              │  LOAD FILES   │  always reads file 12
              │               │  + related files 1–11
              └───────┬───────┘
                      │
               has_data?
               /        \
            no            yes
             │              │
        no_data          ┌──────────┐
        reply            │ Skill 4  │  format_answer  [gpt-4.1]
                         │  ANSWER  │  structured, bilingual
                         └──────────┘
```

**LLM calls per question:** 2 (classify + answer). Off-topic/smalltalk: 0–1.

---

## Treatment Files (data/raw/)

| # | Thai | English |
|---|------|---------|
| 1 | ขูดหินปูนและเกลารากฟัน | Scaling & Root Planing |
| 2 | อุดฟัน | Dental Filling |
| 3 | คลองรากฟัน | Root Canal |
| 4 | ครอบฟันและสะพานฟัน | Crown & Bridge |
| 5 | ฟันปลอมถอดได้ | Removable Denture |
| 6 | วีเนียร์ | Dental Veneer |
| 7 | จัดฟัน | Orthodontics |
| 8 | ถอนฟันและผ่าฟันคุด | Extraction & Wisdom Tooth |
| 9 | ผ่าตัดตกแต่งกระดูกหรือเนื้อเยื่ออ่อน | Oral Surgery |
| 10 | รากฟันเทียม | Dental Implants |
| 11 | ทันตกรรมเด็ก | Pediatric Dentistry |
| **12** | **ราคาและข้อมูลทั่วไป** | **Hospital Info & Pricing _(always read)_** |

Each file 1–11 contains: treatment purpose, procedure steps, risks, alternatives, pre/post advice.

---

## Quickstart

### 1. Environment

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env
```

### 2. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Chat (CLI)

```bash
python main.py chat
```

### 4. Server

```bash
python main.py serve
# POST /chat  {"message": "..."}  →  {"answer": "..."}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `ROUTER_MODEL` | `gpt-4o-mini` | Cheap model for classification |
| `ANSWER_MODEL` | `gpt-4.1` | Quality model for answers |
| `OPENAI_TIMEOUT_S` | `60` | Per-call timeout (seconds) |
| `RAW_DIR` | `data/raw` | Path to treatment `.docx` files |
| `CHAT_API_KEY` | _(open)_ | X-API-Key for `/chat` endpoint |
| `LINE_CHANNEL_SECRET` | — | LINE webhook HMAC secret |
| `LINE_CHANNEL_ACCESS_TOKEN` | — | LINE reply token |
| `FB_PAGE_ACCESS_TOKEN` | — | Facebook page token |
| `FB_VERIFY_TOKEN` | `change-me` | Facebook webhook verify token |

---

## Tracing

Every question generates a trace written to two places:

1. **`logs/<user_id>/YYYY-MM.log`** — pretty ASCII boxes for local debugging
2. **stdout JSON** — structured line for Cloud Logging

Example trace:
```
════════════════════════════════════════════════════════════════════════
  2026-06-10 14:23:01   user: line_U123abc
════════════════════════════════════════════════════════════════════════

  ┌─ QUESTION
  │  ขูดหินปูนราคาเท่าไรครับ

  ┌─ CLASSIFY  →  TREATMENT  files=[1]

  ┌─ READ_FILES  —  2 file(s) loaded
  │  ✓ 1.เอกสารการรักษาขูดหินปูนและเกลารากฟัน.docx (3842 chars)
  │  ✓ 12.ราคาและข้อมูลทั่วไปเกี่ยวกับโรงพยาบาล.docx (2105 chars)

  ┌─ ANSWER
  │  การขูดหินปูนที่โรงพยาบาลวลัยลักษณ์ เริ่มต้นที่ 500 บาท...

  ┌─ FINAL ANSWER
  │  ...
```

---

## Deploy (Google Cloud Run)

```bash
gcloud builds submit --tag gcr.io/<PROJECT>/<IMAGE>
gcloud run deploy hospital-chatbot \
  --image gcr.io/<PROJECT>/<IMAGE> \
  --region asia-southeast1 \
  --set-env-vars OPENAI_API_KEY=... \
  --allow-unauthenticated
```

Set `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`, and other webhook vars in the Cloud Run environment.

---

## Project Structure

```
├── main.py              # CLI: chat | serve
├── server.py            # FastAPI + LINE/Facebook webhooks
├── requirements.txt
├── app/
│   ├── config.py        # Env vars
│   ├── agents.py        # LLM wrapper + static replies
│   ├── skills.py        # Skill 1–4 implementation
│   ├── graph.py         # LangGraph orchestration + ask()
│   ├── tracer.py        # Dual logging (file + Cloud Logging)
│   └── SKILLS.md        # Skill specification
├── data/raw/            # 12 treatment .docx files
├── tests/
│   └── test_skills.py   # Unit tests (29 tests)
└── logs/                # Per-user conversation traces
```

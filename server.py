"""FastAPI server: /health, /chat, plus LINE + Facebook webhook stubs.

Run with: python main.py serve
"""
import asyncio
import hmac
import hashlib
import base64
import logging
import requests
from fastapi import FastAPI, Request, Header, HTTPException

from app.config import (
    LINE_CHANNEL_SECRET,
    LINE_CHANNEL_TOKEN,
    FB_VERIFY_TOKEN,
    FB_PAGE_TOKEN,
    CHAT_API_KEY,
)
from app.graph import ask

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital Chatbot MVP1")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
async def chat(req: Request, x_api_key: str = Header("", alias="X-API-Key")):
    """JSON API: POST {"message": "..."} -> {"answer": "..."}

    Protected by X-API-Key header when CHAT_API_KEY env var is set.
    If CHAT_API_KEY is empty, the endpoint is open (useful for local dev).
    """
    # why: prevents random scrapers from burning OpenAI budget on a public Cloud Run URL
    if CHAT_API_KEY and not hmac.compare_digest(x_api_key, CHAT_API_KEY):
        raise HTTPException(401, "missing or invalid X-API-Key")

    body = await req.json()
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "missing 'message'")
    return {"answer": ask(msg)}


# ---------- LINE ----------
def _handle_line_message(user_id: str, text: str, reply_token: str) -> None:
    try:
        reply = ask(text, user_id=user_id)
    except Exception as e:
        logger.error("ask() failed for user %s: %s", user_id, e)
        reply = (
            "ขออภัย ระบบเกิดข้อผิดพลาดชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ\n"
            "Sorry, a temporary error occurred. Please try again."
        )
    try:
        requests.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {LINE_CHANNEL_TOKEN}"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": reply[:4500]}]},
            timeout=10,
        )
        logger.info("LINE reply sent to %s: %s", user_id, reply[:80])
    except Exception as e:
        logger.error("LINE reply failed for user %s: %s", user_id, e)


@app.post("/webhook/line")
async def line_webhook(req: Request, x_line_signature: str = Header("")):
    raw = await req.body()
    if LINE_CHANNEL_SECRET:
        sig = base64.b64encode(
            hmac.new(LINE_CHANNEL_SECRET.encode(), raw, hashlib.sha256).digest()
        ).decode()
        logger.debug("LINE sig check | secret_len=%d | computed=%s | received=%s",
                     len(LINE_CHANNEL_SECRET), sig[:10], x_line_signature[:10])
        if not hmac.compare_digest(sig, x_line_signature):
            logger.warning("LINE bad signature | computed=%s | received=%s",
                           sig[:20], x_line_signature[:20])
            raise HTTPException(403, "bad signature")  # why: LINE requires HMAC verification

    LINE_VERIFY_TOKEN = "00000000000000000000000000000000"
    jobs = []
    loop = asyncio.get_running_loop()
    for ev in (await req.json()).get("events", []):
        reply_token = ev.get("replyToken", "")
        if reply_token == LINE_VERIFY_TOKEN:
            continue  # LINE "Verify" button test event — not a real message
        if ev.get("type") == "message" and ev["message"]["type"] == "text":
            text = (ev["message"].get("text") or "").strip()
            if not text:
                continue
            user_id = ev.get("source", {}).get("userId", "unknown")
            logger.info("LINE message from %s: %s", user_id, text)
            # why: run_in_executor keeps the request active — Cloud Run's CPU stays
            # fully allocated the whole time (unlike background_tasks which runs after
            # the 200 response when CPU is already throttled back to near-zero).
            jobs.append(loop.run_in_executor(None, _handle_line_message, user_id, text, reply_token))
    if jobs:
        await asyncio.gather(*jobs)
    return {"ok": True}


# ---------- Facebook Messenger ----------
@app.get("/webhook/facebook")
async def fb_verify(req: Request):
    p = req.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == FB_VERIFY_TOKEN:
        return int(p.get("hub.challenge", 0))
    raise HTTPException(403)


def _send_fb_reply(sender: str, text: str) -> None:
    reply = ask(text)
    resp = requests.post(
        "https://graph.facebook.com/v21.0/me/messages",
        params={"access_token": FB_PAGE_TOKEN},
        json={
            "recipient": {"id": sender},
            "message": {"text": reply[:1900]},
            "messaging_type": "RESPONSE",
        },
        timeout=30,
    )
    if not resp.ok:
        logger.error("FB send failed %s: %s", resp.status_code, resp.text)
    else:
        logger.info("FB reply sent to %s", sender)


@app.post("/webhook/facebook")
async def fb_webhook(req: Request):
    payload = await req.json()
    loop = asyncio.get_running_loop()
    jobs = []
    for entry in payload.get("entry", []):
        for ev in entry.get("messaging", []):
            sender = ev.get("sender", {}).get("id")
            text = ev.get("message", {}).get("text")
            if sender and text:
                logger.info("FB message from %s: %s", sender, text)
                jobs.append(loop.run_in_executor(None, _send_fb_reply, sender, text))
    if jobs:
        await asyncio.gather(*jobs)
    return {"ok": True}

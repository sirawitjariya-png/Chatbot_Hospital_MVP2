"""Central configuration — env vars loaded once, paths pinned absolutely."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent

# --- OpenAI ----------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file or environment before starting."
    )

# Two model knobs: cheap for routing, quality for user-facing answers
ROUTER_MODEL = os.getenv("ROUTER_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
ANSWER_MODEL = os.getenv("ANSWER_MODEL") or os.getenv("LLM_MODEL") or "gpt-4.1"

OPENAI_TIMEOUT_S = int(os.getenv("OPENAI_TIMEOUT_S", "60"))

# --- Data ------------------------------------------------------------------
RAW_DIR = os.getenv("RAW_DIR", str(_REPO_ROOT / "data" / "raw"))

# --- Auth ------------------------------------------------------------------
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")

# --- Logs ------------------------------------------------------------------
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(_REPO_ROOT / "logs")))

# --- Channel credentials ---------------------------------------------------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
FB_PAGE_TOKEN       = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
FB_VERIFY_TOKEN     = os.getenv("FB_VERIFY_TOKEN", "change-me")

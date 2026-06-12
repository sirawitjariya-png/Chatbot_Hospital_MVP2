"""Write per-user conversation traces.

Two outputs per call:
  1. logs/<user_id>/YYYY-MM.log  — pretty human-readable ASCII boxes (local debug)
  2. logging.info(json.dumps(...)) — single-line JSON to stdout (Cloud Logging)

File write is best-effort: skipped if read-only filesystem.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import LOGS_DIR

log = logging.getLogger("trace")
W = 72


def _box(label: str) -> str:
    return f"  ┌─ {label}"


def _lines(text: str, indent: str = "  │  ") -> list[str]:
    return [f"{indent}{ln}" for ln in text.splitlines()]


def _render_pretty(user_id: str, question: str, trace: list, final_answer: str, ts: str) -> str:
    out: list[str] = []
    out.append("=" * W)
    out.append(f"  {ts}   user: {user_id}")
    out.append("=" * W)
    out.append("")
    out.append(_box("QUESTION"))
    out += _lines(question)
    out.append("")

    for entry in trace:
        node = entry.get("node", "?")

        if node == "classify":
            route = entry.get("route", "?").upper()
            files = entry.get("files", [])
            files_str = f"  files={files}" if files else ""
            out.append(_box(f"CLASSIFY  →  {route}{files_str}"))
            if entry.get("error"):
                out.append(f"  │  ⚠ error: {entry['error']}")
            out.append("")

        elif node == "read_files":
            file_list = entry.get("files", [])
            found = [f for f in file_list if f.get("found")]
            out.append(_box(f"READ_FILES  —  {len(found)} file(s) loaded"))
            for f in file_list:
                status = "✓" if f.get("found") else "✗ not found"
                chars = f" ({f['chars']} chars)" if f.get("chars") else ""
                name = f.get("name", f"file {f['file']}")
                out.append(f"  │  {status} {name}{chars}")
            if entry.get("error"):
                out.append(f"  │  ⚠ error: {entry['error']}")
            out.append("")

        elif node == "answer":
            out.append(_box("ANSWER"))
            # show first 400 chars in the pretty log; full answer is in FINAL ANSWER
            out += _lines(entry.get("draft", "")[:400])
            out.append("")

        elif node == "smalltalk":
            out.append(_box("SMALLTALK"))
            out.append("")

        elif node == "off_topic":
            out.append(_box("OFF-TOPIC  (not hospital-related)"))
            out.append("")

        elif node == "no_data":
            out.append(_box("NO DATA  —  files loaded but no information found"))
            out.append("")

    out.append(_box("FINAL ANSWER"))
    out += _lines(final_answer)
    out.append("")
    out.append("-" * W)
    out.append("")
    return "\n".join(out) + "\n"


def write_token_log(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> None:
    """Append one token-usage line to logs/cli/YYYY-MM.log."""
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts}  model={model}"
        f"  input={prompt_tokens}"
        f"  cache={cached_tokens}"
        f"  output={completion_tokens}"
        f"  total={prompt_tokens + completion_tokens}\n"
    )
    try:
        cli_dir = LOGS_DIR / "cli"
        cli_dir.mkdir(parents=True, exist_ok=True)
        log_file = cli_dir / f"{now.strftime('%Y-%m')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        log.warning("token log write failed: %s", e)


def write_trace(user_id: str, question: str, trace: list, final_answer: str) -> None:
    uid = user_id or "unknown"
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    pretty = _render_pretty(uid, question, trace, final_answer, ts)

    # 1. Pretty local file (best-effort)
    try:
        user_dir = LOGS_DIR / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        log_file = user_dir / f"{now.strftime('%Y-%m')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(pretty)
    except Exception as e:
        log.warning("local trace write failed (%s) — continuing with JSON log only", e)

    # 1b. Pretty format to stdout — captured by Cloud Logging
    print(pretty, flush=True)

    # 2. Structured JSON to stdout for Cloud Logging
    record = {
        "type": "trace",
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": uid,
        "question": question,
        "trace": trace,
        "final_answer": final_answer,
    }
    try:
        log.info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        log.warning("json trace emit failed: %s", e)

"""Unit tests for the skill-based hospital chatbot agent."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _mock_llm(json_response: str):
    """Return a llm_fn that always returns json_response."""
    def _fn(messages, *, use_router=False, **kw):
        return json_response
    return _fn


# ---------------------------------------------------------------------------
# Skill 1 — classify_question
# ---------------------------------------------------------------------------

class TestClassifyQuestion:
    def setup_method(self):
        from app.skills import classify_question
        self.classify = classify_question

    def test_treatment_route(self):
        llm_fn = _mock_llm('{"route": "treatment", "files": [3]}')
        result = self.classify("ขูดหินปูนราคาเท่าไร", [], llm_fn)
        assert result["route"] == "treatment"
        assert result["files"] == [3]
        assert result["trace_entry"]["node"] == "classify"

    def test_general_route(self):
        llm_fn = _mock_llm('{"route": "general", "files": []}')
        result = self.classify("โรงพยาบาลเปิดกี่โมง", [], llm_fn)
        assert result["route"] == "general"
        assert result["files"] == []

    def test_smalltalk_route(self):
        llm_fn = _mock_llm('{"route": "smalltalk", "files": []}')
        result = self.classify("สวัสดีครับ", [], llm_fn)
        assert result["route"] == "smalltalk"
        assert result["files"] == []

    def test_off_topic_route(self):
        llm_fn = _mock_llm('{"route": "off_topic", "files": []}')
        result = self.classify("บอลเมื่อคืนใครชนะ", [], llm_fn)
        assert result["route"] == "off_topic"

    def test_invalid_json_defaults_to_general(self):
        llm_fn = _mock_llm("not valid json at all")
        result = self.classify("some question", [], llm_fn)
        assert result["route"] == "general"
        assert result["files"] == []

    def test_invalid_route_defaults_to_general(self):
        llm_fn = _mock_llm('{"route": "unknown_route", "files": []}')
        result = self.classify("some question", [], llm_fn)
        assert result["route"] == "general"

    def test_files_out_of_range_filtered(self):
        # files 12 and 15 are out of 1-11 range
        llm_fn = _mock_llm('{"route": "treatment", "files": [3, 12, 15]}')
        result = self.classify("question", [], llm_fn)
        assert 12 not in result["files"]
        assert 15 not in result["files"]
        assert 3 in result["files"]

    def test_llm_raises_defaults_to_general(self):
        def bad_llm(messages, **kw):
            raise RuntimeError("API error")
        result = self.classify("question", [], bad_llm)
        assert result["route"] == "general"
        assert result["files"] == []

    def test_markdown_fenced_json(self):
        llm_fn = _mock_llm('```json\n{"route": "treatment", "files": [1]}\n```')
        result = self.classify("question", [], llm_fn)
        assert result["route"] == "treatment"
        assert result["files"] == [1]

    def test_multiple_treatment_files(self):
        llm_fn = _mock_llm('{"route": "treatment", "files": [1, 3, 10]}')
        result = self.classify("ขูดหินปูน อุดฟัน และรากเทียม", [], llm_fn)
        assert set(result["files"]) == {1, 3, 10}

    def test_trace_entry_structure(self):
        llm_fn = _mock_llm('{"route": "general", "files": []}')
        result = self.classify("question", [], llm_fn)
        assert "node" in result["trace_entry"]
        assert "route" in result["trace_entry"]
        assert "files" in result["trace_entry"]


# ---------------------------------------------------------------------------
# Skill 2/3 — load_files
# ---------------------------------------------------------------------------

class TestLoadFiles:
    def setup_method(self):
        from app.skills import load_files
        self.load_files = load_files

    def test_always_includes_file_12(self):
        result = self.load_files([])
        assert 12 in result["content"] or any(
            f["file"] == 12 for f in result["trace_entry"]["files"]
        )

    def test_loads_real_file_12(self):
        result = self.load_files([])
        if (RAW_DIR / "12.ราคาและข้อมูลทั่วไปเกี่ยวกับโรงพยาบาล.docx").exists():
            assert 12 in result["content"]
            assert len(result["content"][12]) > 50

    def test_loads_treatment_file(self):
        # Try loading file 1 if it exists
        result = self.load_files([1])
        trace_files = {f["file"]: f for f in result["trace_entry"]["files"]}
        assert 1 in trace_files
        assert 12 in trace_files

    def test_missing_file_not_in_content(self):
        # File 99 doesn't exist
        result = self.load_files([99])
        assert 99 not in result["content"]

    def test_has_data_true_when_files_loaded(self):
        result = self.load_files([])
        if (RAW_DIR / "12.ราคาและข้อมูลทั่วไปเกี่ยวกับโรงพยาบาล.docx").exists():
            assert result["has_data"] is True

    def test_deduplication(self):
        result = self.load_files([12, 12, 12])
        # Should only read file 12 once
        file_nums = [f["file"] for f in result["trace_entry"]["files"]]
        assert file_nums.count(12) == 1

    def test_trace_entry_structure(self):
        result = self.load_files([])
        entry = result["trace_entry"]
        assert entry["node"] == "read_files"
        assert "files" in entry
        assert "has_data" in entry


# ---------------------------------------------------------------------------
# Skill 4 — format_answer
# ---------------------------------------------------------------------------

class TestFormatAnswer:
    def setup_method(self):
        from app.skills import format_answer
        self.format_answer = format_answer

    def test_returns_llm_answer(self):
        llm_fn = _mock_llm("ค่ารักษาขูดหินปูนเริ่มต้นที่ 500 บาทค่ะ")
        result = self.format_answer(
            "ขูดหินปูนราคาเท่าไร",
            {1: "ขูดหินปูน...", 12: "ราคา 500 บาท"},
            [],
            llm_fn,
        )
        assert "500" in result

    def test_fallback_on_llm_error(self):
        def bad_llm(messages, **kw):
            raise RuntimeError("API down")
        result = self.format_answer("question", {12: "some content"}, [], bad_llm)
        # Should return a fallback string, not raise
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_content_uses_fallback(self):
        def bad_llm(messages, **kw):
            raise RuntimeError("no content")
        result = self.format_answer("question", {}, [], bad_llm)
        assert isinstance(result, str)

    def test_uses_history(self):
        captured = []
        def capture_llm(messages, **kw):
            captured.extend(messages)
            return "answer"
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        self.format_answer("new question", {12: "content"}, history, capture_llm)
        roles = [m["role"] for m in captured]
        assert "user" in roles


# ---------------------------------------------------------------------------
# Graph-level integration (mocked LLM)
# ---------------------------------------------------------------------------

class TestGraphAsk:
    def test_ask_returns_string(self):
        """ask() always returns a string, never raises."""
        from app.graph import ask
        with patch("app.agents._client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = '{"route": "general", "files": []}'
            mock_client.chat.completions.create.return_value = mock_resp
            result = ask("โรงพยาบาลเปิดกี่โมง", user_id="test_user")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ask_off_topic_no_llm_answer(self):
        """off_topic route returns fixed reply without calling ANSWER_MODEL."""
        from app.graph import ask
        call_count = [0]
        def mock_create(**kw):
            call_count[0] += 1
            resp = MagicMock()
            resp.choices[0].message.content = '{"route": "off_topic", "files": []}'
            return resp
        with patch("app.agents._client") as mock_client:
            mock_client.chat.completions.create.side_effect = mock_create
            result = ask("who won the football game", user_id="test_off")
        assert isinstance(result, str)
        # Only 1 LLM call (the classifier), not 2
        assert call_count[0] == 1

    def test_ask_fallback_on_total_failure(self):
        """ask() returns a non-empty fallback string when graph crashes."""
        from app.graph import ask
        with patch("app.graph._graph") as mock_graph:
            mock_graph.invoke.side_effect = RuntimeError("complete failure")
            result = ask("anything", user_id="crash_user")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Language-aware static replies
# ---------------------------------------------------------------------------

class TestLanguageAwareReplies:
    def test_off_topic_thai(self):
        from app.agents import off_topic
        reply = off_topic("ใครชนะบอลเมื่อคืน")
        assert isinstance(reply, str)
        # Must be Thai-only — no English words
        assert "Sorry" not in reply
        assert "Hospital" not in reply

    def test_off_topic_english(self):
        from app.agents import off_topic
        reply = off_topic("who won the game last night")
        assert isinstance(reply, str)
        # Must be English-only — no Thai characters
        assert not any("฀" <= ch <= "๿" for ch in reply)

    def test_off_topic_empty_defaults(self):
        from app.agents import off_topic
        reply = off_topic()
        assert isinstance(reply, str) and len(reply) > 0

    def test_no_data_thai(self):
        from app.agents import no_data
        reply = no_data("ขูดหินปูนราคาเท่าไร")
        assert isinstance(reply, str)
        assert "Sorry" not in reply

    def test_no_data_english(self):
        from app.agents import no_data
        reply = no_data("how much does scaling cost")
        assert isinstance(reply, str)
        assert not any("฀" <= ch <= "๿" for ch in reply)

    def test_fallback_thai(self):
        from app.agents import fallback_reply
        reply = fallback_reply("ราคาเท่าไร")
        assert "Sorry" not in reply

    def test_fallback_english(self):
        from app.agents import fallback_reply
        reply = fallback_reply("what is the price")
        assert not any("฀" <= ch <= "๿" for ch in reply)

    def test_is_thai_detects_thai(self):
        from app.agents import _is_thai
        assert _is_thai("สวัสดี") is True
        assert _is_thai("hello") is False
        assert _is_thai("hello สวัสดี") is True
        assert _is_thai("") is False


# ---------------------------------------------------------------------------
# _find_docx helper
# ---------------------------------------------------------------------------

class TestFindDocx:
    def test_finds_file_1(self):
        from app.skills import _find_docx
        p = _find_docx(1)
        if (RAW_DIR).exists():
            assert p is not None
            assert p.name.startswith("1.")

    def test_finds_file_12(self):
        from app.skills import _find_docx
        p = _find_docx(12)
        if (RAW_DIR).exists():
            assert p is not None

    def test_missing_file_returns_none(self):
        from app.skills import _find_docx
        p = _find_docx(999)
        assert p is None

    def test_skips_lock_files(self):
        from app.skills import _find_docx
        p = _find_docx(11)
        if p is not None:
            assert not p.name.startswith("~$")

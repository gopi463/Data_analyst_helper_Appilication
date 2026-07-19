"""
tests/test_llm.py — Unit Tests for llm.py (extraction helpers + history builder)
AI Data Analyst Assistant
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import extract_sql, extract_python_code, _build_message_history


# ── extract_sql Tests ─────────────────────────────────────────────────────────
class TestExtractSQL:
    def test_extracts_sql_from_code_block(self):
        response = "Here is the query:\n```sql\nSELECT * FROM data_table LIMIT 10;\n```"
        result = extract_sql(response)
        assert result == "SELECT * FROM data_table LIMIT 10;"

    def test_extracts_sql_case_insensitive(self):
        response = "```SQL\nSELECT id FROM data_table;\n```"
        result = extract_sql(response)
        assert result == "SELECT id FROM data_table;"

    def test_fallback_to_select_statement(self):
        response = "You can use SELECT name FROM data_table WHERE id = 1;"
        result = extract_sql(response)
        assert result is not None
        assert "SELECT" in result

    def test_no_sql_returns_none(self):
        result = extract_sql("There is no SQL here.")
        assert result is None

    def test_multiline_sql_extracted(self):
        response = "```sql\nSELECT name,\n       age\nFROM data_table;\n```"
        result = extract_sql(response)
        assert "SELECT" in result
        assert "data_table" in result


# ── extract_python_code Tests ─────────────────────────────────────────────────
class TestExtractPythonCode:
    def test_extracts_python_from_code_block(self):
        response = "```python\nresult = df['sales'].sum()\n```"
        result = extract_python_code(response)
        assert result == "result = df['sales'].sum()"

    def test_no_code_returns_none(self):
        result = extract_python_code("No code here.")
        assert result is None

    def test_multiline_code_extracted(self):
        response = "```python\ndf_filtered = df[df['age'] > 30]\nresult = df_filtered.mean()\n```"
        result = extract_python_code(response)
        assert "df_filtered" in result
        assert "result" in result

    def test_extracts_first_block_only(self):
        response = "```python\nresult = 1\n```\nSome text\n```python\nresult = 2\n```"
        result = extract_python_code(response)
        assert "result = 1" in result


# ── _build_message_history Tests ─────────────────────────────────────────────
class TestBuildMessageHistory:
    def test_empty_history_returns_empty(self):
        assert _build_message_history([]) == []

    def test_none_history_returns_empty(self):
        assert _build_message_history(None) == []

    def test_filters_out_non_standard_roles(self):
        history = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _build_message_history(history)
        roles = [m["role"] for m in result]
        assert "system" not in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_respects_max_exchanges_limit(self):
        # 10 messages = 5 exchanges, max_exchanges=2 → should return last 4
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = _build_message_history(history, max_exchanges=2)
        assert len(result) <= 4

    def test_content_truncated_to_2000_chars(self):
        history = [{"role": "user", "content": "x" * 5000}]
        result = _build_message_history(history)
        assert len(result[0]["content"]) <= 2000

    def test_sources_field_not_in_output(self):
        history = [{"role": "user", "content": "hello", "sources": ["doc1.pdf"]}]
        result = _build_message_history(history)
        assert "sources" not in result[0]

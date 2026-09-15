"""
Unit tests for modular research tools:
- Calculator (safe AST evaluator, error handling, syntax & zero-division guards)
- Web search (query validation, fallback handling)
- Document search (local retrieval, scoring)
- DateTime tool (UTC & local temporal grounding)
"""
import pytest
from src.tools.calculator import calculate
from src.tools.datetime_tool import get_current_datetime
from src.tools.document_search import search_documents
from src.tools.web_search import search_web
from src.tools import execute_tool

def test_calculator_arithmetic():
    res = calculate("2 + 2 * 10")
    assert res["status"] == "success"
    assert res["result"] == 22
    assert res["error"] is None

def test_calculator_math_functions():
    res = calculate("sqrt(144) + abs(-10)")
    assert res["status"] == "success"
    assert res["result"] == 22.0

def test_calculator_division_by_zero():
    res = calculate("100 / 0")
    assert res["status"] == "error"
    assert "Division by zero" in res["error"]

def test_calculator_security_code_injection():
    # Attempting to call unauthorized functions or code
    res = calculate("__import__('os').system('ls')")
    assert res["status"] == "error"

def test_datetime_tool():
    res = get_current_datetime()
    assert res["status"] == "success"
    assert "current_year" in res
    assert res["current_year"] >= 2025
    assert "utc_now" in res

def test_document_search():
    results = search_documents("LangGraph stateful workflow")
    assert len(results) > 0
    first = results[0]
    assert "source_title" in first
    assert "source_url" in first
    assert "excerpt" in first
    assert first["relevance_score"] > 0.0

def test_web_search_fallback():
    results = search_web("Quantum computing benchmarks")
    assert len(results) > 0
    assert "source_url" in results[0]
    assert "excerpt" in results[0]

def test_execute_tool_dispatcher():
    calc_res = execute_tool("calculator", {"expression": "50 * 2"})
    assert calc_res["status"] == "success"
    assert calc_res["result"] == 100

    unknown_res = execute_tool("non_existent_tool", {})
    assert unknown_res["status"] == "error"

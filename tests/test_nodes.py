"""
Unit tests for individual graph nodes in isolation.
"""
from src.nodes import (
    analyze_evidence,
    analyze_query,
    call_tools,
    collect_evidence,
    create_plan,
    evaluate_confidence,
    execute_research,
    fact_check,
    generate_final_answer,
    human_review,
)
from src.state import create_initial_state

def test_analyze_query_node():
    state = create_initial_state("How does LangGraph implement stateful workflows?")
    update = analyze_query(state)
    assert "analysis" in update
    assert "messages" in update

def test_create_plan_node():
    state = create_initial_state("What is quantum computing?")
    state["analysis"] = "Domain: Quantum Physics"
    update = create_plan(state)
    assert "research_plan" in update
    assert len(update["research_plan"]) >= 2
    assert "tool" in update["research_plan"][0]

def test_call_tools_and_collect_evidence_nodes():
    state = create_initial_state("Calculate 25 * 4")
    state["research_plan"] = [
        {"step_id": 1, "task": "Calculate product", "tool": "calculator", "query": "25 * 4", "status": "pending"}
    ]
    tool_update = call_tools(state)
    state.update(tool_update)

    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0]["status"] == "success"

    evidence_update = collect_evidence(state)
    state.update(evidence_update)
    assert len(state["evidence"]) == 1
    assert "Calculation" in state["evidence"][0]["source_title"]

def test_evaluate_confidence_node():
    state = create_initial_state("Test Query")
    state["fact_check_results"] = {"confidence_score": 0.85, "has_contradictions": False}
    state["evidence"] = [
        {"source_title": "Source 1", "source_url": "url1", "excerpt": "text1", "relevance_score": 0.9},
        {"source_title": "Source 2", "source_url": "url2", "excerpt": "text2", "relevance_score": 0.9},
        {"source_title": "Source 3", "source_url": "url3", "excerpt": "text3", "relevance_score": 0.9}
    ]
    state["iteration_count"] = 0
    update = evaluate_confidence(state)
    assert update["confidence_score"] >= 0.80
    assert update["iteration_count"] == 1

def test_generate_final_answer_node():
    state = create_initial_state("What is LangGraph?")
    state["analysis"] = "LangGraph enables stateful graph agents."
    state["confidence_score"] = 0.92
    state["evidence"] = [
        {"source_title": "LangGraph Doc", "source_url": "file://local/doc.txt", "excerpt": "StateGraph core primitives", "relevance_score": 0.9}
    ]
    update = generate_final_answer(state)
    assert "final_answer" in update
    final_text = update["final_answer"]
    assert "# Answer" in final_text
    assert "## Key Findings" in final_text
    assert "## Evidence" in final_text
    assert "## Sources" in final_text
    assert "## Confidence" in final_text
    assert "## Limitations" in final_text

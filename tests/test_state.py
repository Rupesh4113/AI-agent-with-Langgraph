"""
Unit tests for state management and TypedDict schema.
"""
from src.state import AgentState, create_initial_state

def test_create_initial_state_defaults():
    state = create_initial_state(user_query="What is LangGraph?", conversation_id="test_thread_1")
    
    assert state["user_query"] == "What is LangGraph?"
    assert state["conversation_id"] == "test_thread_1"
    assert state["research_plan"] == []
    assert state["current_step"] == 0
    assert state["research_results"] == []
    assert state["evidence"] == []
    assert state["analysis"] == ""
    assert state["fact_check_results"] == {}
    assert state["confidence_score"] == 0.0
    assert state["tool_calls"] == []
    assert state["errors"] == []
    assert state["human_feedback"] is None
    assert state["final_answer"] is None
    assert state["iteration_count"] == 0
    assert state["messages"] == []

def test_state_sanitization_on_creation():
    state = create_initial_state(user_query="   Whitespace test query   \n", conversation_id="test_thread_2")
    assert state["user_query"] == "Whitespace test query"

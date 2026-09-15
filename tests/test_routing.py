"""
Unit tests for conditional edge routing and cycles.
"""
from src.routing import route_after_confidence, route_after_human_review
from src.state import create_initial_state

def test_route_confidence_sufficient():
    state = create_initial_state("test query")
    state["confidence_score"] = 0.85
    state["iteration_count"] = 1
    
    next_node = route_after_confidence(state)
    assert next_node == "human_review"

def test_route_confidence_insufficient_cycles_back():
    state = create_initial_state("test query")
    state["confidence_score"] = 0.65
    state["iteration_count"] = 1  # Less than MAX_ITERATIONS (3)
    
    next_node = route_after_confidence(state)
    assert next_node == "execute_research"

def test_route_confidence_max_iterations_stops_cycle():
    state = create_initial_state("test query")
    state["confidence_score"] = 0.50
    state["iteration_count"] = 3  # Reached max iterations
    
    next_node = route_after_confidence(state)
    assert next_node == "human_review"

def test_route_human_review_approval():
    state = create_initial_state("test query")
    state["human_feedback"] = None
    next_node = route_after_human_review(state)
    assert next_node == "generate_final_answer"

def test_route_human_review_request_more_research():
    state = create_initial_state("test query")
    state["human_feedback"] = "Request more research on logical qubits"
    next_node = route_after_human_review(state)
    assert next_node == "execute_research"

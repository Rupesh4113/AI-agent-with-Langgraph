"""
Integration tests for end-to-end LangGraph StateGraph execution,
persistence, cycles, and human-in-the-loop checkpoints.
"""
from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_research_graph, get_mermaid_diagram
from src.state import create_initial_state

def test_mermaid_diagram_generation():
    diagram = get_mermaid_diagram()
    assert "flowchart TD" in diagram
    assert "analyze_query" in diagram
    assert "human_review" in diagram
    assert "generate_final_answer" in diagram

def test_full_graph_execution_pipeline():
    # Build graph with fresh in-memory checkpointer and no pause
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer, interrupt_on_human_review=False)

    thread_id = "integration_test_thread_1"
    initial_state = create_initial_state(
        user_query="How does LangGraph coordinate state across cyclic research iterations?",
        conversation_id=thread_id
    )
    config = {"configurable": {"thread_id": thread_id}}

    # Execute full workflow
    final_state = graph.invoke(initial_state, config=config)

    # Assertions on final state
    assert final_state is not None
    assert final_state["final_answer"] is not None
    assert "# Answer" in final_state["final_answer"]
    assert "## Sources" in final_state["final_answer"]
    assert len(final_state["evidence"]) > 0
    assert len(final_state["tool_calls"]) > 0
    assert final_state["confidence_score"] > 0.0
    assert final_state["iteration_count"] >= 1

def test_human_in_the_loop_interrupt_and_resume():
    # Build graph with interrupt on human review
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer, interrupt_on_human_review=True)

    thread_id = "hitl_test_thread"
    initial_state = create_initial_state(
        user_query="Explain fault tolerance in quantum computing.",
        conversation_id=thread_id
    )
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Run until human_review interrupt
    state_at_interrupt = graph.invoke(initial_state, config=config)
    
    # At this point, final_answer should still be None because human review has not approved
    assert state_at_interrupt.get("final_answer") is None
    assert len(state_at_interrupt["evidence"]) > 0

    # 2. Simulate user approving and resuming execution
    # Resume execution with None as input to continue from checkpoint
    resumed_state = graph.invoke(None, config=config)

    assert resumed_state["final_answer"] is not None
    assert "# Answer" in resumed_state["final_answer"]
    assert "## Confidence" in resumed_state["final_answer"]

"""
LangGraph StateGraph assembly for the AI Research Agent.
Constructs nodes, edges, conditional routing, cyclic self-correction,
and human-in-the-loop checkpoint compilation.
"""
from typing import Any, Dict, Generator, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.memory.checkpoint import get_checkpointer
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
from src.routing import route_after_confidence, route_after_human_review
from src.state import AgentState
from src.utils.logging import get_logger

logger = get_logger("graph")

def build_research_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_on_human_review: bool = False
):
    """
    Constructs and compiles the LangGraph StateGraph.
    
    Args:
        checkpointer: LangGraph checkpoint saver (MemorySaver or SqliteSaver).
        interrupt_on_human_review: If True, adds interrupt_before=["human_review"]
                                   enabling true human-in-the-loop pauses.
                                   
    Returns:
        Compiled StateGraph instance ready for invoke or stream execution.
    """
    workflow = StateGraph(AgentState)

    # 1. Add all 10 dedicated nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_research", execute_research)
    workflow.add_node("call_tools", call_tools)
    workflow.add_node("collect_evidence", collect_evidence)
    workflow.add_node("analyze_evidence", analyze_evidence)
    workflow.add_node("fact_check", fact_check)
    workflow.add_node("evaluate_confidence", evaluate_confidence)
    workflow.add_node("human_review", human_review)
    workflow.add_node("generate_final_answer", generate_final_answer)

    # 2. Add sequential linear edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "create_plan")
    workflow.add_edge("create_plan", "execute_research")
    workflow.add_edge("execute_research", "call_tools")
    workflow.add_edge("call_tools", "collect_evidence")
    workflow.add_edge("collect_evidence", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "fact_check")
    workflow.add_edge("fact_check", "evaluate_confidence")

    # 3. Add conditional edge from evaluate_confidence
    #    Routes to 'execute_research' (cycle) or 'human_review'
    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {
            "execute_research": "execute_research",
            "human_review": "human_review",
            "generate_final_answer": "generate_final_answer"
        }
    )

    # 4. Add conditional edge from human_review
    #    Routes back to 'execute_research' if user requests revision, or 'generate_final_answer'
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "execute_research": "execute_research",
            "generate_final_answer": "generate_final_answer"
        }
    )

    # 5. Connect final node to END
    workflow.add_edge("generate_final_answer", END)

    # Compile with checkpointer and optional human interrupt
    saver = checkpointer if checkpointer is not None else get_checkpointer()
    interrupts = ["human_review"] if interrupt_on_human_review else []

    compiled = workflow.compile(
        checkpointer=saver,
        interrupt_before=interrupts
    )

    logger.info(f"Graph compiled successfully (interrupts={interrupts}).")
    return compiled


def get_mermaid_diagram() -> str:
    """
    Returns the Mermaid flowchart representation of the research agent workflow.
    """
    return """```mermaid
flowchart TD
    START([User Question]) --> analyze_query[1. Query Analyzer]
    analyze_query --> create_plan[2. Planner]
    create_plan --> execute_research[3. Execute Research]
    
    subgraph Tooling Layer
        execute_research --> call_tools[4. Tool Execution]
        call_tools -.-> WebSearch[Web Search Tool]
        call_tools -.-> Calculator[Calculator Tool]
        call_tools -.-> DocSearch[Document Search]
        call_tools -.-> DateTime[DateTime Tool]
    end

    call_tools --> collect_evidence[5. Evidence Collector]
    collect_evidence --> analyze_evidence[6. Evidence Synthesis]
    analyze_evidence --> fact_check[7. Fact Checker]
    fact_check --> evaluate_confidence[8. Confidence Evaluator]

    evaluate_confidence -->|Confidence < 0.80 & Iteration < 3| execute_research
    evaluate_confidence -->|Confidence >= 0.80 or Max Iterations| human_review{9. Human Review}

    human_review -->|Request More Research| execute_research
    human_review -->|Approve| generate_final_answer[10. Final Response]

    generate_final_answer --> END([Structured Answer + Citations + Memory])

    style START fill:#4B5563,stroke:#9CA3AF,stroke-width:2px,color:#FFF
    style END fill:#10B981,stroke:#059669,stroke-width:2px,color:#FFF
    style human_review fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#FFF
    style execute_research fill:#3B82F6,stroke:#2563EB,stroke-width:2px,color:#FFF
    style fact_check fill:#8B5CF6,stroke:#7C3AED,stroke-width:2px,color:#FFF
```"""

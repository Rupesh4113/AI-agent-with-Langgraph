"""
Routing logic and conditional edge functions for LangGraph AI Research Agent.
Implements dynamic routing based on confidence score, iteration limits, and human review decisions.
"""
from typing import Literal
from src.config import settings
from src.state import AgentState
from src.utils.logging import get_logger

logger = get_logger("routing")

def route_after_confidence(state: AgentState) -> Literal["human_review", "execute_research", "generate_final_answer"]:
    """
    Evaluates confidence score and iteration count to determine next node:
    - If confidence >= threshold: proceed to human_review
    - If confidence < threshold and iterations < max_iterations: cycle back to execute_research
    - If iterations >= max_iterations: proceed to human_review with uncertainty note
    """
    confidence = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)
    threshold = settings.MIN_CONFIDENCE_THRESHOLD
    max_iter = settings.MAX_RESEARCH_ITERATIONS

    logger.info(
        f"Routing Decision: confidence={confidence:.2f} (threshold={threshold:.2f}), "
        f"iteration={iterations}/{max_iter}"
    )

    if confidence >= threshold:
        logger.info("Confidence sufficient. Routing to 'human_review'.")
        return "human_review"

    if iterations < max_iter:
        logger.info(f"Confidence below threshold. Cycling back to 'execute_research' (iteration {iterations + 1}).")
        return "execute_research"

    logger.warning("Max research iterations reached. Proceeding to 'human_review' with uncertainty warning.")
    return "human_review"


def route_after_human_review(state: AgentState) -> Literal["generate_final_answer", "execute_research"]:
    """
    Routes based on human feedback:
    - If user requested more research / new direction: cycle back to execute_research
    - Otherwise: proceed to generate_final_answer
    """
    feedback = state.get("human_feedback")
    if feedback and feedback.strip().lower().startswith(("request more", "retry", "re-research", "investigate", "more research")):
        logger.info(f"Human requested additional research ('{feedback}'). Routing to 'execute_research'.")
        return "execute_research"

    logger.info("Human review approved or passed. Routing to 'generate_final_answer'.")
    return "generate_final_answer"

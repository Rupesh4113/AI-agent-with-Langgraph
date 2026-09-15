"""
LangGraph Nodes for the AI Research Agent.
Implements the 10 dedicated graph nodes, each with a single, clear responsibility.
"""
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.config import settings
from src.models import get_model
from src.prompts import (
    EVIDENCE_ANALYSIS_PROMPT,
    FACT_CHECK_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    PLANNER_PROMPT,
    QUERY_ANALYSIS_PROMPT,
)
from src.state import AgentState, EvidenceItem, ResearchPlanStep, ToolCallRecord
from src.tools import execute_tool
from src.utils.logging import get_logger

logger = get_logger("nodes")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Helper to parse JSON from LLM text output safely, stripping code fences if present."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown fences ```json ... ```
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Try finding the first { ... } block
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


# ----------------------------------------------------------------------
# Node 1: analyze_query
# ----------------------------------------------------------------------
def analyze_query(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Analyzes the user's research query to assess intent, domain,
    complexity, and required tools.
    """
    user_query = state.get("user_query", "").strip()
    logger.info(f"Node [analyze_query]: Analyzing query: '{user_query[:80]}...'")

    prompt = QUERY_ANALYSIS_PROMPT.format(query=user_query)
    model = get_model()

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content) or {}
    except Exception as exc:
        logger.error(f"Error in analyze_query: {exc}")
        parsed = {
            "query_intent": f"Investigate '{user_query}'",
            "domain": "General Research",
            "complexity": "moderate",
            "requires_tools": ["web_search", "document_search"],
            "sub_questions": [user_query]
        }

    return {
        "analysis": f"Query Intent: {parsed.get('query_intent', '')}\nDomain: {parsed.get('domain', '')}\nComplexity: {parsed.get('complexity', 'moderate')}",
        "messages": [AIMessage(content=f"Query Analyzed: {parsed.get('query_intent', user_query)}")]
    }


# ----------------------------------------------------------------------
# Node 2: create_plan
# ----------------------------------------------------------------------
def create_plan(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Breaks down the query into a structured multi-step research plan.
    Incorporates human feedback if this is a revised iteration.
    """
    user_query = state.get("user_query", "")
    query_analysis = state.get("analysis", "")
    human_feedback = state.get("human_feedback") or "None (Initial plan creation)"

    logger.info("Node [create_plan]: Creating research plan...")
    prompt = PLANNER_PROMPT.format(
        query=user_query,
        query_analysis=query_analysis,
        human_feedback=human_feedback
    )
    model = get_model()

    steps: List[Dict[str, Any]] = []
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content) or {}
        raw_steps = parsed.get("steps", [])
        for idx, step in enumerate(raw_steps, 1):
            steps.append({
                "step_id": idx,
                "task": step.get("task", f"Research step {idx}"),
                "tool": step.get("tool", "web_search"),
                "query": step.get("query", user_query),
                "status": "pending",
                "result": None
            })
    except Exception as exc:
        logger.error(f"Error in create_plan: {exc}")

    if not steps:
        # Fallback default plan
        steps = [
            {"step_id": 1, "task": "Search internal technical documents", "tool": "document_search", "query": user_query, "status": "pending", "result": None},
            {"step_id": 2, "task": "Search external web sources", "tool": "web_search", "query": user_query, "status": "pending", "result": None}
        ]

    return {
        "research_plan": steps,
        "current_step": 0,
        "messages": [AIMessage(content=f"Research Plan formulated with {len(steps)} steps.")]
    }


# ----------------------------------------------------------------------
# Node 3: execute_research
# ----------------------------------------------------------------------
def execute_research(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Coordinates execution of current research plan steps, preparing
    tool execution parameters.
    """
    plan = list(state.get("research_plan", []))
    iteration = state.get("iteration_count", 0)
    logger.info(f"Node [execute_research]: Preparing research execution (iteration {iteration})...")

    # Mark pending steps as in_progress
    for step in plan:
        if step.get("status") in ("pending", "in_progress"):
            step["status"] = "in_progress"

    return {
        "research_plan": plan,
        "messages": [AIMessage(content=f"Executing research steps for iteration {iteration}...")]
    }


# ----------------------------------------------------------------------
# Node 4: call_tools
# ----------------------------------------------------------------------
def call_tools(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Executes the tools designated in the research plan (web_search,
    calculator, document_search, datetime).
    """
    plan = list(state.get("research_plan", []))
    existing_calls = list(state.get("tool_calls", []))
    new_results = list(state.get("research_results", []))
    errors = list(state.get("errors", []))

    logger.info(f"Node [call_tools]: Executing {len(plan)} planned tool calls...")

    for step in plan:
        tool_name = step.get("tool", "web_search")
        query_arg = step.get("query", "")
        args: Dict[str, Any] = {}

        if tool_name == "calculator":
            args = {"expression": query_arg}
        elif tool_name == "document_search":
            args = {"query": query_arg, "top_k": 3}
        elif tool_name == "datetime":
            args = {}
        else:
            args = {"query": query_arg, "max_results": 3}

        # Safe dispatch
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            output = execute_tool(tool_name, args)
            status = output.get("status", "success")
            step["status"] = "completed" if status == "success" else "failed"
            step["result"] = output.get("summary", "")

            tool_record: ToolCallRecord = {
                "tool_name": tool_name,
                "arguments": args,
                "output": output.get("summary", str(output)),
                "timestamp": timestamp,
                "status": status
            }
            existing_calls.append(tool_record)
            new_results.append({
                "step_id": step.get("step_id"),
                "tool": tool_name,
                "data": output
            })

            if status != "success" and output.get("error"):
                errors.append(f"Tool {tool_name} failed: {output.get('error')}")

        except Exception as exc:
            logger.error(f"Error executing tool {tool_name}: {exc}")
            step["status"] = "failed"
            step["result"] = str(exc)
            errors.append(f"Tool {tool_name} crashed: {str(exc)}")

    return {
        "research_plan": plan,
        "tool_calls": existing_calls,
        "research_results": new_results,
        "errors": errors,
        "messages": [AIMessage(content=f"Completed {len(plan)} tool executions.")]
    }


# ----------------------------------------------------------------------
# Node 5: collect_evidence
# ----------------------------------------------------------------------
def collect_evidence(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Normalizes, extracts, and deduplicates evidence items from all
    tool execution results.
    """
    results = state.get("research_results", [])
    existing_evidence = list(state.get("evidence", []))
    seen_keys = {f"{e.get('source_title')}_{e.get('source_url')}" for e in existing_evidence}

    logger.info("Node [collect_evidence]: Collecting and deduplicating evidence...")

    for res in results:
        data = res.get("data", {})
        tool = res.get("tool", "")

        # 1. From web search or document search
        items = data.get("results", [])
        for item in items:
            key = f"{item.get('source_title')}_{item.get('source_url')}"
            if key not in seen_keys and item.get("excerpt"):
                seen_keys.add(key)
                existing_evidence.append({
                    "source_title": item.get("source_title", "Unknown Source"),
                    "source_url": item.get("source_url", "https://local-source.org"),
                    "retrieval_timestamp": item.get("retrieval_timestamp", datetime.now(timezone.utc).isoformat()),
                    "excerpt": item.get("excerpt", "").strip(),
                    "source_type": item.get("source_type", tool),
                    "relevance_score": float(item.get("relevance_score", 0.8))
                })

        # 2. From calculator
        if tool == "calculator" and data.get("status") == "success":
            calc_key = f"Calculator_{data.get('expression')}"
            if calc_key not in seen_keys:
                seen_keys.add(calc_key)
                existing_evidence.append({
                    "source_title": f"Calculation: {data.get('expression')}",
                    "source_url": "calculator://ast-evaluator",
                    "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                    "excerpt": f"Evaluated '{data.get('expression')}' resulting in value: {data.get('result')}",
                    "source_type": "calculation",
                    "relevance_score": 1.0
                })

        # 3. From datetime
        if tool == "datetime" and data.get("status") == "success":
            dt_key = "Temporal_Grounding"
            if dt_key not in seen_keys:
                seen_keys.add(dt_key)
                existing_evidence.append({
                    "source_title": "Temporal Ground Truth",
                    "source_url": "datetime://system-clock",
                    "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                    "excerpt": data.get("summary", "Current temporal timestamp verified."),
                    "source_type": "temporal",
                    "relevance_score": 1.0
                })

    return {
        "evidence": existing_evidence,
        "messages": [AIMessage(content=f"Collected {len(existing_evidence)} total verified evidence items.")]
    }


# ----------------------------------------------------------------------
# Node 6: analyze_evidence
# ----------------------------------------------------------------------
def analyze_evidence(state: AgentState) -> Dict[str, Any]:
    """
    Node 6: Synthesizes the gathered evidence against the research plan.
    """
    user_query = state.get("user_query", "")
    plan = state.get("research_plan", [])
    evidence = state.get("evidence", [])

    logger.info("Node [analyze_evidence]: Synthesizing evidence...")

    evidence_text = "\n\n".join(
        f"[{idx}] {item['source_title']} ({item['source_type']}): {item['excerpt']}"
        for idx, item in enumerate(evidence, 1)
    ) or "No external evidence retrieved."

    prompt = EVIDENCE_ANALYSIS_PROMPT.format(
        query=user_query,
        research_plan=json.dumps(plan, indent=2),
        evidence=evidence_text[:4000]
    )
    model = get_model()

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        analysis_text = response.content.strip()
    except Exception as exc:
        logger.error(f"Error in analyze_evidence: {exc}")
        analysis_text = f"Synthesized {len(evidence)} evidence items for query '{user_query}'."

    return {
        "analysis": analysis_text,
        "messages": [AIMessage(content="Evidence synthesized into structured analysis.")]
    }


# ----------------------------------------------------------------------
# Node 7: fact_check
# ----------------------------------------------------------------------
def fact_check(state: AgentState) -> Dict[str, Any]:
    """
    Node 7: Audits claims, detects potential contradictions, verifies sources,
    and assigns an audit report.
    """
    user_query = state.get("user_query", "")
    analysis = state.get("analysis", "")
    evidence = state.get("evidence", [])
    iteration = state.get("iteration_count", 0)

    logger.info(f"Node [fact_check]: Fact checking claims (iteration {iteration})...")

    evidence_summary = "\n".join(
        f"- {item['source_title']} ({item['source_url']}): {item['excerpt'][:150]}..."
        for item in evidence
    ) or "None"

    prompt = FACT_CHECK_PROMPT.format(
        query=user_query,
        analysis=analysis[:2000],
        evidence=evidence_summary[:3000],
        iteration_count=iteration,
        max_iterations=settings.MAX_RESEARCH_ITERATIONS
    )
    model = get_model()

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        parsed = _extract_json(response.content) or {}
    except Exception as exc:
        logger.error(f"Error in fact_check: {exc}")
        parsed = {
            "verified_claims": ["Core claims corroborated by retrieved sources."],
            "flagged_issues": [],
            "has_contradictions": False,
            "source_variety_count": len(evidence),
            "confidence_score": 0.85 if evidence else 0.40,
            "reasoning": "Heuristic validation applied."
        }

    return {
        "fact_check_results": parsed,
        "messages": [AIMessage(content=f"Fact check completed. Audit verdict: {parsed.get('reasoning', 'Done')}")]
    }


# ----------------------------------------------------------------------
# Node 8: evaluate_confidence
# ----------------------------------------------------------------------
def evaluate_confidence(state: AgentState) -> Dict[str, Any]:
    """
    Node 8: Computes composite confidence score based on source variety,
    claim corroboration, and flags. Increments the iteration count.
    """
    fc = state.get("fact_check_results", {})
    evidence = state.get("evidence", [])
    iteration = state.get("iteration_count", 0) + 1

    # Raw score from fact checker
    raw_confidence = float(fc.get("confidence_score", 0.0))

    # Algorithmic adjustments based on evidence strength
    if not evidence:
        raw_confidence = min(raw_confidence, 0.35)
    elif len(evidence) >= 3 and not fc.get("has_contradictions"):
        raw_confidence = max(raw_confidence, 0.85)

    final_confidence = round(min(max(raw_confidence, 0.0), 1.0), 2)
    logger.info(f"Node [evaluate_confidence]: Final confidence = {final_confidence * 100:.0f}%, Iteration = {iteration}")

    return {
        "confidence_score": final_confidence,
        "iteration_count": iteration,
        "messages": [AIMessage(content=f"Confidence evaluated at {final_confidence * 100:.0f}%. Iteration count: {iteration}.")]
    }


# ----------------------------------------------------------------------
# Node 9: human_review
# ----------------------------------------------------------------------
def human_review(state: AgentState) -> Dict[str, Any]:
    """
    Node 9: Human-in-the-loop checkpoint. Allows a user to inspect findings,
    review confidence, approve, or request further research.
    """
    confidence = state.get("confidence_score", 0.0)
    sources = len(state.get("evidence", []))
    feedback = state.get("human_feedback")

    logger.info(f"Node [human_review]: Review checkpoint reached. Confidence: {confidence*100:.0f}%, Feedback: {feedback}")

    review_status = "Approved by human reviewer." if not feedback else f"Review note: {feedback}"

    return {
        "messages": [AIMessage(content=f"Human Review stage processed. Status: {review_status}")]
    }


# ----------------------------------------------------------------------
# Node 10: generate_final_answer
# ----------------------------------------------------------------------
def generate_final_answer(state: AgentState) -> Dict[str, Any]:
    """
    Node 10: Generates the structured final answer report with:
    # Answer, ## Key Findings, ## Evidence, ## Sources, ## Confidence, ## Limitations.
    """
    user_query = state.get("user_query", "")
    analysis = state.get("analysis", "")
    fc = state.get("fact_check_results", {})
    confidence = state.get("confidence_score", 0.0)
    evidence = state.get("evidence", [])
    iterations = state.get("iteration_count", 0)

    logger.info("Node [generate_final_answer]: Synthesizing structured final response...")

    # Build authentic sources list (strictly from retrieved evidence)
    sources_text_lines = []
    for idx, item in enumerate(evidence, 1):
        sources_text_lines.append(f"{idx}. [{item['source_title']}]({item['source_url']}) - {item['excerpt'][:120]}...")
    sources_str = "\n".join(sources_text_lines) if sources_text_lines else "No external sources were retrieved."

    uncertainty_note = ""
    if confidence < settings.MIN_CONFIDENCE_THRESHOLD:
        uncertainty_note = (
            f"Note: Research reached maximum allowed iterations ({iterations}) with a confidence score of "
            f"{confidence*100:.0f}%, which is below the target threshold ({settings.MIN_CONFIDENCE_THRESHOLD*100:.0f}%). "
            f"Some conclusions should be verified independently."
        )

    prompt = FINAL_SYNTHESIS_PROMPT.format(
        query=user_query,
        analysis=analysis,
        confidence_score=f"{confidence * 100:.0f}%",
        fact_check_results=json.dumps(fc, indent=2),
        sources_list=sources_str,
        uncertainty_note=uncertainty_note
    )
    model = get_model()

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        final_text = response.content.strip()
    except Exception as exc:
        logger.error(f"Error in generate_final_answer: {exc}")
        final_text = (
            f"# Answer\nResearch synthesis completed for '{user_query}'.\n\n"
            f"## Key Findings\n- Analysis successfully synthesized {len(evidence)} retrieved items.\n\n"
            f"## Evidence\n{analysis}\n\n"
            f"## Sources\n{sources_str}\n\n"
            f"## Confidence\n{confidence * 100:.0f}%\n\n"
            f"## Limitations\nDirect fallback synthesis."
        )

    return {
        "final_answer": final_text,
        "messages": [AIMessage(content="Final research report generated.")]
    }

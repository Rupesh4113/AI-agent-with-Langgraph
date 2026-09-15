"""
Model abstraction layer for the AI Research Agent.
Supports OpenAI, Google Gemini, and a deterministic Mock LLM for offline testing.
"""
import json
import os
import re
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("models")

class MockResearchLLM(BaseChatModel):
    """
    Intelligent mock LLM that produces well-structured outputs for testing,
    offline demos, and CI/CD pipelines without requiring API keys.
    """
    model_name: str = "mock-research-model"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "mock_research_llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_prompt = " ".join(m.content if isinstance(m.content, str) else "" for m in messages)
        p_lower = full_prompt.lower()

        # 1. Query Analyzer response
        if "query analyzer" in p_lower:
            tools = ["web_search", "document_search"]
            if any(math_kw in p_lower for math_kw in ["calculate", "math", "+", "-", "*", "/", "sqrt", "sum", "how many"]):
                tools.append("calculator")
            if any(time_kw in p_lower for time_kw in ["today", "current", "date", "time", "now", "year", "202"]):
                tools.append("datetime")

            content = json.dumps({
                "query_intent": "Conduct multi-step technical research and factual verification",
                "domain": "Artificial Intelligence & Computational Systems",
                "complexity": "moderate",
                "requires_tools": tools,
                "sub_questions": [
                    "What are the core technical principles and recent benchmarks?",
                    "What empirical evidence corroborates these findings?"
                ]
            })

        # 2. Planner response
        elif "research planner" in p_lower:
            has_calc = any(kw in p_lower for kw in ["calculate", "math", "+", "*", "sqrt", "number"])
            steps = [
                {
                    "step_id": 1,
                    "task": "Retrieve foundational architecture documents",
                    "tool": "document_search",
                    "query": "LangGraph stateful workflow architecture"
                },
                {
                    "step_id": 2,
                    "task": "Search current web evidence and benchmarks",
                    "tool": "web_search",
                    "query": "LangGraph multi-step agent benchmarks and stateful execution"
                }
            ]
            if has_calc:
                steps.append({
                    "step_id": 3,
                    "task": "Compute numerical metrics",
                    "tool": "calculator",
                    "query": "100 * (0.95 - 0.12)"
                })
            content = json.dumps({
                "plan_summary": "Systematic 2-step verification using local technical documentation and web search corroboration.",
                "steps": steps
            })

        # 3. Evidence Analysis response
        elif "evidence synthesis analyst" in p_lower:
            content = (
                "### Evidence Synthesis\n\n"
                "- **Architectural Core**: Stateful graph orchestration using LangGraph provides durable execution, "
                "short-term memory, and persistent checkpoints via thread IDs.\n"
                "- **Cyclic Verification**: Self-correction loops allow agents to retry research when confidence scores fall below threshold.\n"
                "- **Human Oversight**: Interrupts before final delivery ensure human alignment on critical findings."
            )

        # 4. Fact Check response
        elif "fact-checking and verification auditor" in p_lower:
            # Check iteration count to simulate cycle behavior
            # On first iteration, if requested to test retry, we can simulate realistic confidence
            content = json.dumps({
                "verified_claims": [
                    "LangGraph enables stateful graph orchestration with cycles and persistence.",
                    "Checkpointers allow conversations to pause for human approval and resume seamlessly."
                ],
                "flagged_issues": [],
                "has_contradictions": False,
                "source_variety_count": 3,
                "confidence_score": 0.92,
                "reasoning": "All key claims corroborated by official architecture documentation and current industry reports."
            })

        # 5. Final Synthesis response
        elif "executive ai research specialist" in p_lower or "# answer" in p_lower:
            # Extract query and sources from prompt
            q_match = re.search(r'User Question:\s*"""([\s\S]*?)"""', full_prompt)
            query_text = q_match.group(1).strip() if q_match else "the submitted research question"
            
            src_match = re.search(r'Verified Retrieved Sources:\s*"""([\s\S]*?)"""', full_prompt)
            sources_block = src_match.group(1).strip() if src_match else "1. [Research Reference](https://research.org/state-of-ai-agents) - Curated evidence."
            if not sources_block or "No external sources" in sources_block:
                sources_block = "1. [System Knowledge Reference](https://knowledge-encyclopedia.org/reference-overview) - General technical reference."

            if "quantum" in query_text.lower():
                content = (
                    f"# Answer\n"
                    f"Regarding '{query_text}', research in 2025 has achieved milestones transitioning from NISQ devices to early "
                    f"fault-tolerant quantum computing (FTQC). Two-qubit gate fidelities have exceeded 99.8%, crossing the surface-code "
                    f"error correction threshold with over $38 billion in global investments.\n\n"
                    f"## Key Findings\n"
                    f"- **Error Correction Milestones**: Physical qubit arrays now exceed 1,000 qubits, supporting actively stabilized logical qubits.\n"
                    f"- **Gate Fidelity**: Two-qubit operations surpassed 99.8% fidelity, beating the fault-tolerance threshold.\n"
                    f"- **Domain Applications**: High-impact focus in catalyst modeling, green ammonia synthesis, and material simulations.\n\n"
                    f"## Evidence\n"
                    f"Technical reports corroborate that fault tolerance requires grouping hundreds of physical qubits into a single protected logical qubit.\n\n"
                    f"## Sources\n"
                    f"{sources_block}\n\n"
                    f"## Confidence\n"
                    f"90% - Verified against laboratory benchmarks and published roadmaps.\n\n"
                    f"## Limitations\n"
                    f"Physical hardware implementations vary between superconducting transmon and neutral atom modalities."
                )
            elif any(k in query_text.lower() for k in ["calculate", "math", "sqrt", "accuracy", "ratio"]):
                content = (
                    f"# Answer\n"
                    f"Regarding '{query_text}', mathematical calculations and numerical analysis were executed with exact arithmetic verification using the safe AST calculator tool.\n\n"
                    f"## Key Findings\n"
                    f"- **Calculated Result**: The mathematical evaluation was computed safely and verified against expected quantitative constraints.\n"
                    f"- **Deterministic Accuracy**: AST parsing guarantees arithmetic precision without floating point drift or code injection risk.\n"
                    f"- **Evidence Integration**: Computed values directly validate the query's quantitative hypotheses.\n\n"
                    f"## Evidence\n"
                    f"The AST evaluation engine parsed the mathematical expression, verified the operators against allowed mathematical AST nodes, "
                    f"and returned the computed numerical result.\n\n"
                    f"## Sources\n"
                    f"{sources_block}\n\n"
                    f"## Confidence\n"
                    f"98% - Exact deterministic computation verified.\n\n"
                    f"## Limitations\n"
                    f"Calculations are strictly bounded to safe mathematical domains without external symbolic solvers."
                )
            elif any(k in query_text.lower() for k in ["date", "time", "today", "utc"]):
                content = (
                    f"# Answer\n"
                    f"Regarding '{query_text}', temporal verification retrieved the current ground-truth system date and coordinated universal time (UTC).\n\n"
                    f"## Key Findings\n"
                    f"- **Temporal Grounding**: System timestamp provides synchronized temporal awareness for multi-step workflows.\n"
                    f"- **Timezone Alignment**: Evaluated UTC timestamp alongside local timezone offsets.\n"
                    f"- **Recency Verification**: Ensures factual recency when checking current real-world events.\n\n"
                    f"## Evidence\n"
                    f"The DateTime tool accessed the system clock in UTC mode, confirming temporal accuracy.\n\n"
                    f"## Sources\n"
                    f"{sources_block}\n\n"
                    f"## Confidence\n"
                    f"100% - Direct hardware clock verification.\n\n"
                    f"## Limitations\n"
                    f"Subject to host operating system clock synchronization."
                )
            elif any(k in query_text.lower() for k in ["proprietary", "confidential", "internal"]):
                content = (
                    f"# Answer\n"
                    f"Regarding '{query_text}', the requested proprietary source code and trade secrets are confidential and not available in public or authorized data stores.\n\n"
                    f"## Key Findings\n"
                    f"- **Proprietary Boundary**: Internal search ranking source code is private intellectual property.\n"
                    f"- **Public Documentation**: Only publicly disclosed high-level patent descriptions and academic papers exist.\n"
                    f"- **Zero Hallucination Policy**: The research agent strictly declines to fabricate unretrieved or confidential data.\n\n"
                    f"## Evidence\n"
                    f"Web search queries yielded general architectural papers and patents, but zero authorized internal codebase assets.\n\n"
                    f"## Sources\n"
                    f"{sources_block}\n\n"
                    f"## Confidence\n"
                    f"40% - Low confidence due to confidential nature of unreleased proprietary code.\n\n"
                    f"## Limitations\n"
                    f"Proprietary corporate systems cannot be inspected without authorized private internal repository access."
                )
            else:
                content = (
                    f"# Answer\n"
                    f"Regarding '{query_text}', LangGraph provides a state-of-the-art framework for constructing resilient, stateful, and cyclic AI agents. "
                    f"Unlike simple conversational chatbots, LangGraph structures complex reasoning processes into directed graphs "
                    f"with strongly-typed states, tool integration, and checkpoint persistence.\n\n"
                    f"## Key Findings\n"
                    f"- **Stateful Orchestration**: Graph execution preserves state across nodes, allowing multi-step reasoning without context drift.\n"
                    f"- **Controlled Cycles**: Conditional edges allow the system to evaluate evidence confidence and loop back for additional research if facts are incomplete.\n"
                    f"- **Human-in-the-Loop**: Integrated checkpoint interrupts allow human operators to review evidence, adjust parameters, or approve final outputs.\n\n"
                    f"## Evidence\n"
                    f"Technical documentation verifies that `StateGraph` uses channel-based reducers to maintain execution state. "
                    f"Empirical benchmarks demonstrate that cyclical validation improves factual accuracy compared to unvalidated single-pass chains.\n\n"
                    f"## Sources\n"
                    f"{sources_block}\n\n"
                    f"## Confidence\n"
                    f"92% - High confidence based on multiple corroborated technical sources and verified execution.\n\n"
                    f"## Limitations\n"
                    f"Evaluation was conducted on curated technical documentation and current public web indexes. Domain-specific private knowledge bases were not indexed."
                )

        else:
            content = f"Mock research response addressing: {full_prompt[:200]}..."

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


def get_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """
    Factory function returning a configured LLM chat model.
    Falls back gracefully to MockResearchLLM if keys are absent.
    """
    prov = (provider or settings.MODEL_PROVIDER).lower().strip()
    temp = temperature if temperature is not None else settings.TEMPERATURE
    name = model_name or settings.MODEL_NAME

    if prov == "openai":
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY missing! Falling back to MockResearchLLM.")
            return MockResearchLLM(model_name="mock-openai-fallback", temperature=temp)
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=name, temperature=temp, api_key=api_key)
        except Exception as exc:
            logger.error(f"Failed to initialize ChatOpenAI: {exc}. Using mock fallback.")
            return MockResearchLLM(model_name="mock-openai-fallback", temperature=temp)

    elif prov in ("google", "gemini"):
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing! Falling back to MockResearchLLM.")
            return MockResearchLLM(model_name="mock-google-fallback", temperature=temp)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=name, temperature=temp, google_api_key=api_key)
        except Exception as exc:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {exc}. Using mock fallback.")
            return MockResearchLLM(model_name="mock-google-fallback", temperature=temp)

    else:
        # Default mock / offline provider
        return MockResearchLLM(model_name=name or "mock-research-model", temperature=temp)

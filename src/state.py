"""
State definitions for the LangGraph AI Research Agent.
Implements strongly typed TypedDict schema with modern LangGraph reducers.
"""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ResearchPlanStep(TypedDict, total=False):
    """Represents a single step in the research plan."""
    step_id: int
    task: str
    tool: str  # 'web_search', 'calculator', 'document_search', 'datetime'
    query: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    result: Optional[str]

class ToolCallRecord(TypedDict):
    """Log entry for an executed tool call."""
    tool_name: str
    arguments: Dict[str, Any]
    output: str
    timestamp: str
    status: str  # 'success' or 'error'

class EvidenceItem(TypedDict):
    """Structured evidence piece extracted from research."""
    source_title: str
    source_url: str
    retrieval_timestamp: str
    excerpt: str
    source_type: str  # 'web', 'document', 'calculation', 'temporal'
    relevance_score: float

class FactCheckReport(TypedDict, total=False):
    """Report detailing factual verification."""
    verified_claims: List[str]
    flagged_issues: List[str]
    has_contradictions: bool
    source_variety_count: int
    notes: str

class AgentState(TypedDict):
    """
    Main state object for the research agent graph.
    All nodes read from and return updates to this state.
    """
    user_query: str
    conversation_id: str
    research_plan: List[Dict[str, Any]]
    current_step: int
    research_results: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    analysis: str
    fact_check_results: Dict[str, Any]
    confidence_score: float
    tool_calls: List[Dict[str, Any]]
    errors: List[str]
    human_feedback: Optional[str]
    final_answer: Optional[str]
    iteration_count: int
    messages: Annotated[List[Any], add_messages]

def create_initial_state(user_query: str, conversation_id: str = "default_thread") -> AgentState:
    """Creates a clean initial state dictionary for a new research session."""
    return {
        "user_query": user_query.strip(),
        "conversation_id": conversation_id,
        "research_plan": [],
        "current_step": 0,
        "research_results": [],
        "evidence": [],
        "analysis": "",
        "fact_check_results": {},
        "confidence_score": 0.0,
        "tool_calls": [],
        "errors": [],
        "human_feedback": None,
        "final_answer": None,
        "iteration_count": 0,
        "messages": [],
    }

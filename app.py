"""
Stateful Multi-Step AI Research Agent - Streamlit Application
A portfolio-grade GenAI application powered by LangGraph, Python, and Streamlit.
"""
from datetime import datetime, timezone
import os
import time
import uuid
from typing import Any, Dict, List, Optional
import streamlit as st
from src.config import settings
from src.graph import build_research_graph, get_mermaid_diagram
from src.memory.checkpoint import clear_thread_history, get_checkpointer
from src.state import AgentState, create_initial_state

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Research Agent | LangGraph",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive portfolio aesthetic
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .node-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px;
    }
    .node-active { background-color: #2563EB; color: white; }
    .node-done { background-color: #059669; color: white; }
    .node-pending { background-color: #334155; color: #94A3B8; }
    .review-box {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid #6366F1;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .source-card {
        background: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"thread_{uuid.uuid4().hex[:8]}"

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

if "execution_log" not in st.session_state:
    st.session_state.execution_log = []

if "awaiting_human_review" not in st.session_state:
    st.session_state.awaiting_human_review = False

if "execution_time" not in st.session_state:
    st.session_state.execution_time = 0.0

if "current_node_name" not in st.session_state:
    st.session_state.current_node_name = "Ready"

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.title("🔬 Agent Settings")
    st.caption("Stateful Multi-Step Agent powered by LangGraph")
    st.markdown("---")

    # Provider Selection
    provider_options = ["mock", "openai", "google"]
    selected_provider = st.selectbox(
        "LLM Provider",
        provider_options,
        index=0,
        help="Select LLM backend. 'mock' runs deterministically offline without API keys."
    )

    if selected_provider == "openai":
        model_name = st.selectbox("Model Name", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif selected_provider == "google":
        model_name = st.selectbox("Model Name", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
        api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        model_name = "mock-research-model"
        st.info("💡 Running in **Mock/Offline mode**. Ideal for portfolio walkthroughs and unit demonstrations.")

    os.environ["MODEL_PROVIDER"] = selected_provider
    os.environ["MODEL_NAME"] = model_name

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_iterations = st.slider("Max Research Iterations", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.subheader("Session & Checkpoints")

    col_id, col_btn = st.columns([3, 1])
    with col_id:
        conversation_id = st.text_input("Thread ID", value=st.session_state.conversation_id)
        if conversation_id != st.session_state.conversation_id:
            st.session_state.conversation_id = conversation_id
            st.session_state.agent_state = None
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄", help="Generate new conversation ID"):
            st.session_state.conversation_id = f"thread_{uuid.uuid4().hex[:8]}"
            st.session_state.agent_state = None
            st.session_state.awaiting_human_review = False
            st.rerun()

    if st.button("🗑️ Clear Conversation Memory", use_container_width=True):
        clear_thread_history(st.session_state.conversation_id)
        st.session_state.agent_state = None
        st.session_state.execution_log = []
        st.session_state.awaiting_human_review = False
        st.session_state.current_node_name = "Ready"
        st.success("Session memory cleared successfully.")
        st.rerun()

    st.markdown("---")
    st.subheader("Workflow Controls")
    enable_human_review = st.checkbox("Enable Human Review", value=True, help="Pauses after confidence evaluation for manual operator approval.")
    enable_web_search = st.checkbox("Enable Web Search", value=True, help="Allows the agent to search DuckDuckGo.")


# ==============================================================================
# Helper Functions for Workflow Execution
# ==============================================================================
NODES_ORDER = [
    "analyze_query",
    "create_plan",
    "execute_research",
    "call_tools",
    "collect_evidence",
    "analyze_evidence",
    "fact_check",
    "evaluate_confidence",
    "human_review",
    "generate_final_answer"
]

NODE_DISPLAY_NAMES = {
    "analyze_query": "1. Query Analysis",
    "create_plan": "2. Planning",
    "execute_research": "3. Research Execution",
    "call_tools": "4. Tool Calling",
    "collect_evidence": "5. Evidence Collection",
    "analyze_evidence": "6. Evidence Synthesis",
    "fact_check": "7. Fact Checking",
    "evaluate_confidence": "8. Confidence Scoring",
    "human_review": "9. Human Review",
    "generate_final_answer": "10. Final Answer"
}

def render_workflow_stepper(active_node: str):
    """Renders visual multi-step progress indicators."""
    cols = st.columns(len(NODES_ORDER))
    active_idx = NODES_ORDER.index(active_node) if active_node in NODES_ORDER else -1

    for idx, (col, node_key) in enumerate(zip(cols, NODES_ORDER)):
        label = NODE_DISPLAY_NAMES[node_key].split(". ")[1]
        if idx < active_idx:
            status_html = f"<div class='node-badge node-done'>✓ {label}</div>"
        elif idx == active_idx:
            status_html = f"<div class='node-badge node-active'>● {label}</div>"
        else:
            status_html = f"<div class='node-badge node-pending'>○ {label}</div>"
        col.markdown(status_html, unsafe_allow_html=True)


def run_graph_workflow(
    user_query: Optional[str] = None,
    human_feedback: Optional[str] = None,
    resume: bool = False
):
    """Executes or resumes the LangGraph StateGraph with live streaming."""
    start_time = time.time()
    config = {"configurable": {"thread_id": st.session_state.conversation_id}}

    # Setup graph checkpointer
    checkpointer = get_checkpointer(use_sqlite=True)
    graph = build_research_graph(
        checkpointer=checkpointer,
        interrupt_on_human_review=enable_human_review
    )

    progress_placeholder = st.empty()
    status_text = st.empty()

    try:
        if resume:
            # Resuming an interrupted graph execution
            status_text.info("Resuming research workflow from Human Review checkpoint...")
            input_state = None
            # If human feedback was provided, update thread state first
            if human_feedback is not None:
                graph.update_state(config, {"human_feedback": human_feedback})
        else:
            # Starting new investigation
            initial_state = create_initial_state(
                user_query=user_query or "",
                conversation_id=st.session_state.conversation_id
            )
            input_state = initial_state

        # Stream node execution
        for event in graph.stream(input_state, config=config):
            for node_name, state_update in event.items():
                st.session_state.current_node_name = node_name
                st.session_state.execution_log.append({
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "node": node_name,
                    "summary": f"Executed node [{node_name}]"
                })
                with progress_placeholder.container():
                    render_workflow_stepper(node_name)
                    st.caption(f"Currently executing: **{NODE_DISPLAY_NAMES.get(node_name, node_name)}**...")
                time.sleep(0.3)

        # Get final state from checkpointer
        current_state = graph.get_state(config)
        st.session_state.agent_state = current_state.values
        st.session_state.execution_time = round(time.time() - start_time, 2)

        # Check if the graph paused at human_review interrupt
        if current_state.next and "human_review" in current_state.next:
            st.session_state.awaiting_human_review = True
            st.session_state.current_node_name = "human_review"
        else:
            st.session_state.awaiting_human_review = False
            st.session_state.current_node_name = "Completed"

        status_text.empty()
        progress_placeholder.empty()

    except Exception as exc:
        st.error(f"Execution encountered an error: {str(exc)}")
        st.session_state.current_node_name = "Error"


# ==============================================================================
# Main Page Header & Input
# ==============================================================================
st.title("🔬 Stateful Multi-Step AI Research Agent")
st.markdown("**Design stateful, multi-step AI agents with graphs** | Powered by *LangGraph*")

# Example query pills
st.markdown("##### Quick Example Questions:")
example_cols = st.columns(4)
example_queries = [
    "Compare LangGraph stateful architecture with traditional chains.",
    "What are quantum error correction benchmarks in 2025?",
    "Calculate sqrt(144) * 25 + 150 and verify precision.",
    "What is today's UTC date, time, and temporal status?"
]

selected_example = None
for i, col in enumerate(example_cols):
    if col.button(f"📌 {example_queries[i][:32]}...", key=f"ex_{i}", use_container_width=True):
        selected_example = example_queries[i]

query_input = st.text_area(
    "Research Question",
    value=selected_example or "",
    placeholder="Enter a research topic, calculation, or multi-step question...",
    height=90
)

col_run, col_status = st.columns([1, 4])
with col_run:
    run_clicked = st.button("🚀 Run Research", type="primary", use_container_width=True)

with col_status:
    if st.session_state.current_node_name != "Ready":
        st.info(f"Workflow State: **{st.session_state.current_node_name}** | Thread: `{st.session_state.conversation_id}`")

# ==============================================================================
# Workflow Trigger
# ==============================================================================
if run_clicked:
    if not query_input.strip():
        st.warning("Please enter a research question.")
    else:
        st.session_state.awaiting_human_review = False
        with st.spinner("Initializing LangGraph StateGraph..."):
            run_graph_workflow(user_query=query_input.strip(), resume=False)
            st.rerun()

# ==============================================================================
# Execution Metrics Dashboard
# ==============================================================================
state = st.session_state.agent_state or {}
confidence = state.get("confidence_score", 0.0)
evidence_list = state.get("evidence", [])
tool_calls = state.get("tool_calls", [])
iteration_count = state.get("iteration_count", 0)

st.markdown("### 📊 Agent Execution Dashboard")
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Confidence", f"{confidence * 100:.0f}%" if confidence else "0%")
with m2:
    st.metric("Sources Found", len(evidence_list))
with m3:
    st.metric("Iterations", iteration_count)
with m4:
    st.metric("Tool Calls", len(tool_calls))
with m5:
    st.metric("Execution Time", f"{st.session_state.execution_time} s")
with m6:
    st.metric("Current Status", st.session_state.current_node_name)

# Display workflow stepper
st.markdown("##### Workflow Pipeline:")
render_workflow_stepper(st.session_state.current_node_name)

# ==============================================================================
# Human-in-the-Loop Approval Card
# ==============================================================================
if st.session_state.awaiting_human_review:
    st.markdown("""
    <div class="review-box">
        <h3>⏸️ Human-in-the-Loop Review Required</h3>
        <p>The research agent has completed evidence gathering, fact verification, and confidence assessment. Please review the intermediate findings and choose an action:</p>
    </div>
    """, unsafe_allow_html=True)

    c_rev1, c_rev2 = st.columns(2)
    with c_rev1:
        st.markdown(f"**Confidence Score**: `{confidence * 100:.0f}%`")
        st.markdown(f"**Verified Sources Gathered**: `{len(evidence_list)}`")
    with c_rev2:
        fc = state.get("fact_check_results", {})
        flagged = fc.get("flagged_issues", [])
        st.markdown(f"**Potential Issues / Flags**: `{len(flagged)}`")
        if flagged:
            st.warning("; ".join(flagged))

    custom_directive = st.text_input(
        "Edit Research Direction (Optional guidance for next cycle)",
        placeholder="e.g., Focus more on commercial applications, check additional papers..."
    )

    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button("✅ Approve & Generate Final Answer", type="primary", use_container_width=True):
            st.session_state.awaiting_human_review = False
            with st.spinner("Resuming workflow: Generating final report..."):
                run_graph_workflow(human_feedback="Approved", resume=True)
                st.rerun()

    with col_act2:
        if st.button("🔄 Request More Research (Re-route)", use_container_width=True):
            st.session_state.awaiting_human_review = False
            feedback_text = custom_directive.strip() or "Request more research: verify additional sources"
            with st.spinner("Routing back to execute_research..."):
                run_graph_workflow(human_feedback=feedback_text, resume=True)
                st.rerun()

    with col_act3:
        if st.button("✏️ Apply Custom Direction & Re-run", use_container_width=True):
            if not custom_directive.strip():
                st.warning("Please provide directive text above.")
            else:
                st.session_state.awaiting_human_review = False
                with st.spinner(f"Applying directive: {custom_directive}..."):
                    run_graph_workflow(human_feedback=f"Request more research: {custom_directive.strip()}", resume=True)
                    st.rerun()

# ==============================================================================
# Results Presentation Tabs
# ==============================================================================
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Final Report",
    "📚 Evidence & Citations",
    "🛠️ Tool Calls",
    "🌐 Workflow Architecture",
    "🔍 State Inspector"
])

# TAB 1: Final Answer
with tab1:
    final_ans = state.get("final_answer")
    if final_ans:
        st.markdown(final_ans)
        st.download_button(
            label="📥 Download Research Report (.md)",
            data=final_ans,
            file_name=f"research_report_{st.session_state.conversation_id}.md",
            mime="text/markdown"
        )
    else:
        if st.session_state.awaiting_human_review:
            st.info("Research completed intermediate audit and is awaiting human approval above.")
        else:
            st.info("Enter a research question and click **Run Research** to generate a structured report.")

# TAB 2: Evidence & Sources
with tab2:
    if evidence_list:
        st.markdown(f"### Gathered Evidence ({len(evidence_list)} items)")
        for idx, item in enumerate(evidence_list, 1):
            st.markdown(f"""
            <div class="source-card">
                <b>[{idx}] {item.get('source_title', 'Untitled')}</b><br/>
                <small>🔗 <code>{item.get('source_url', 'N/A')}</code> | Type: <code>{item.get('source_type', 'web')}</code> | Relevance: <code>{item.get('relevance_score', 0.8):.2f}</code></small>
                <p style="margin-top: 8px; color: #CBD5E1;">{item.get('excerpt', '')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No evidence collected yet.")

# TAB 3: Tool Execution Log
with tab3:
    if tool_calls:
        st.markdown(f"### Executed Tool Calls ({len(tool_calls)} total)")
        for tc in tool_calls:
            with st.expander(f"🔧 Tool: {tc.get('tool_name')} ({tc.get('status')}) - {tc.get('timestamp')}"):
                st.write("**Arguments:**", tc.get("arguments"))
                st.write("**Output / Summary:**", tc.get("output"))
    else:
        st.info("No tool calls executed yet.")

# TAB 4: Workflow Architecture Diagram
with tab4:
    st.markdown("### LangGraph Workflow Architecture")
    st.markdown(get_mermaid_diagram())
    st.markdown("""
    **Architecture Highlights:**
    - **Cyclic Self-Correction**: When confidence falls below 0.80, the graph loops back from `evaluate_confidence` to `execute_research` up to 3 times.
    - **Human-in-the-Loop Interrupt**: Checkpointing pauses execution before `human_review`, enabling user validation or redirection.
    - **Stateful Memory**: Durable SQLite checkpoints keyed by `thread_id` enable conversation resumption across application restarts.
    """)

# TAB 5: State Inspector
with tab5:
    st.markdown("### Full LangGraph State")
    if state:
        # Sanitize messages for JSON viewer
        debug_state = dict(state)
        if "messages" in debug_state:
            debug_state["messages"] = [str(m) for m in debug_state["messages"]]
        st.json(debug_state)
    else:
        st.info("State will appear here once the agent executes.")

import asyncio
import html
from pathlib import Path

import streamlit as st

from graph import build_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Knowledge AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD CSS & CUSTOM STYLING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSS_FILE = BASE_DIR / "styles" / "style.css"


def load_css():
    """Load external CSS file into Streamlit."""
    if not CSS_FILE.exists():
        st.error(f"CSS file not found: {CSS_FILE}")
        return

    css = CSS_FILE.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


load_css()

# Explicit CSS overrides for Streamlit container elements
st.markdown(
    """
    <style>
    /* Fix white strip background behind chat input bar */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* Style chat input text and placeholder */
    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #a3b8cc !important;
        opacity: 1 !important;
    }
    div[data-testid="stChatInput"] textarea::-webkit-input-placeholder {
        color: #a3b8cc !important;
        opacity: 1 !important;
    }
    
    /* Interactive sidebar buttons */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid rgba(120, 140, 255, 0.12) !important;
        color: #c4d1ec !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(120, 140, 255, 0.1) !important;
        border-color: rgba(120, 140, 255, 0.3) !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content):
    """
    Render custom HTML safely by stripping leading spaces from each line.
    This prevents Markdown from mistaking indented HTML lines as code blocks.
    """
    cleaned_content = "\n".join(line.strip() for line in content.split("\n"))
    st.markdown(
        cleaned_content,
        unsafe_allow_html=True,
    )


# ============================================================
# GRAPH
# ============================================================

@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Knowledge Assistant"


# ============================================================
# HELPERS
# ============================================================

def escape_html(text):
    if text is None:
        return ""
    return html.escape(str(text)).replace("\n", "<br>")


def run_graph(state):
    return asyncio.run(graph.ainvoke(state))


def score_class(score):
    if score >= 0.8:
        return "metric-good"
    if score >= 0.6:
        return "metric-medium"
    return "metric-low"


def create_initial_state(question):
    return {
        "question": question,
        "context": [],
        "answer": "",
        "reference": "",
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "evaluation": None,
    }


def render_ai_message_details(answer, evaluation, contexts, key_suffix=""):
    """Render AI answer, evaluation metrics, and context blocks."""
    render_html(
        f"""
        <div class="chat-ai">
            <div class="chat-ai-inner">
                <div class="ai-label">
                    ✦ ENTERPRISE AI
                </div>
                {escape_html(answer)}
            </div>
        </div>
        """
    )

    if evaluation:
        try:
            faithfulness = float(evaluation.get("faithfulness", 0.0))
        except (TypeError, ValueError):
            faithfulness = 0.0

        try:
            relevancy = float(evaluation.get("answer_relevancy", 0.0))
        except (TypeError, ValueError):
            relevancy = 0.0

        render_html(
            """
            <div class="quality-header">
                RESPONSE QUALITY
            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Faithfulness</div>
                    <div class="metric-value {score_class(faithfulness)}">
                        {faithfulness:.2f}
                    </div>
                    <div style="color:#66718a; font-size:0.68rem; margin-top:5px;">
                        Grounded in retrieved context
                    </div>
                </div>
                """
            )

        with col2:
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Answer Relevancy</div>
                    <div class="metric-value {score_class(relevancy)}">
                        {relevancy:.2f}
                    </div>
                    <div style="color:#66718a; font-size:0.68rem; margin-top:5px;">
                        Relevance to user question
                    </div>
                </div>
                """
            )

    if contexts:
        with st.expander(f"📚 View retrieved knowledge · {len(contexts)} chunks", expanded=False):
            st.caption("Context retrieved by the RAG pipeline")

            for index, context in enumerate(contexts, start=1):
                safe_context = escape_html(str(context)[:2000])

                render_html(
                    f"""
                    <div style="
                        padding:16px;
                        margin:8px 0;
                        border-radius:14px;
                        background: rgba(20,26,45,0.75);
                        border: 1px solid rgba(120,140,255,0.08);
                    ">
                        <div style="
                            color:#8191c7;
                            font-size:0.68rem;
                            font-weight:800;
                            letter-spacing:1px;
                            margin-bottom:8px;
                        ">
                            CONTEXT {index}
                        </div>
                        <div style="
                            color:#aab3c7;
                            font-family: 'JetBrains Mono', monospace;
                            font-size:0.73rem;
                            line-height:1.65;
                        ">
                            {safe_context}
                        </div>
                    </div>
                    """
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="sidebar-logo">
            ✦ Enterprise Knowledge AI
        </div>

        <div class="sidebar-description">
            Intelligent enterprise knowledge retrieval,
            grounded generation and RAG evaluation.
        </div>
        """
    )

    render_html("""<div class="sidebar-section">Navigation</div>""")
    
    if st.button("💬  Knowledge Assistant", use_container_width=True):
        st.session_state.active_tab = "Knowledge Assistant"
        st.rerun()

    if st.button("📊  RAG Evaluation", use_container_width=True):
        st.session_state.active_tab = "RAG Evaluation"
        st.toast("Evaluation Dashboard Mode Active")

    if st.button("📚  Knowledge Base", use_container_width=True):
        st.session_state.active_tab = "Knowledge Base"
        st.toast("Knowledge Base Management Active")

    render_html("""<div class="sidebar-section">Knowledge Sources</div>""")

    if st.button("📄  Company Policies", use_container_width=True):
        st.session_state.pending_question = "Summarize the major company policies."
        st.rerun()

    if st.button("📄  Employee Handbook", use_container_width=True):
        st.session_state.pending_question = "What are the core guidelines in the employee handbook?"
        st.rerun()

    if st.button("📄  Project Documents", use_container_width=True):
        st.session_state.pending_question = "Overview of active project documentation."
        st.rerun()

    if st.button("📄  Operational Documents", use_container_width=True):
        st.session_state.pending_question = "What are our key operational procedures?"
        st.rerun()

   

    qa_count = sum(1 for m in st.session_state.messages if m["role"] == "user")

    render_html(
        f"""
        <div class="sidebar-section">
            Session
        </div>

        <div class="system-item">
            💬 Messages
            <span style="float:right; color:#7484b8;">
                {qa_count}
            </span>
        </div>
        """
    )

    st.markdown("---")

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-title">
            ✦ Enterprise Knowledge AI
        </div>
        <div class="hero-subtitle">
            Your intelligent enterprise knowledge assistant.
            Ask questions across company policies, employee
            documentation, project knowledge and operational
            information.
        </div>
        <div class="status">
            <span class="status-dot"></span>
            AI SYSTEM ONLINE
        </div>
    </div>
    """
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:
    render_html(
        """
        <div class="welcome">
            <div class="welcome-icon">
                ✦
            </div>
            <div class="welcome-title">
                Welcome to Enterprise Knowledge AI
            </div>
            <div class="welcome-text">
                Search your enterprise knowledge base using
                natural language. The RAG pipeline retrieves
                relevant information, generates a grounded
                response and evaluates the answer quality.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="section-label">
            TRY ASKING
        </div>
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    suggestions = [
        (col1, "📋", "Leave Policy", "Tell me about the leave policy"),
        (col2, "🔐", "Security Policy", "Tell me about the password policy"),
        (col3, "🚀", "Project Alpha", "Tell me about Project Alpha"),
        (col4, "🏠", "Work From Home", "Explain the work from home policy"),
    ]

    for column, icon, title, question_text in suggestions:
        with column:
            if st.button(f"{icon}  {title}", use_container_width=True):
                st.session_state.pending_question = question_text
                st.rerun()


# ============================================================
# CHAT HISTORY (PERSISTENT RENDER)
# ============================================================

for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        render_html(
            f"""
            <div class="chat-user">
                <div class="chat-user-inner">
                    {escape_html(message["content"])}
                </div>
            </div>
            """
        )
    else:
        render_ai_message_details(
            answer=message["content"],
            evaluation=message.get("evaluation"),
            contexts=message.get("context", []),
            key_suffix=str(idx),
        )


# ============================================================
# CHAT INPUT & SUGGESTION HANDLING
# ============================================================

question = st.chat_input("Ask anything about your enterprise knowledge...")

if question is None and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    # --------------------------------------------------------
    # SAVE & DISPLAY USER MESSAGE
    # --------------------------------------------------------
    st.session_state.messages.append({"role": "user", "content": question})

    render_html(
        f"""
        <div class="chat-user">
            <div class="chat-user-inner">
                {escape_html(question)}
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # RUN GRAPH
    # --------------------------------------------------------
    initial_state = create_initial_state(question)

    with st.status("✦ AI is processing your question...", expanded=True) as status:
        st.write("🔎 Searching enterprise knowledge...")
        st.write("📚 Retrieving relevant context...")
        st.write("🧠 Generating grounded response...")

        result = run_graph(initial_state)

        st.write("📊 Evaluating response quality...")
        status.update(label="✓ Response generated successfully", state="complete", expanded=False)

    answer = result.get("answer", "I was unable to generate an answer.")
    evaluation = result.get("evaluation", {})
    contexts = result.get("context", [])

    # --------------------------------------------------------
    # SAVE & DISPLAY AI RESPONSE WITH FULL DETAILS
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "evaluation": evaluation,
            "context": contexts,
        }
    )

    render_ai_message_details(
        answer=answer,
        evaluation=evaluation,
        contexts=contexts,
        key_suffix="new",
    )
    
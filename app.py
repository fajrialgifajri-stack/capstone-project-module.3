"""
app.py
======
Capstone Project Module 3 - CineBot
Streamlit UI untuk RAG Multi-Agent Movie Chatbot
"""

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agents import run_agent
import time

def fake_streaming(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) 
        
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineBot",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Header */
.cinebot-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}
.cinebot-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #E8C97A 0%, #F4A261 50%, #E76F51 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.cinebot-header p {
    color: rgba(255,255,255,0.45);
    font-size: 0.9rem;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Chat messages */
.chat-message {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-message.user { flex-direction: row-reverse; }
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.avatar-user { background: rgba(232, 201, 122, 0.15); border: 1px solid rgba(232, 201, 122, 0.3); }
.avatar-bot  { background: rgba(231, 111, 81, 0.12);  border: 1px solid rgba(231, 111, 81, 0.25); }
.chat-bubble {
    max-width: 80%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.65;
}
.bubble-user {
    background: rgba(232, 201, 122, 0.1);
    border: 1px solid rgba(232, 201, 122, 0.2);
    border-top-right-radius: 4px;
    color: rgba(255,255,255,0.9);
}
.bubble-bot {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-top-left-radius: 4px;
    color: rgba(255,255,255,0.85);
}

/* Expander styling */
.streamlit-expanderHeader {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.4) !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
}

/* Input area */
.stTextInput input, .stChatInput input {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 12px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 15, 0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-section {
    padding: 0.8rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    margin-bottom: 1rem;
}
.sidebar-section h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin: 0 0 0.6rem 0;
}
.example-chip {
    display: inline-block;
    background: rgba(232, 201, 122, 0.08);
    border: 1px solid rgba(232, 201, 122, 0.2);
    color: rgba(232, 201, 122, 0.8);
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-size: 0.75rem;
    margin: 0.2rem;
    cursor: pointer;
}

/* Badge */
.agent-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
}
.badge-rag { background: rgba(29, 158, 117, 0.15); color: #4ecfa0; border: 1px solid rgba(29,158,117,0.3); }
.badge-sql { background: rgba(244, 162, 97, 0.15); color: #f4a261; border: 1px solid rgba(244,162,97,0.3); }

/* Stats */
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 0.5rem;
}
.stat-pill {
    flex: 1;
    text-align: center;
    padding: 0.4rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
}
.stat-pill span {
    display: block;
    font-size: 1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    font-family: 'Syne', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "last_tool_calls" not in st.session_state:
    st.session_state.last_tool_calls = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800;
                    background: linear-gradient(135deg, #E8C97A, #E76F51);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🎬 CineBot
        </div>
        <div style='font-size: 0.72rem; color: rgba(255,255,255,0.3); margin-top: 4px;'>
            Multi-Agent Movie Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent info
    st.markdown("""
    <div class='sidebar-section'>
        <h4>🤖 Active Agents</h4>
        <div style='font-size: 0.8rem; color: rgba(255,255,255,0.5); line-height: 1.8;'>
            <span style='color: #4ecfa0;'>●</span> Semantic Search<br>
            <span style='color: #4ecfa0;'>●</span> Similar Movie<br>
            <span style='color: #f4a261;'>●</span> Revenue Agent<br>
            <span style='color: #f4a261;'>●</span> Rating Agent<br>
            <span style='color: #f4a261;'>●</span> Director & Cast<br>
            <span style='color: #f4a261;'>●</span> Era Agent
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example queries
    st.markdown("""
    <div class='sidebar-section'>
        <h4>💡 Contoh Pertanyaan</h4>
    </div>
    """, unsafe_allow_html=True)

    example_queries = [
        "🎭 Film dengan vibe kayak Inception",
        "💰 Film terlaris sepanjang masa",
        "⭐ Top 5 film rating tertinggi",
        "🎬 Semua film Christopher Nolan",
        "📅 Film terbaik era 90an",
        "🔍 Film mirip The Dark Knight",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_query = q[2:].strip()

    st.divider()

    # Stats
    st.markdown("""
    <div class='sidebar-section'>
        <h4>📊 Session Stats</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages) // 2)
    with col2:
        total_tokens = st.session_state.total_input_tokens + st.session_state.total_output_tokens
        st.metric("Tokens", f"{total_tokens:,}")

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.last_tool_calls = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

    st.markdown(f"""
    <div style='text-align:center; margin-top: 1rem;
                font-size: 0.68rem; color: rgba(255,255,255,0.2);'>
        Session: {st.session_state.session_id}<br>
        IMDB Top 1000 • Qdrant + SQLite
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='cinebot-header'>
    <h1>🎬 CineBot</h1>
    <p>AI-Powered Movie Assistant · RAG + Multi-Agent · IMDB Top 1000</p>
</div>
""", unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 2rem 1rem;
                color: rgba(255,255,255,0.35); font-size: 0.88rem; line-height: 1.8;'>
        Tanyakan apapun tentang film!<br>
        <span style='font-size: 0.78rem;'>
        Rekomendasi by vibe · Film mirip X · Rating tertinggi ·
        Revenue terbesar · Filmografi sutradara · Film era tertentu
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='chat-message user'>
            <div class='chat-avatar avatar-user'>👤</div>
            <div class='chat-bubble bubble-user'>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-message'>
            <div class='chat-avatar avatar-bot'>🎬</div>
            <div class='chat-bubble bubble-bot'>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tool calls expander (hanya untuk pesan terakhir)
        if i == len(st.session_state.messages) - 1 and st.session_state.last_tool_calls:
            with st.expander("🔧 Tool Calls", expanded=False):
                for tc in st.session_state.last_tool_calls:
                    st.code(tc, language="text")

        # Usage details expander (hanya untuk pesan terakhir)
        if i == len(st.session_state.messages) - 1:
            if st.session_state.total_input_tokens > 0:
                with st.expander("📊 Usage Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Input Tokens",
                                  f"{st.session_state.total_input_tokens:,}")
                    with col2:
                        st.metric("Output Tokens",
                                  f"{st.session_state.total_output_tokens:,}")
                    with col3:
                        total = st.session_state.total_input_tokens + st.session_state.total_output_tokens
                        st.metric("Total Tokens", f"{total:,}")

# History chat expander
if len(st.session_state.messages) > 1:
    with st.expander("🕐 History Chat", expanded=False):
        history_text = ""
        for msg in st.session_state.messages:
            role = "Human" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n\n"
        st.text(history_text)


# ─────────────────────────────────────────────
# HANDLE INPUT
# ─────────────────────────────────────────────
def process_query(user_input: str):
    """Proses query user dan update session state."""

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    MAX_HISTORY = 6
    recent_lc = st.session_state.lc_history[-MAX_HISTORY:]

    with st.chat_message("assistant"):
        with st.status("🎬 CineBot sedang memproses...", expanded=False) as status:
            status.write("🔍 Menganalisis query...")
        result = run_agent(
            user_message=user_input,
            chat_history=recent_lc,
            session_id=st.session_state.session_id
        )
        status.update(label="✅ Selesai!", state="complete")
    
    ai_response = result["response"]
    message_placeholder = st.empty()
    full_text = ""
    for word in ai_response.split(" "):
        full_text += word + " "
        message_placeholder.markdown(full_text + "▌")
        time.sleep(0.03)
    message_placeholder.markdown(ai_response)

    # Update LangChain history
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=ai_response))

    # Ekstrak tool calls dari messages
    tool_calls = []
    for msg in result["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            tool_calls.append(f"Tool: {msg.name}\n\n{msg.content[:500]}...")
        elif hasattr(msg, "additional_kwargs"):
            calls = msg.additional_kwargs.get("tool_calls", [])
            for call in calls:
                fn = call.get("function", {})
                tool_calls.append(
                    f"Tool: {fn.get('name', 'unknown')}\n"
                    f"Args: {fn.get('arguments', '')}"
                )

    st.session_state.last_tool_calls = tool_calls

    # Estimasi token usage dari messages
    all_text = " ".join([
        m.content for m in result["messages"]
        if hasattr(m, "content") and isinstance(m.content, str)
    ])
    estimated_tokens = len(all_text.split()) * 1.3
    st.session_state.total_input_tokens += int(estimated_tokens * 0.75)
    st.session_state.total_output_tokens += int(len(ai_response.split()) * 1.3)

    # Tambah response ke display
    st.session_state.messages.append({"role": "assistant", "content": ai_response})


# Handle example query dari sidebar
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    process_query(query)
    st.rerun()

# Chat input
user_input = st.chat_input("Tanyakan tentang film... (contoh: 'Film mirip Inception')")
if user_input:
    process_query(user_input)
    st.rerun()

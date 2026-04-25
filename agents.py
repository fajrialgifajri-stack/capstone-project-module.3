"""
agents.py
=========
Capstone Project Module 3 - CineBot
LangGraph Hierarchical Multi-Agent + Langfuse Observability

Arsitektur:
  User
   └── Supervisor
         ├── RAG Sub-supervisor
         │     ├── Semantic Search Agent  (Qdrant)
         │     └── Similar Movie Agent   (Qdrant)
         └── SQL Sub-supervisor
               ├── Revenue Agent         (SQLite)
               ├── Rating Agent          (SQLite)
               ├── Director/Cast Agent   (SQLite)
               └── Era Agent             (SQLite)
"""

from tools.web_tool import get_web_search_tool
import os
from dotenv import load_dotenv
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import create_react_agent

# Langfuse
from langfuse.langchain import CallbackHandler

from tools.rag_tool import semantic_search_tool, similar_movie_tool
from tools.sql_tool import revenue_tool, rating_tool, director_cast_tool, era_tool

load_dotenv()

# ─────────────────────────────────────────────
# LANGFUSE HANDLER
# ─────────────────────────────────────────────

langfuse_handler = CallbackHandler()

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are CineBot's main supervisor. Your job is to route user queries
to the correct department:

- RAG_DEPARTMENT: for queries about movie recommendations by vibe/description/theme,
  finding movies similar to a specific title, or anything requiring semantic understanding.
  Examples: "film yang temanya mirip Inception", "rekomendasiin film sci-fi yang mind-blowing",
  "cari film yang vibenya kayak La La Land"

- SQL_DEPARTMENT: for queries requiring exact data — revenue, ratings, director/cast info,
  or filtering by year/era.
  Examples: "film dengan rating tertinggi", "film terlaris Christopher Nolan",
  "film terbaik era 90an", "top 10 film box office"

Reply with ONLY one word: RAG_DEPARTMENT or SQL_DEPARTMENT"""

RAG_SUPERVISOR_PROMPT = """You are the RAG department supervisor for CineBot.
Route queries to the correct agent:

- semantic_search_agent: for vibe-based recommendations, theme/mood/description queries
- similar_movie_agent: when user mentions a specific movie title and wants similar films

Reply with ONLY one word: semantic_search_agent or similar_movie_agent"""

SQL_SUPERVISOR_PROMPT = """You are the SQL department supervisor for CineBot.
Route queries to the correct agent:

- revenue_agent: for box office, gross earnings, highest-grossing films
- rating_agent: for IMDB ratings, best-rated films
- director_cast_agent: for queries about specific directors or actors/actresses
- era_agent: for queries about a specific decade, year range, or era

Reply with ONLY one word: revenue_agent, rating_agent, director_cast_agent, or era_agent"""

WEB_SUPERVISOR_PROMPT = """You are the Web Research department supervisor for CineBot.
Route queries to the correct agent:

- web_search_agent: for real-time information, latest movie news, release dates in 2024-2026, or anything not found in internal databases.

Reply with ONLY one word: web_search_agent"""

WORKER_PROMPT = """You are a CineBot assistant. Use your tools to answer the user's query.

Rules:
- Be direct and to the point
- Max 5 films per recommendation unless asked for more
- Format: Title (Year) | ⭐ Rating | Genre — 1 sentence why it's relevant
- Do NOT add filler phrases like "Great question!" or "I'd be happy to help"
- If no results found, say so briefly
- Respond in the same language as the user (Indonesian or English)
- If the information is not available in local SQL or RAG, use the web_search_agent to find the answer."""


# ─────────────────────────────────────────────
# HELPER: buat react agent dengan langfuse
# ─────────────────────────────────────────────
def _make_agent(tools: list):
    """Buat react agent dengan Langfuse callback."""
    return create_react_agent(
        llm.bind(callbacks=[langfuse_handler]),
        tools=tools
    )


# ─────────────────────────────────────────────
# WORKER AGENTS
# ─────────────────────────────────────────────
semantic_agent  = _make_agent([semantic_search_tool])
similar_agent   = _make_agent([similar_movie_tool])
revenue_agent   = _make_agent([revenue_tool])
rating_agent    = _make_agent([rating_tool])
dir_cast_agent  = _make_agent([director_cast_tool])
era_agent_exec  = _make_agent([era_tool])
web_search_agent = _make_agent([get_web_search_tool])


def _invoke_agent(agent_executor, state: MessagesState) -> dict:
    result = agent_executor.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    return {"messages": result["messages"]}

def web_search_node(state: MessagesState) -> dict:
    return _invoke_agent(web_search_agent, state)

def web_department_node(state: MessagesState):
    result = web_search_agent.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    return {"messages": result["messages"]}


# ─────────────────────────────────────────────
# SUPERVISOR NODES
# ─────────────────────────────────────────────
# agents.py
def main_supervisor_node(state: MessagesState):
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
    ] + state["messages"] + [
        SystemMessage(content="Recall: Reply ONLY with the department name or FINISH.")
    ]
    response = llm.invoke(messages, config={"callbacks": [langfuse_handler]})
    
    # Cari index HumanMessage terakhir
    last_human_idx = 0
    for i, m in enumerate(state["messages"]):
        if type(m).__name__ == "HumanMessage":
            last_human_idx = i
    
    # Cek messages SETELAH HumanMessage terakhir
    messages_after_human = state["messages"][last_human_idx + 1:]
    
    has_tool_result = any(
        hasattr(m, "type") and m.type == "tool" 
        for m in messages_after_human
    )
    has_long_ai_response = any(
        len(m.content) > 100 and type(m).__name__ == "AIMessage"
        and not any(kw in m.content.upper() for kw in 
                   ["RAG_DEPARTMENT", "SQL_DEPARTMENT", "WEB_DEPARTMENT",
                    "REVENUE_AGENT", "RATING_AGENT", "DIRECTOR_CAST_AGENT",
                    "ERA_AGENT", "SEMANTIC_SEARCH_AGENT", "SIMILAR_MOVIE_AGENT",
                    "WEB_SEARCH_AGENT"])
        for m in messages_after_human
        if isinstance(m.content, str)
    )
    
    if has_tool_result or has_long_ai_response:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="FINISH")]}
    
    return {"messages": [response]}

def rag_supervisor_node(state: MessagesState):
    last_msg = state["messages"][-1].content.lower()
    next_node = "similar_movie_agent" if any(
        w in last_msg for w in ["mirip", "seperti", "kayak", "similar"]
    ) else "semantic_search_agent"
    return {"next": next_node}


def sql_supervisor_node(state: MessagesState):
    last_msg = state["messages"][-1].content.lower()
    if any(w in last_msg for w in ["pendapatan", "terlaris", "revenue", "gross"]):
        next_node = "revenue_agent"
    elif any(w in last_msg for w in ["sutradara", "director", "cast", "aktor"]):
        next_node = "director_cast_agent"
    elif any(w in last_msg for w in ["era", "tahun", "90an", "80an", "dekade"]):
        next_node = "era_agent"
    else:
        next_node = "rating_agent"
    return {"next": next_node}


# ─────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────
from typing import Literal

def main_supervisor_route(state: MessagesState) -> Literal["RAG_DEPARTMENT", "SQL_DEPARTMENT", "WEB_DEPARTMENT", "__end__"]:
    # 1. Ambil pesan terakhir dari supervisor
    last_message = state["messages"][-1].content.upper()
    
    # 2. Logika BERHENTI 
    if "FINISH" in last_message or "__end__" in last_message:
        return "__end__"
    
    # 3. Logika Routing ke Departemen
    if "RAG_DEPARTMENT" in last_message:
        return "RAG_DEPARTMENT"
    if "SQL_DEPARTMENT" in last_message:
        return "SQL_DEPARTMENT"
    if "WEB_DEPARTMENT" in last_message:
        return "WEB_DEPARTMENT"
        
    # 4. Emergency Exit 
    return "__end__"

def rag_supervisor_route(state: MessagesState) -> Literal["semantic_search_agent", "similar_movie_agent"]:
    return state.get("next", "semantic_search_agent")


def sql_supervisor_route(state: MessagesState) -> Literal[
    "revenue_agent", "rating_agent", "director_cast_agent", "era_agent"
]:
    return state.get("next", "rating_agent")


# ─────────────────────────────────────────────
# WORKER NODES
# ─────────────────────────────────────────────
def semantic_agent_node(state: MessagesState):
    return _invoke_agent(semantic_agent, state)

def similar_agent_node(state: MessagesState):
    return _invoke_agent(similar_agent, state)

def revenue_agent_node(state: MessagesState):
    return _invoke_agent(revenue_agent, state)

def rating_agent_node(state: MessagesState):
    return _invoke_agent(rating_agent, state)

def director_cast_agent_node(state: MessagesState):
    return _invoke_agent(dir_cast_agent, state)

def era_agent_node(state: MessagesState):
    return _invoke_agent(era_agent_exec, state)


# ─────────────────────────────────────────────
# RAG SUBGRAPH
# ─────────────────────────────────────────────
rag_graph = StateGraph(MessagesState)
rag_graph.add_node("rag_supervisor", rag_supervisor_node)
rag_graph.add_node("semantic_search_agent", semantic_agent_node)
rag_graph.add_node("similar_movie_agent", similar_agent_node)
rag_graph.set_entry_point("rag_supervisor")
rag_graph.add_conditional_edges("rag_supervisor", rag_supervisor_route)
rag_graph.add_edge("semantic_search_agent", END)
rag_graph.add_edge("similar_movie_agent", END)
rag_subgraph = rag_graph.compile()


# ─────────────────────────────────────────────
# SQL SUBGRAPH
# ─────────────────────────────────────────────
sql_graph = StateGraph(MessagesState)
sql_graph.add_node("sql_supervisor", sql_supervisor_node)
sql_graph.add_node("revenue_agent", revenue_agent_node)
sql_graph.add_node("rating_agent", rating_agent_node)
sql_graph.add_node("director_cast_agent", director_cast_agent_node)
sql_graph.add_node("era_agent", era_agent_node)
sql_graph.set_entry_point("sql_supervisor")
sql_graph.add_conditional_edges("sql_supervisor", sql_supervisor_route)
sql_graph.add_edge("revenue_agent", END)
sql_graph.add_edge("rating_agent", END)
sql_graph.add_edge("director_cast_agent", END)
sql_graph.add_edge("era_agent", END)
sql_subgraph = sql_graph.compile()


# ─────────────────────────────────────────────
# MAIN GRAPH
# ─────────────────────────────────────────────
def rag_department_node(state: MessagesState):
    result = rag_subgraph.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    return {"messages": result["messages"]}


def sql_department_node(state: MessagesState):
    result = sql_subgraph.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    return {"messages": result["messages"]}

def web_department_node(state: MessagesState):
    result = web_search_agent.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    return {"messages": result["messages"]}


main_graph = StateGraph(MessagesState)
main_graph.add_node("supervisor", main_supervisor_node)

main_graph.add_node("RAG_DEPARTMENT", rag_department_node)
main_graph.add_node("SQL_DEPARTMENT", sql_department_node)
main_graph.add_node("WEB_DEPARTMENT", web_department_node)

main_graph.set_entry_point("supervisor")
main_graph.add_conditional_edges("supervisor", main_supervisor_route)

main_graph.add_edge("RAG_DEPARTMENT", "supervisor")
main_graph.add_edge("SQL_DEPARTMENT", "supervisor")
main_graph.add_edge("WEB_DEPARTMENT", "supervisor")

graph = main_graph.compile()

# ─────────────────────────────────────────────
# HELPER FUNCTION (dipanggil dari app.py)
# ─────────────────────────────────────────────
def run_agent(user_message: str, chat_history: list = None, session_id: str = None) -> dict:
    messages = (chat_history or []) + [HumanMessage(content=user_message)]

    config = {
    "callbacks": [],
    "metadata": {"session_id": session_id or "default"}
}
    result = graph.invoke(
    {"messages": messages}, 
    config={
        **config,
        "recursion_limit": 10
    }
)


    ROUTING_KEYWORDS = {"FINISH", "RAG_DEPARTMENT", "SQL_DEPARTMENT",
                        "WEB_DEPARTMENT", "__end__", "revenue_agent",
                        "rating_agent", "director_cast_agent", "era_agent",
                        "semantic_search_agent", "similar_movie_agent",
                        "web_search_agent"}

    ai_response = "Maaf, tidak ada jawaban."
    for msg in reversed(result["messages"]):
        content = msg.content.strip() if isinstance(msg.content, str) else ""
        if content and content.upper() not in ROUTING_KEYWORDS:
            ai_response = content
            break

    return {
        "response": ai_response,
        "messages": result["messages"],
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Rekomendasiin film yang vibenya kayak Inception",
        "Film terlaris sepanjang masa apa?",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = run_agent(query, session_id="test-session")
        print(result["response"])

    print("\n✅ Cek Langfuse dashboard untuk melihat traces!")
"""
rag_streamlit_app.py
───────────────────────────────
Streamlit interface for your RAG → Yahoo → RSS financial assistant.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Import Qdrant client (auto-initializes + creates collection if needed)
from AI_Chatbot.clients.qdrant_client import qdrant, COLLECTION_NAME, initialize_qdrant

# Import pipeline logic
from AI_Chatbot.pipeline.nodes import (
    semantic_search,
    summarize,
    extract_ticker,
    yahoo_fetch_with_fallback,
    summarize_articles,
    PipelineState,
)

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ Missing GEMINI_API_KEY in .env file. Please add it and restart.")
    st.stop()

# ─────────────────────────────────────────────
# Initialize Qdrant collection at startup
# ─────────────────────────────────────────────
with st.spinner("⚙️ Initializing Qdrant collection..."):
    try:
        initialize_qdrant()
        st.success(f"✅ Qdrant ready — using collection `{COLLECTION_NAME}`.")
    except Exception as e:
        st.error(f"❌ Qdrant initialization failed: {e}")
        st.stop()

# ─────────────────────────────────────────────
# Build Workflow Graph
# ─────────────────────────────────────────────
graph = (
    StateGraph(PipelineState)
    .add_node("semantic_search", semantic_search)
    .add_node("summarize_rag", summarize)
    .add_node("extract_ticker", extract_ticker)
    .add_node("yahoo_fetch_with_fallback", yahoo_fetch_with_fallback)
    .add_node("summarize_articles", summarize_articles)
    .add_edge(START, "semantic_search")
    .add_conditional_edges(
        "semantic_search",
        lambda s: "summarize_rag" if s.get("retrieved_docs") else "extract_ticker",
        {"summarize_rag": "summarize_rag", "extract_ticker": "extract_ticker"},
    )
    .add_edge("summarize_rag", END)
    .add_edge("extract_ticker", "yahoo_fetch_with_fallback")
    .add_edge("yahoo_fetch_with_fallback", "summarize_articles")
    .add_edge("summarize_articles", END)
    .compile()
)

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="📈 Financial RAG Chatbot", layout="wide")
st.title("📈 Financial Assistant — Gemini + Qdrant + Yahoo")

st.markdown(
    "Type a question about **stocks or companies**, and I’ll use **Qdrant first** "
    "for RAG-based answers, then **Yahoo Finance** or **RSS** as fallback."
)

query = st.text_input("💬 Your question:", placeholder="e.g. What’s new about AAPL?")
submit = st.button("Run Analysis")

if submit and query:
    with st.spinner("🔎 Analyzing... please wait..."):
        state = {"query": query}
        result = graph.invoke(state)

        answer = result.get("answer", "No answer generated.")
        source = result.get("source", "unknown")

        st.markdown("### 🧠 Summary")
        st.write(answer)
        st.markdown(f"**Source:** `{source}`")

        # Display retrieved docs if RAG was used
        if result.get("retrieved_docs"):
            st.markdown("### 📚 Retrieved from Qdrant")
            for i, doc in enumerate(result["retrieved_docs"], start=1):
                st.markdown(f"**{i}. {doc['title']} ({doc['ticker']})** — Score: {doc['score']:.3f}")
                st.text(doc["full_text"][:250] + "…")

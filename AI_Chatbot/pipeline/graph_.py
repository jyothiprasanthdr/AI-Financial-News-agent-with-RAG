# AI_Chatbot/pipeline/graph.py
# ─────────────────────────────────────────────
# Build LangGraph workflow:
#   START → semantic_search → summarize_rag OR extract_ticker
#   extract_ticker → yahoo_fetch → summarize_yahoo OR rss_fallback
#   summarize_yahoo / rss_fallback / summarize_rag → END
# ─────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from AI_Chatbot.pipeline.nodes import (
    semantic_search,
    summarize,
    extract_ticker,
    yahoo_fetch_with_fallback,
    summarize_articles,
    PipelineState,
    get_yahoo_rss_news,
)
import sys


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────
def route_after_semantic(state: PipelineState) -> str:
    """Route after semantic_search → summarize_rag or extract_ticker."""
    docs = state.get("retrieved_docs")
    if docs and len(docs) > 0:
        return "summarize_rag"
    print("[ROUTE] ❌ No RAG docs found → extracting ticker.")
    return "extract_ticker"


def route_after_yahoo(state: PipelineState) -> str:
    """Route after yahoo_fetch → summarize_yahoo or rss_fallback."""
    fetched = state.get("fetched_articles", [])
    src = state.get("source", "none")
    if fetched and len(fetched) > 0 and src == "yahoo_api":
        print("[ROUTE] ✅ Yahoo API success → summarize_yahoo.")
        return "summarize_yahoo"
    print("[ROUTE] ⤵️ Yahoo failed/empty → fallback to RSS.")
    return "rss_fallback"


# ─────────────────────────────────────────────
# Node wrapper aliases for clarity
# ─────────────────────────────────────────────
def semantic_search_node(state): return semantic_search(state)
def summarize_rag_node(state): return summarize(state)
def extract_ticker_node(state): return extract_ticker(state)
def yahoo_fetch_node(state): return yahoo_fetch_with_fallback(state)
def summarize_yahoo_node(state): return summarize_articles(state)
def rss_fallback_node(state):
    """RSS-only fallback summarization if Yahoo fetch fails."""
    ticker = state.get("ticker", "Unknown")
    rss_articles = get_yahoo_rss_news(ticker)
    if not rss_articles:
        print(f"[WARN] RSS returned empty for {ticker}.")
        return {"answer": f"No RSS news found for {ticker}.", "source": "rss_feed"}
    return summarize_articles({
        "ticker": ticker,
        "fetched_articles": rss_articles,
        "source": "rss_feed"
    })


# ─────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────
def build_graph():
    """Constructs the entire workflow graph."""
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("summarize_rag", summarize_rag_node)
    workflow.add_node("extract_ticker", extract_ticker_node)
    workflow.add_node("yahoo_fetch", yahoo_fetch_node)
    workflow.add_node("summarize_yahoo", summarize_yahoo_node)
    workflow.add_node("rss_fallback", rss_fallback_node)

    # Start → semantic_search
    workflow.add_edge(START, "semantic_search")

    # Conditional: semantic_search → summarize_rag OR extract_ticker
    workflow.add_conditional_edges("semantic_search", route_after_semantic, {
        "summarize_rag": "summarize_rag",
        "extract_ticker": "extract_ticker"
    })

    # summarize_rag → END
    workflow.add_edge("summarize_rag", END)

    # extract_ticker → yahoo_fetch
    workflow.add_edge("extract_ticker", "yahoo_fetch")

    # Conditional: yahoo_fetch → summarize_yahoo OR rss_fallback
    workflow.add_conditional_edges("yahoo_fetch", route_after_yahoo, {
        "summarize_yahoo": "summarize_yahoo",
        "rss_fallback": "rss_fallback"
    })

    # Summarize nodes → END
    workflow.add_edge("summarize_yahoo", END)
    workflow.add_edge("rss_fallback", END)

    print("[INFO] ✅ Workflow graph successfully built.")
    return workflow.compile()


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    g = build_graph()
    print("📈 Graph structure:")
    print(g.get_graph().draw_ascii())

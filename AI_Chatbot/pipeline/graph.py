import os
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

# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────
def route_after_semantic(state: PipelineState) -> str:
    """Route after semantic_search → summarize_rag or extract_ticker."""
    docs = state.get("retrieved_docs")
    if docs and len(docs) > 0:
        print("[ROUTE] ✅ Docs found in RAG → summarize_rag.")
        return "summarize_rag"
    print("[ROUTE] ❌ No RAG docs found → extract_ticker.")
    return "extract_ticker"


def route_after_yahoo(state: PipelineState) -> str:
    """Route after yahoo_fetch → summarize_yahoo or rss_fallback."""
    fetched = state.get("fetched_articles", [])
    src = state.get("source", "none")
    if fetched and len(fetched) > 0 and src == "yahoo_api":
        print("[ROUTE] ✅ Yahoo API success → summarize_yahoo.")
        return "summarize_yahoo"
    print("[ROUTE] ⤵️ Yahoo API failed or empty → rss_fallback.")
    return "rss_fallback"


# ─────────────────────────────────────────────
# Node wrappers
# ─────────────────────────────────────────────
def semantic_search_node(state): return semantic_search(state)
def summarize_rag_node(state): return summarize(state)
def extract_ticker_node(state): return extract_ticker(state)
def yahoo_fetch_node(state): return yahoo_fetch_with_fallback(state)
def summarize_yahoo_node(state): return summarize_articles(state)
def rss_fallback_node(state):
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
# Build Graph
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

    # semantic_search → summarize_rag OR extract_ticker
    workflow.add_conditional_edges(
        "semantic_search",
        route_after_semantic,
        {"summarize_rag": "summarize_rag", "extract_ticker": "extract_ticker"}
    )

    # summarize_rag → END
    workflow.add_edge("summarize_rag", END)

    # extract_ticker → yahoo_fetch
    workflow.add_edge("extract_ticker", "yahoo_fetch")

    # yahoo_fetch → summarize_yahoo OR rss_fallback
    workflow.add_conditional_edges(
        "yahoo_fetch",
        route_after_yahoo,
        {"summarize_yahoo": "summarize_yahoo", "rss_fallback": "rss_fallback"}
    )

    # summarize_yahoo & rss_fallback → END
    workflow.add_edge("summarize_yahoo", END)
    workflow.add_edge("rss_fallback", END)

    print("[INFO] ✅ Workflow graph successfully built.")
    return workflow.compile()

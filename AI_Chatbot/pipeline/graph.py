"""Graph builder for the RAG workflow.

Provides routing and node wrapper functions and a `build_graph` helper that
returns a compiled StateGraph ready for invocation.
"""

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


def route_after_semantic(state: PipelineState) -> str:
    """Return next node name based on whether semantic search returned docs."""
    docs = state.get("retrieved_docs")
    if docs and len(docs) > 0:
        return "summarize_rag"
    return "extract_ticker"


def route_after_yahoo(state: PipelineState) -> str:
    """Return next node name after yahoo_fetch based on results."""
    fetched = state.get("fetched_articles", [])
    src = state.get("source", "none")
    if fetched and len(fetched) > 0 and src == "yahoo_api":
        return "summarize_yahoo"
    return "rss_fallback"


def semantic_search_node(state):
    return semantic_search(state)


def summarize_rag_node(state):
    return summarize(state)


def extract_ticker_node(state):
    return extract_ticker(state)


def yahoo_fetch_node(state):
    return yahoo_fetch_with_fallback(state)


def summarize_yahoo_node(state):
    return summarize_articles(state)


def rss_fallback_node(state):
    ticker = state.get("ticker", "Unknown")
    rss_articles = get_yahoo_rss_news(ticker)
    if not rss_articles:
        return {"answer": f"No RSS news found for {ticker}.", "source": "rss_feed"}
    return summarize_articles({
        "ticker": ticker,
        "fetched_articles": rss_articles,
        "source": "rss_feed",
    })


def build_graph():
    """Construct and compile the workflow StateGraph."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("summarize_rag", summarize_rag_node)
    workflow.add_node("extract_ticker", extract_ticker_node)
    workflow.add_node("yahoo_fetch", yahoo_fetch_node)
    workflow.add_node("summarize_yahoo", summarize_yahoo_node)
    workflow.add_node("rss_fallback", rss_fallback_node)

    workflow.add_edge(START, "semantic_search")

    workflow.add_conditional_edges(
        "semantic_search",
        route_after_semantic,
        {"summarize_rag": "summarize_rag", "extract_ticker": "extract_ticker"},
    )

    workflow.add_edge("summarize_rag", END)
    workflow.add_edge("extract_ticker", "yahoo_fetch")

    workflow.add_conditional_edges(
        "yahoo_fetch",
        route_after_yahoo,
        {"summarize_yahoo": "summarize_yahoo", "rss_fallback": "rss_fallback"},
    )

    workflow.add_edge("summarize_yahoo", END)
    workflow.add_edge("rss_fallback", END)

    return workflow.compile()

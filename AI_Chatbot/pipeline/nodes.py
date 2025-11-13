from typing_extensions import TypedDict
from qdrant_client import QdrantClient
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import feedparser, requests, os
from AI_Chatbot.clients.yahoo_client import get_yahoo_news

# ─────────────────────────────────────────────
# Load environment & model setup
# ─────────────────────────────────────────────
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.0-flash"
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "news_embeddings"

# ─────────────────────────────────────────────
# Define pipeline state
# ─────────────────────────────────────────────
class PipelineState(TypedDict):
    query: str
    retrieved_docs: list
    answer: str
    ticker: str
    fetched_articles: list
    source: str


# ─────────────────────────────────────────────
# 1️⃣ Semantic Search (RAG)
# ─────────────────────────────────────────────
def semantic_search(state: PipelineState) -> dict:
    """Retrieve top-k docs with cosine similarity ≥ threshold."""
    query = state["query"]
    query_vec = embedder.encode([query])[0].tolist()

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=3,
        with_payload=True,
    )

    filtered = [r for r in results if r.score >= 0.5]
    if not filtered:
        print("[INFO] ❌ No docs found in RAG → fallback to Yahoo")
        return {"retrieved_docs": None}

    retrieved_docs = [
        {
            "title": r.payload.get("title", "Untitled"),
            "ticker": r.payload.get("ticker", ""),
            "full_text": r.payload.get("full_text", ""),
            "score": r.score,
        }
        for r in filtered
    ]
    return {"retrieved_docs": retrieved_docs}


# ─────────────────────────────────────────────
# 2️⃣ Summarize from RAG
# ─────────────────────────────────────────────
def summarize(state: PipelineState) -> dict:
    """Use Gemini to answer strictly from retrieved context."""
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {"answer": "No relevant data found.", "source": "rag_empty"}

    # Build context from retrieved docs
    context = "\n\n".join(
        [f"[{i+1}] {doc['title']} ({doc['ticker']})\n{doc['full_text']}"
         for i, doc in enumerate(retrieved_docs)]
    )[:7000]

    prompt = f"""
You are a financial assistant. Answer the question *only* using the following context.
If the context is irrelevant to the question, reply EXACTLY: "No relevant data found."

Context:
{context}

Question: {query}
Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    answer_text = response.text.strip() if response and response.text else "No relevant data found."

    # detect fallback trigger
    if "No relevant data found" in answer_text:
        return {"answer": answer_text, "source": "rag_empty"}
    
    return {"answer": answer_text, "source": "rag"}



# ─────────────────────────────────────────────
# 3️⃣ Extract Ticker
# ─────────────────────────────────────────────
def extract_ticker(state: PipelineState) -> dict:
    """
    Extract or infer ticker symbols from user query using Gemini.
    Output is normalized for both yfinance and Yahoo RSS feeds.
    """
    query = state.get("query", "").strip()
    if not query:
        print("[WARN] Empty query in extract_ticker.")
        return {"ticker": "N/A"}

    # ─────────────────────────────
    # Step 1: Ask Gemini to extract ticker(s)
    # ─────────────────────────────
    extract_prompt = f"""
You are a **financial data assistant**.

Task:
Extract the **official stock ticker symbol(s)** for any company or organization 
mentioned in the user's question below.

Rules:
• Output only ticker symbols (e.g., AAPL, TSLA, GOOGL, MSFT, TCS.NS)
• If multiple tickers, return comma-separated (no spaces)
• If unsure, infer from company name (e.g., Google → GOOG, TCS → TCS.NS)
• If no company is found, return exactly: N/A
• Do NOT include explanations or text — only tickers.

User question:
{query}

Answer:
"""

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[extract_prompt],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=20),
        )
        raw_ticker = resp.text.strip() if resp and resp.text else "N/A"
    except Exception as e:
        print(f"[ERROR] extract_ticker failed: {e}")
        raw_ticker = "N/A"

    # ─────────────────────────────
    # Step 2: Normalize ticker(s)
    # ─────────────────────────────
    ticker = (
        raw_ticker.replace('"', "")
        .replace("'", "")
        .replace(" ", "")
        .replace("\n", "")
        .upper()
    )

    # Sometimes Gemini outputs weird stuff like: "Ticker: AAPL"
    ticker = ticker.replace("TICKER:", "").replace("SYMBOL:", "").strip(",")

    # Normalize multiple tickers: AAPL,,TSLA → AAPL,TSLA
    tickers = [t for t in ticker.split(",") if t and t != "N/A"]
    if not tickers:
        print(f"[WARN] No ticker extracted from query: {query}")
        return {"ticker": "N/A"}

    # For consistency across APIs (yfinance + RSS)
    # Ensure all tickers are uppercased and have no whitespace
    clean_tickers = ",".join(sorted(set(tickers)))
    print(f"[INFO] Extracted ticker(s): {clean_tickers}")
    return {"ticker": clean_tickers}

# ─────────────────────────────────────────────
# 4️⃣ Helper: RSS + Scraper
# ─────────────────────────────────────────────
def scrape_yahoo_article(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.select("article p") or soup.select("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        if len(text) < 200:
            desc = soup.find("meta", {"name": "description"})
            if desc and desc.get("content"):
                text = desc["content"]
        return text.strip()
    except Exception as e:
        print(f"[WARN] Scrape failed for {url}: {e}")
        return ""


def get_yahoo_rss_news(ticker: str, num_articles: int = 5):
    """Fetch RSS feed as fallback (no API key)."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker.upper()}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    if not feed.entries:
        return []
    articles = []
    for e in feed.entries[:num_articles]:
        full_text = scrape_yahoo_article(e.link) or e.get("summary", "")
        articles.append(
            {
                "ticker": ticker.upper(),
                "title": e.title,
                "link": e.link,
                "summary": e.get("summary", ""),
                "full_text": full_text,
                "published": e.get("published", ""),
            }
        )
    return articles


# ─────────────────────────────────────────────
# 5️⃣ Yahoo Fetch → RSS Fallback
# ─────────────────────────────────────────────

def yahoo_fetch_with_fallback(state: PipelineState) -> dict:
    """Try Yahoo Finance API first; if empty or error, fallback to RSS feed."""
    ticker = state.get("ticker", "N/A")
    if not ticker or ticker == "N/A":
        print("[WARN] No ticker extracted → cannot fetch news")
        return {"fetched_articles": [], "source": "none"}

    all_articles = []
    source = "none"

    # ① Try Yahoo API
    try:
        articles = get_yahoo_news(ticker, num_articles=5)
        if isinstance(articles, list) and len(articles) > 0:
            print(f"[INFO] ✅ Yahoo API returned {len(articles)} articles for {ticker}")
            return {"fetched_articles": articles, "source": "yahoo_api"}
        else:
            print(f"[WARN] Yahoo API returned empty for {ticker}")
    except Exception as e:
        print(f"[ERROR] YahooFinanceNewsTool failed for {ticker}: {e}")

    # ② Fallback → Yahoo RSS feed
    try:
        print(f"[INFO] ⤵️ Falling back to RSS for {ticker}")
        rss_articles = get_yahoo_rss_news(ticker, num_articles=5)
        if isinstance(rss_articles, list) and len(rss_articles) > 0:
            print(f"[INFO] ✅ RSS fallback returned {len(rss_articles)} for {ticker}")
            return {"fetched_articles": rss_articles, "source": "rss_feed"}
        else:
            print(f"[WARN] RSS returned empty for {ticker}")
    except Exception as e:
        print(f"[ERROR] RSS fetch failed for {ticker}: {e}")

    # ③ Nothing found
    print(f"[FAIL] ❌ No news sources found for {ticker}")
    return {"fetched_articles": [], "source": "none"}


# ─────────────────────────────────────────────
# 6️⃣ Summarize Articles (Yahoo or RSS)
# ─────────────────────────────────────────────
def summarize_articles(state: PipelineState) -> dict:
    """Summarize fetched Yahoo/RSS articles for a ticker."""
    ticker = state.get("ticker", "Unknown")
    articles = state.get("fetched_articles", [])
    source = state.get("source", "unknown")

    if not articles:
        return {"answer": f"No recent articles found for {ticker}.", "source": source}

    # Build readable context
    context = "\n\n".join(
        [
            f"[{i+1}] {a.get('title', 'Untitled')}\n{a.get('summary', '')}\n{a.get('full_text', '')}\n{a.get('link', '')}"
            for i, a in enumerate(articles)
        ]
    )[:7000]

    prompt = f"""
You are a financial assistant. 
Summarize the following {source.replace('_', ' ')} articles about {ticker}.
Highlight market updates, analyst opinions, and investor sentiment.

Articles:
{context}

Summary:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        answer_text = response.text.strip() if response and response.text else "No relevant data found."
    except Exception as e:
        print(f"[ERROR] summarize_articles failed for {ticker}: {e}")
        answer_text = "No relevant data found."

    return {"answer": answer_text, "source": source}

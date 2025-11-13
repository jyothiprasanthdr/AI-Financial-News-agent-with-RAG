"""Yahoo Finance news fetcher utilities.

Provides a helper that returns recent Yahoo Finance news articles for a
given ticker or comma-separated tickers. The function normalizes the
tool output and attempts to scrape article bodies when a link is available.
"""

from bs4 import BeautifulSoup
import requests
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


# Initialize the Yahoo news tool once.
yahoo_news_tool = YahooFinanceNewsTool()


def get_yahoo_news(ticker: str, num_articles: int = 5):
    """Return a list of normalized article dictionaries for the ticker.

    Each dict contains: ticker, title, link, summary (snippet), and
    full_text (when scraping succeeds).
    """
    parsed_articles = []
    tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]

    for t in tickers:
        try:
            raw_articles = yahoo_news_tool.run({"query": t, "num_articles": num_articles})
        except Exception as e:
            print(f"Yahoo: tool failed for {t}: {e}")
            continue

        if isinstance(raw_articles, str) or not raw_articles:
            # Tool may return a string message or an empty result.
            continue

        for item in raw_articles:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "Untitled")
            link = item.get("link", "")
            if not link:
                continue

            full_text = ""
            try:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
                resp = requests.get(link, headers=headers, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    paragraphs = soup.select("article p") or soup.select("p")
                    full_text = " ".join(p.get_text(strip=True) for p in paragraphs[:15])
            except Exception:
                # Scraping failures are non-fatal; proceed with what we have.
                full_text = item.get("summary", "") or ""

            parsed_articles.append(
                {
                    "ticker": t,
                    "title": title,
                    "link": link,
                    "summary": full_text[:500] if full_text else "",
                    "full_text": full_text or "",
                }
            )

    return parsed_articles

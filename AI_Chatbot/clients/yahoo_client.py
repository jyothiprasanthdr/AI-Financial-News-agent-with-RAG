# AI_Chatbot/clients/yahoo_client.py

from bs4 import BeautifulSoup
import requests
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

# Initialize once globally
yahoo_news_tool = YahooFinanceNewsTool()

def get_yahoo_news(ticker: str, num_articles: int = 5):
    """
    Fetch recent Yahoo Finance news articles for one or more tickers.
    Handles malformed tickers, multi-ticker inputs, and HTML scraping.
    Returns a normalized list of article dicts for downstream summarization.
    """
    parsed_articles = []
    tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]

    for t in tickers:
        try:
            # ✅ Correct input: YahooFinanceNewsTool expects "query"
            raw_articles = yahoo_news_tool.run({
                "query": t,
                "num_articles": num_articles
            })
        except Exception as e:
            print(f"[ERROR] YahooFinanceNewsTool failed for {t}: {e}")
            continue

        # Handle tool returning string (error or info message)
        if isinstance(raw_articles, str):
            print(f"[INFO] YahooFinanceNewsTool returned string output for {t}.")
            continue

        if not raw_articles:
            print(f"[WARN] No Yahoo articles found for {t}")
            continue

        # Parse each article result
        for item in raw_articles:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "Untitled")
            link = item.get("link", "")
            if not link:
                continue

            # Scrape the body text if possible
            full_text = ""
            try:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
                resp = requests.get(link, headers=headers, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    paragraphs = soup.select("article p") or soup.select("p")
                    full_text = " ".join(p.get_text(strip=True) for p in paragraphs[:15])
            except Exception as e:
                print(f"[WARN] Could not scrape article for {t}: {e}")

            parsed_articles.append({
                "ticker": t,
                "title": title,
                "link": link,
                "summary": full_text[:500] if full_text else "",
                "full_text": full_text or "",
            })

    # Summary logs
    if parsed_articles:
        print(f"[INFO] ✅ Retrieved {len(parsed_articles)} total articles for {', '.join(tickers)}")
    else:
        print(f"[WARN] ❌ Yahoo Finance returned no articles for {', '.join(tickers)}")

    return parsed_articles

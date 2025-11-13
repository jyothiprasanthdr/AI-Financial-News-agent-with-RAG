"""
Quick test for Yahoo Finance article fetch
— both API and RSS fallback.
"""

from AI_Chatbot.pipeline.nodes import yahoo_fetch_with_fallback

def test_yahoo_news(ticker="GOOG"):
    state = {"ticker": ticker}
    result = yahoo_fetch_with_fallback(state)
    print("📊 Result:")
    print(f"Source: {result.get('source')}")
    articles = result.get("fetched_articles", [])
    print(f"Fetched {len(articles)} articles\n")
    for i, a in enumerate(articles[:3]):
        print(f"📰 {i+1}. {a.get('title')}")
        print(f"URL: {a.get('link')}")
        print(f"Snippet: {a.get('summary', '')[:180]}")
        print("-" * 80)

if __name__ == "__main__":
    test_yahoo_news("GOOG")

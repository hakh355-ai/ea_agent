"""
News Agent (OpenAI GPT-4o).
Fetches Bloomberg RSS, Reuters RSS, and NewsAPI headlines,
sends them to GPT-4o for sentiment analysis, caches result 5 minutes.
"""
import asyncio
import math
import os
import sys
import time
from pathlib import Path

import feedparser
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

RSS_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
]

ANALYSIS_SYSTEM = """You are a professional financial market analyst.
Analyze the following news headlines for their impact on forex, gold, indices, and crypto markets.

Return ONLY valid JSON — no markdown, no explanation:
{
  "sentiment_score": <float from -1.0 (strongly bearish) to 1.0 (strongly bullish)>,
  "risk_flags": [],
  "summary": "<2-3 sentence market overview>",
  "high_impact_events": []
}

risk_flags rules (STRICT):
- ONLY add a flag if a headline EXPLICITLY states the exact time of a high-impact event (NFP, FOMC, CPI, ECB, BOE, BOJ).
- Format: "<EVENT> in <N>min" or "<EVENT> in <N>h" — only if the headline gives a specific time within 4 hours.
- If no headline gives an explicit near-term time for an event, risk_flags MUST be an empty array [].
- Do NOT guess, infer, or copy example formats. If uncertain, leave risk_flags empty."""


async def _fetch_rss(url: str, client: httpx.AsyncClient) -> list[str]:
    try:
        r = await client.get(url, timeout=8.0, follow_redirects=True)
        feed = feedparser.parse(r.text)
        return [e.title for e in feed.entries[:12]]
    except Exception:
        return []


async def _fetch_newsapi(client: httpx.AsyncClient) -> list[str]:
    key = os.getenv("NEWSAPI_KEY", "")
    if not key:
        return []
    try:
        r = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "forex OR gold OR federal reserve OR ECB OR inflation OR interest rate",
                "apiKey": key,
                "pageSize": 10,
                "sortBy": "publishedAt",
                "language": "en",
            },
            timeout=8.0,
        )
        articles = r.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception:
        return []


async def get_sentiment(symbols: list[str]) -> dict:
    import bridge.state as state

    cache_key = str(math.floor(time.time() / 2400))  # cache ~40 min
    cached = state.get_cached_news(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_fetch_rss(url, client) for url in RSS_FEEDS],
            _fetch_newsapi(client),
            return_exceptions=True,
        )

    headlines = []
    for r in results:
        if isinstance(r, list):
            headlines.extend(r)

    fallback = {
        "sentiment_score": 0.0,
        "risk_flags": [],
        "summary": "No news available.",
        "high_impact_events": [],
    }

    if not headlines:
        state.set_news_cache(cache_key, fallback)
        return fallback

    context = "\n".join(f"- {h}" for h in headlines[:30])
    user_msg = f"Trading instruments: {', '.join(symbols)}\n\nNews headlines:\n{context}"

    try:
        client_ai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.1,
        )
        import json
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[news_agent] GPT-4o error: {e}", file=sys.stderr)
        result = fallback

    state.set_news_cache(cache_key, result)
    return result


if __name__ == "__main__":
    import json
    result = asyncio.run(get_sentiment(["EURUSD", "XAUUSD", "BTCUSD"]))
    print(json.dumps(result, indent=2))

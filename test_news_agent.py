"""Quick test for news_agent — run: python test_news_agent.py"""
import asyncio, time, sys
sys.path.insert(0, ".")

async def main():
    print("Testing news_agent.get_sentiment()...")
    t0 = time.time()
    from agents.news_agent import get_sentiment
    result = await get_sentiment(["EURUSD"])
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"  sentiment_score : {result.get('sentiment_score')}")
    print(f"  risk_flags      : {result.get('risk_flags')}")
    print(f"  summary         : {result.get('summary', '')[:120]}")

asyncio.run(main())

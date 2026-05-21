"""Quick test for trading_agent — run: python test_trading_agent.py"""
import asyncio, time, sys, json
sys.path.insert(0, ".")

DUMMY_OHLC = {
    "M5": [{"t": "2026-05-09 10:00:00", "o": 1.0821, "h": 1.0835, "l": 1.0818, "c": 1.0830, "v": 1234}] * 20,
    "H1": [{"t": "2026-05-09 10:00:00", "o": 1.0810, "h": 1.0840, "l": 1.0805, "c": 1.0830, "v": 5000}] * 10,
    "H4": [{"t": "2026-05-09 08:00:00", "o": 1.0800, "h": 1.0850, "l": 1.0790, "c": 1.0830, "v": 15000}] * 5,
}
DUMMY_INDICATORS = {"rsi": 52.3, "atr": 0.0012, "ema20": 1.0825, "ema50": 1.0815,
                    "macd": 0.0003, "macd_signal": 0.0001, "adx": 22.0,
                    "bb_width": 0.005, "vpa_signal": "neutral"}
DUMMY_NEWS = {"sentiment_score": 0.3, "risk_flags": [], "summary": "Stable market."}
DUMMY_REGIME_PARAMS = {"sl_atr_multiplier": 1.5, "tp_sl_ratio": 2.0,
                        "min_confidence_threshold": 0.65}
DUMMY_PARAMS = {"trading_prompt": "", "min_confidence_threshold": 0.65}

async def main():
    print("Testing trading_agent.get_signal()...")
    t0 = time.time()
    from agents.trading_agent import get_signal
    result = await get_signal("EURUSD", DUMMY_OHLC, DUMMY_INDICATORS, DUMMY_NEWS,
                              "ranging", DUMMY_REGIME_PARAMS, DUMMY_PARAMS)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"  action     : {result.get('action')}")
    print(f"  confidence : {result.get('confidence')}")
    print(f"  reasoning  : {str(result.get('reasoning', ''))[:120]}")

asyncio.run(main())

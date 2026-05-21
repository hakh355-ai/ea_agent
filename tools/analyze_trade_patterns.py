"""
Stufe 2: Analyze closed trades, extract learned rules, update strategy_params.json.
Runs automatically after every 50 fills (triggered by scheduler).
Usage: python tools/analyze_trade_patterns.py [--min-trades 50]
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from tools.trade_logger import read_recent_trades

load_dotenv()

SYSTEM = """You are a quantitative trading analyst. You will receive a list of recent trades
with their entry context (indicators, news sentiment, regime) and outcome (win/loss, PnL).

Identify patterns in what works and what fails. Return ONLY valid JSON:
{
  "learned_rules": [
    "<specific actionable rule based on observed patterns>",
    "<another rule>"
  ],
  "win_rate": <float 0-1>,
  "avg_pnl": <float>,
  "insights": "<2-3 sentence summary>"
}

Rules must be specific and measurable, e.g.:
- "Avoid BUY signals when news_sentiment < -0.3 (historical win rate 23% in this case)"
- "In 'volatile' regime, require confidence >= 0.80 before entering (0.65-0.79 shows 31% win rate)"
- "XAUUSD performs best when ADX > 30 and macd_hist > 0"
Limit to the 5 most impactful rules."""


def analyze(min_trades: int = 50) -> dict:
    trades = read_recent_trades(days=30)
    closes = [t for t in trades if t["type"] == "close"]

    if len(closes) < min_trades:
        print(f"Only {len(closes)} closed trades (need {min_trades}). Skipping analysis.")
        return {}

    # Match closes with their signal context
    signals_by_symbol = {}
    for t in trades:
        if t["type"] == "signal":
            signals_by_symbol[t["symbol"]] = t  # last signal per symbol (approximation)

    summary = []
    for close in closes[-150:]:
        sig = signals_by_symbol.get(close.get("symbol"), {})
        summary.append({
            "symbol":         close.get("symbol"),
            "outcome":        close.get("outcome"),
            "pnl":            close.get("pnl", 0),
            "regime":         sig.get("regime"),
            "rsi":            sig.get("rsi"),
            "adx":            sig.get("adx"),
            "macd_hist":      sig.get("macd_hist"),
            "bb_width":       sig.get("bb_width"),
            "news_sentiment": sig.get("news_sentiment"),
            "confidence":     sig.get("confidence"),
            "action":         sig.get("action"),
        })

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": json.dumps(summary, indent=2)},
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
        temperature=0.1,
    )
    result = json.loads(response.choices[0].message.content)

    params_path = Path(os.getenv("STRATEGY_PARAMS_PATH", ".tmp/strategy_params.json"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["learned_rules"] = result.get("learned_rules", [])
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))

    print(f"Win rate : {result.get('win_rate', 0):.1%}")
    print(f"Avg PnL  : {result.get('avg_pnl', 0):.2f}")
    print(f"Rules    : {len(result.get('learned_rules', []))} updated in strategy_params.json")
    print(f"Insights : {result.get('insights', '')}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-trades", type=int, default=50)
    args = parser.parse_args()
    analyze(args.min_trades)

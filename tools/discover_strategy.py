"""
Stufe 4a: Ask GPT-4o to propose entirely new trading strategies.
Strategies are added to strategy_pool in strategy_params.json with status "candidate".
Usage: python tools/discover_strategy.py [--candidates 3]
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM = """You are a quantitative trading strategy researcher with expertise in technical analysis.
Propose a NEW and specific trading strategy using combinations of these indicators:
RSI(14), ATR(14), EMA20, EMA50, MACD(12,26,9), ADX(14), Bollinger Band Width, news_sentiment score.

The strategy must be concise, testable, and different from existing ones listed by the user.

Return ONLY valid JSON:
{
  "strategy_name": "<short descriptive name>",
  "description": "<2-3 sentences>",
  "entry_rules": {
    "buy":  ["<condition 1 using indicator thresholds>", "<condition 2>"],
    "sell": ["<condition 1>", "<condition 2>"]
  },
  "exit_rules": {
    "sl_atr_multiplier": <float>,
    "tp_sl_ratio": <float>,
    "min_confidence": <float 0-1>
  },
  "best_regimes":     ["trending"|"ranging"|"volatile"|"quiet"],
  "best_instruments": ["EURUSD"|"GBPUSD"|"USDJPY"|"AUDUSD"|"XAUUSD"|"US500"|"GER40"|"BTCUSD"|"ETHUSD"],
  "hypothesis": "<why this setup should produce edge>"
}"""


def discover(n_candidates: int = 3) -> list[dict]:
    params_path = Path(os.getenv("STRATEGY_PARAMS_PATH", ".tmp/strategy_params.json"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    pool = params.get("strategy_pool", [])
    existing_names = [s.get("strategy_name", "") for s in pool]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    new_strategies = []

    print(f"Discovering {n_candidates} new strategy candidates...")
    for i in range(n_candidates):
        context = (
            f"Already in pool (must be different): {existing_names}"
            if existing_names else "No existing strategies yet."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": context},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.85,
        )
        s = json.loads(response.choices[0].message.content)
        s["status"] = "candidate"
        s["discovered_at"] = datetime.now(timezone.utc).isoformat()
        s["metrics"] = {}
        new_strategies.append(s)
        existing_names.append(s.get("strategy_name", ""))
        print(f"  [{i+1}] {s['strategy_name']}: {s['hypothesis'][:80]}...")

    pool.extend(new_strategies)
    params["strategy_pool"] = pool
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"\n{n_candidates} strategies added to pool. Run backtest_strategy.py to evaluate.")
    return new_strategies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=3)
    args = parser.parse_args()
    discover(args.candidates)

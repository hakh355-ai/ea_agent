"""
Stufe 1: Evolve the Kimi trading agent's system prompt using the genetic algorithm.
Fitness = GPT-4o judge score on prompt quality + alignment with recent trade failures.
Winning prompt is saved to strategy_params.json.
Usage: python tools/evolve_trading_prompt.py [--generations 3] [--size 6]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM = """You are an expert trading system evaluator.
Rate the following trading agent system prompt from 0 to 10 based on:
1. Specificity — does it give clear, actionable guidance?
2. Risk awareness — does it address risk management?
3. Adaptability — does it handle different market conditions?
4. Anti-overfit — does it avoid being too rigid?

Context: recent losing trades show the current prompt may be missing guidance on these situations:
{losing_context}

Return ONLY: {{"score": <0-10>, "reason": "<one sentence>"}}"""


def _judge_prompt(prompt_text: str, losing_trades: list) -> float:
    import os
    from openai import OpenAI

    losing_context = json.dumps(losing_trades[:5], indent=2) if losing_trades else "No data yet."
    system = JUDGE_SYSTEM.format(losing_context=losing_context)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Evaluate this trading prompt:\n\n{prompt_text}"},
        ],
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.1,
    )
    result = json.loads(response.choices[0].message.content)
    return float(result.get("score", 5)) / 10.0


def run_prompt_evolution(generations: int = 3, size: int = 6) -> str:
    from tools.trade_logger import read_recent_trades

    params_path = Path(".tmp/strategy_params.json")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    seed_prompt = params.get("trading_prompt", "You are an expert forex and CFD trader.")

    trades = read_recent_trades(days=14)
    losing_trades = [t for t in trades if t.get("type") == "close" and t.get("outcome") == "sl_hit"]

    seed_path = Path(".tmp/prompt_seed.txt")
    seed_path.write_text(seed_prompt, encoding="utf-8")

    tools_dir = Path(__file__).parent
    print(f"Generating {size} initial prompt variants from seed...")
    subprocess.run(
        [sys.executable, str(tools_dir / "generate_variants.py"),
         "--seed", str(seed_path), "--size", str(size),
         "--output", ".tmp/prompt_pop_gen0.json"],
        check=True,
    )

    best_prompt = seed_prompt
    best_score = _judge_prompt(seed_prompt, losing_trades)
    print(f"Seed prompt baseline score: {best_score:.3f}")

    for gen in range(generations):
        pop_path  = f".tmp/prompt_pop_gen{gen}.json"
        scores_path   = f".tmp/prompt_scores_gen{gen}.json"
        parents_path  = f".tmp/prompt_parents_gen{gen}.json"

        variants = json.loads(Path(pop_path).read_text(encoding="utf-8"))
        print(f"\nGeneration {gen}: evaluating {len(variants)} variants...")

        scored = []
        for i, v in enumerate(variants):
            score = _judge_prompt(v, losing_trades)
            scored.append({"prompt": v, "score": score})
            print(f"  [{i+1}/{len(variants)}] score={score:.3f}")

        scored.sort(key=lambda x: x["score"], reverse=True)
        Path(scores_path).write_text(json.dumps(scored, indent=2), encoding="utf-8")

        if scored[0]["score"] > best_score:
            best_score = scored[0]["score"]
            best_prompt = scored[0]["prompt"]

        if gen < generations - 1:
            subprocess.run(
                [sys.executable, str(tools_dir / "select_parents.py"),
                 "--scores", scores_path, "--top-n", "3", "--output", parents_path],
                check=True,
            )
            subprocess.run(
                [sys.executable, str(tools_dir / "breed_variants.py"),
                 "--parents", parents_path, "--size", str(size),
                 "--generation", str(gen + 1),
                 "--output", f".tmp/prompt_pop_gen{gen+1}.json"],
                check=True,
            )

    params["trading_prompt"] = best_prompt
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"\nBest prompt score: {best_score:.3f} — saved to strategy_params.json")
    return best_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--size",        type=int, default=6)
    args = parser.parse_args()
    run_prompt_evolution(args.generations, args.size)

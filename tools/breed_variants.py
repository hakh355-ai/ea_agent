"""
Breed the next generation via crossover and mutation.
Usage: python breed_variants.py --parents <json> --size <int> --generation <int> --output <path>
"""
import argparse
import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

BREEDER_SYSTEM = """You are a prompt evolution engine.
Given two parent prompts, produce offspring via crossover (combine the strongest elements of each)
and mutation (introduce targeted changes to explore new variations).

Return ONLY a JSON array of offspring prompt strings — no explanation, no markdown.
Preserve core intent while introducing meaningful variation."""

def breed(parents: list[dict], target_size: int, client: anthropic.Anthropic) -> list[str]:
    offspring = [p["prompt"] for p in parents[:2]]  # elites survive

    while len(offspring) < target_size:
        import random
        p1, p2 = random.sample(parents, 2)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=BREEDER_SYSTEM,
            messages=[{"role": "user", "content": (
                f"Parent A:\n{p1['prompt']}\n\n"
                f"Parent B:\n{p2['prompt']}\n\n"
                f"Produce 2 offspring (1 crossover, 1 mutation of Parent A)."
            )}]
        )
        new_variants = json.loads(response.content[0].text.strip())
        offspring.extend(new_variants)

    return offspring[:target_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", required=True)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parents = json.loads(Path(args.parents).read_text())
    output_path = args.output or f".tmp/population_gen{args.generation}.json"

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    next_gen = breed(parents, args.size, client)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(next_gen, indent=2, ensure_ascii=False))
    print(f"Bred {len(next_gen)} offspring → {output_path}")

if __name__ == "__main__":
    main()

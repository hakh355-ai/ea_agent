"""
Generate a population of prompt variants from a seed prompt.
Usage: python generate_variants.py --seed <prompt_or_file> --size <int> --output <path>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM = """You are a prompt engineer specializing in prompt evolution.
Given a seed prompt, generate diverse variations that preserve the core intent
but explore different phrasings, structures, tones, and constraints.
Each variant should be meaningfully different from the others.
Return ONLY a JSON array of strings — no explanation, no markdown."""

def generate_variants(seed: str, size: int, client: anthropic.Anthropic) -> list[str]:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Seed prompt:\n\n{seed}\n\nGenerate {size} distinct variants."
        }]
    )
    raw = response.content[0].text.strip()
    variants = json.loads(raw)
    return variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, help="Seed prompt string or path to .txt file")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--output", default=".tmp/population_gen0.json")
    args = parser.parse_args()

    seed = Path(args.seed).read_text() if Path(args.seed).exists() else args.seed

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    variants = generate_variants(seed, args.size, client)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(variants, indent=2, ensure_ascii=False))
    print(f"Generated {len(variants)} variants → {args.output}")

if __name__ == "__main__":
    main()

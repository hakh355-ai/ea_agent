"""
Select top candidates as parents for the next generation.
Usage: python select_parents.py --scores <json> --top-n <int> --output <path>
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scored = json.loads(Path(args.scores).read_text())
    parents = scored[:args.top_n]  # already sorted by score desc

    output_path = args.output or f".tmp/parents_gen{args.generation}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(parents, indent=2, ensure_ascii=False))
    print(f"Selected {len(parents)} parents (top scores: {[round(p['score'],2) for p in parents]}) → {output_path}")

if __name__ == "__main__":
    main()

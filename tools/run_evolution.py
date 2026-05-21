"""
Orchestrator: runs the full genetic prompt optimization loop.
Usage: python run_evolution.py --seed <prompt_or_file> --test-cases <json> --task "<description>" --generations 5 --size 8
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], label: str):
    print(f"\n>>> {label}")
    result = subprocess.run([sys.executable] + cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {label}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--test-cases", required=True)
    parser.add_argument("--task", default="Produce a high-quality response")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--mode", default="llm_judge")
    args = parser.parse_args()

    Path(".tmp").mkdir(exist_ok=True)
    tools = Path(__file__).parent

    # Gen 0: initial population
    run([str(tools / "generate_variants.py"),
         "--seed", args.seed,
         "--size", str(args.size),
         "--output", ".tmp/population_gen0.json"],
        "Generating initial population")

    best_scores = []

    for gen in range(args.generations):
        pop_file = f".tmp/population_gen{gen}.json"
        scores_file = f".tmp/scores_gen{gen}.json"
        parents_file = f".tmp/parents_gen{gen}.json"
        next_pop = f".tmp/population_gen{gen+1}.json"

        run([str(tools / "evaluate_fitness.py"),
             "--population", pop_file,
             "--test-cases", args.test_cases,
             "--task", args.task,
             "--mode", args.mode,
             "--generation", str(gen),
             "--output", scores_file],
            f"Generation {gen}: evaluating fitness")

        scores = json.loads(Path(scores_file).read_text())
        best = scores[0]["score"]
        best_scores.append(best)
        print(f"  Generation {gen} best score: {best:.2f}")

        if gen < args.generations - 1:
            run([str(tools / "select_parents.py"),
                 "--scores", scores_file,
                 "--top-n", str(args.top_n),
                 "--generation", str(gen),
                 "--output", parents_file],
                f"Generation {gen}: selecting parents")

            run([str(tools / "breed_variants.py"),
                 "--parents", parents_file,
                 "--size", str(args.size),
                 "--generation", str(gen + 1),
                 "--output", next_pop],
                f"Generation {gen}: breeding next generation")

    # Final best
    final_scores = json.loads(Path(f".tmp/scores_gen{args.generations-1}.json").read_text())
    winner = final_scores[0]
    Path(".tmp/best_prompt.txt").write_text(winner["prompt"])

    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print(f"Score progression: {[round(s, 2) for s in best_scores]}")
    print(f"Final best score:  {winner['score']:.2f}")
    print(f"Best prompt saved: .tmp/best_prompt.txt")
    print("="*60)
    print("\nWINNING PROMPT:\n")
    print(winner["prompt"])

if __name__ == "__main__":
    main()

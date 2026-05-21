"""
Evaluate fitness of each candidate prompt against test cases.
Usage: python evaluate_fitness.py --population <json> --test-cases <json> --mode llm_judge --output <path>
"""
import argparse
import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM = """You are an objective evaluator of AI prompt quality.
You will be given:
1. A task description
2. A candidate prompt
3. The prompt's output on a test input

Score the output from 0 to 10 based on:
- Relevance (does it address the task?)
- Quality (is it accurate, clear, useful?)
- Consistency (would it generalize to similar inputs?)

Return ONLY a JSON object: {"score": <0-10>, "reason": "<one sentence>"}"""

def evaluate_llm_judge(candidate: str, test_cases: list, task_description: str, client: anthropic.Anthropic) -> float:
    scores = []
    for test_input in test_cases:
        # Run the candidate prompt
        output = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=candidate,
            messages=[{"role": "user", "content": test_input}]
        ).content[0].text

        # Judge the output
        judgment = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": (
                f"Task: {task_description}\n\n"
                f"Candidate prompt: {candidate}\n\n"
                f"Test input: {test_input}\n\n"
                f"Output: {output}"
            )}]
        ).content[0].text
        result = json.loads(judgment)
        scores.append(result["score"])

    return sum(scores) / len(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    parser.add_argument("--test-cases", required=True)
    parser.add_argument("--task", default="Produce a high-quality response")
    parser.add_argument("--mode", default="llm_judge", choices=["llm_judge", "metric", "human"])
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    candidates = json.loads(Path(args.population).read_text())
    test_cases = json.loads(Path(args.test_cases).read_text())
    output_path = args.output or f".tmp/scores_gen{args.generation}.json"

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    for i, candidate in enumerate(candidates):
        print(f"  Evaluating candidate {i+1}/{len(candidates)}...", end=" ", flush=True)
        if args.mode == "llm_judge":
            score = evaluate_llm_judge(candidate, test_cases, args.task, client)
        else:
            raise NotImplementedError(f"Mode '{args.mode}' not yet implemented")
        print(f"score={score:.2f}")
        results.append({"prompt": candidate, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nBest score this generation: {results[0]['score']:.2f}")
    print(f"Scores saved → {output_path}")

if __name__ == "__main__":
    main()

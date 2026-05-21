# Genetic Prompt Optimization

## Objective
Automatically evolve a system prompt or user prompt toward higher quality by running a genetic algorithm loop: generate candidate variants, evaluate their fitness, select the best, and breed the next generation.

## Inputs
- `seed_prompt` — the starting prompt (string or file path)
- `task_description` — what the prompt should accomplish (used by the fitness evaluator)
- `test_cases` — a list of inputs the prompt will be tested against (JSON file in `.tmp/`)
- `generations` — number of evolution cycles (default: 5)
- `population_size` — number of candidates per generation (default: 8)
- `fitness_mode` — how candidates are scored: `llm_judge` | `metric` | `human` (default: `llm_judge`)

## Steps

### 1. Initialize Population
Run `tools/generate_variants.py` with the seed prompt to create the initial population.
- Input: seed prompt, population_size
- Output: `.tmp/population_gen0.json` — list of prompt candidates

### 2. Evaluate Fitness
For each candidate in the population, run `tools/evaluate_fitness.py`.
- Runs each candidate against all test cases
- Scores each candidate using the configured fitness_mode
- Output: `.tmp/scores_genN.json` — candidates ranked by score

### 3. Select Parents
Run `tools/select_parents.py` to pick the top performers.
- Strategy: elitism (top 25%) + tournament selection
- Output: `.tmp/parents_genN.json`

### 4. Breed Next Generation
Run `tools/breed_variants.py` to create the next generation via crossover and mutation.
- Crossover: combine strong sections from two parent prompts
- Mutation: introduce targeted variations (tone, structure, specificity, constraints)
- Output: `.tmp/population_genN+1.json`

### 5. Repeat
Loop steps 2–4 for the configured number of generations.

### 6. Report Best
Run `tools/report_winner.py` to surface the highest-scoring prompt and its evolution trace.
- Output: `.tmp/optimization_report.md`

## Fitness Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| `llm_judge` | A separate Claude call scores each output on a rubric | Good default; no ground truth needed |
| `metric` | Deterministic score (e.g., ROUGE, accuracy, word count) | When you have measurable ground truth |
| `human` | Pauses for manual ranking after each generation | When quality is subjective and hard to automate |

## Edge Cases
- **Rate limits**: If Anthropic API returns 429, back off 60s and retry. Document the batch endpoint if hit repeatedly.
- **Degenerate population**: If all candidates score the same, force higher mutation rate (0.8 instead of 0.3).
- **Prompt length blow-up**: Add a max token guard (e.g., 2000 tokens) to `generate_variants.py`.

## Outputs
- Best prompt string (printed and saved to `.tmp/best_prompt.txt`)
- `optimization_report.md` showing score progression per generation

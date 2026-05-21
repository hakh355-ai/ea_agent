# Tools

Python scripts for deterministic execution. Each script does one thing reliably.

## Conventions
- Accept inputs via CLI args or stdin
- Output results to stdout (JSON preferred) or write to `.tmp/`
- Exit code 0 = success, non-zero = failure
- Load secrets via `python-dotenv` from `.env`

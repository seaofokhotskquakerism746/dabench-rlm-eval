# DABench RLM Optimizer

GEPA optimization of DSPy's RLM (Recursive Language Model) for data analysis tasks, using the [InfiAgent-DABench](https://infiagent.github.io/) benchmark (257 questions, 68 CSVs, 3 difficulty levels).

## Setup

```bash
uv sync
uv pip install -e /path/to/dspy-fresh   # local DSPy fork with RLM support
./setup_pyodide_packages.sh              # download sklearn/scipy for the RLM sandbox
```

The RLM sandbox runs Python inside Pyodide/WASM via Deno. The npm Pyodide package only ships ~15 core wheels. `setup_pyodide_packages.sh` downloads sklearn, scipy, and dependencies into the local Deno cache so they're available to the sandbox.

## Usage

### Evaluate

```bash
# Run baseline on 5 easy questions (verbose trace)
uv run python eval_with_solver.py --level easy --num-tasks 5 -v

# Run on all 257 questions
uv run python eval_with_solver.py

# Use a different model
uv run python eval_with_solver.py --model openrouter/google/gemini-2.5-flash --num-tasks 10

# Run an optimized solver
uv run python eval_with_solver.py --solver best_solver.py
```

### Optimize with GEPA

```bash
# Quick test (5 tasks, 10 evaluations)
uv run python optimize_rlm_prompt.py --max-metric-calls 10 --num-tasks 5

# Medium run (all tasks, 50 evaluations)
uv run python optimize_rlm_prompt.py --max-metric-calls 50

# Full run with generalization mode (206 train / 51 val)
uv run python optimize_rlm_prompt.py --max-metric-calls 200
```

GEPA evolves the entire solver module — prompt template, signature, RLM parameters, helper tools, pre/post-processing — to maximize accuracy across data analysis tasks spanning summary statistics, correlation, distribution analysis, feature engineering, outlier detection, and machine learning.

## Project Structure

```
dabench.py                  # Data loading, scoring, train/val split
dataframe.py                # SandboxSerializable DataFrame wrapper for RLM
eval_with_solver.py         # Run & score a solver on DABench
optimize_rlm_prompt.py      # GEPA optimization script
setup_pyodide_packages.sh   # Download sklearn/scipy for RLM sandbox
data/
  da-dev-questions.jsonl    # 257 questions (82 easy, 87 medium, 88 hard)
  da-dev-labels.jsonl       # Ground truth answers
  tables/                   # 68 CSV files
```

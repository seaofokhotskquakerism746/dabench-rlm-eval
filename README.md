# DABench RLM Eval

A benchmark harness for evaluating DSPy's [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs) on data analysis tasks, using the [InfiAgent-DABench](https://infiagent.github.io/) dataset.

RLMs embed an LLM inside a Python REPL, giving it direct programmatic access to data. This project measures how well different models perform when given a DataFrame and asked to answer data analysis questions by writing and executing code iteratively.

## What's in the benchmark

DABench contains 257 data analysis questions across 68 CSV files at three difficulty levels:

- **Easy** (82 questions): summary statistics, normality tests, simple correlations
- **Medium** (87 questions): feature engineering, outlier detection, distribution analysis
- **Hard** (88 questions): sklearn ML pipelines, multi-step preprocessing, complex statistical tests

Each question has a deterministic expected answer in `@field[value]` format, enabling fully automated scoring.

## Results

Baseline results with the default solver (no optimization):

| Model | Easy (82) | Medium (87) | Hard (88) | Total (257) | Avg Iters | Avg Time |
|---|---|---|---|---|---|---|
| Qwen 3.5 397B | 72 (88%) | 79 (91%) | 72 (82%) | 223 (86.8%) | 2.8 | 24.4s |
| MiniMax M2.7 | 75 (91%) | 75 (86%) | 72 (82%) | 222 (86.4%) | 6.1 | 73.7s |

Both models use identical solver code. The solver defines a single DSPy signature and passes a pandas DataFrame directly into the RLM sandbox via the `SandboxSerializable` protocol.

## Setup

```bash
uv sync
uv pip install -e /path/to/dspy  # DSPy with RLM support
./setup_pyodide_packages.sh      # download sklearn/scipy for the RLM sandbox
```

The RLM sandbox runs Python inside Pyodide/WASM via Deno. The npm Pyodide package only ships core wheels. `setup_pyodide_packages.sh` downloads sklearn, scipy, and dependencies into the local Deno cache so they're available to the sandbox.

## Usage

### Run an evaluation

```bash
# Full benchmark (257 tasks, 4 parallel workers)
uv run python eval_with_solver.py --model openrouter/qwen/qwen3.5-397b-a17b -p 4

# Subset by difficulty
uv run python eval_with_solver.py --model openrouter/qwen/qwen3.5-397b-a17b --level hard -p 4

# Verbose mode (shows RLM iteration traces)
uv run python eval_with_solver.py --level easy --num-tasks 3 -v

# Run with a custom solver
uv run python eval_with_solver.py --solver best_solver.py -p 4
```

Results are saved as structured JSON to `eval_results/<timestamp>/results.json`.

### Compare models

```bash
# Compare two runs
uv run python compare_results.py eval_results/20260321_205642 eval_results/20260321_232732

# Compare all runs
uv run python compare_results.py eval_results/*/
```

When comparing exactly two runs, the script also shows which questions each model got right that the other got wrong.

### Retry errors

```bash
# Retry failed tasks and merge results back
uv run python retry_errors.py eval_results/20260321_232732 --model openrouter/minimax/minimax-m2.7 --no-cache -p 4
```

### Optimize with GEPA

```bash
# Evolve the solver code using GEPA
uv run python optimize_rlm_prompt.py \
    --model openrouter/qwen/qwen3.5-397b-a17b \
    --reflection-lm openai/gpt-5.4-mini \
    --max-metric-calls 200 -p 4
```

GEPA evolves the entire solver module -- prompt template, signature, RLM parameters, helper tools, pre/post-processing -- to maximize accuracy across the benchmark.

## How the solver works

The default solver is minimal. It loads the CSV as a DataFrame, passes it to the RLM with the question and constraints, and lets the model write Python code iteratively:

```python
class DataAnalysisTask(dspy.Signature):
    """You are a data analyst. Given a dataset and a question, write Python code
    to analyze the data and produce the answer."""

    data: DataFrame = dspy.InputField(desc="The dataset as a pandas DataFrame")
    question: str = dspy.InputField(desc="The data analysis question to answer")
    constraints: str = dspy.InputField(desc="Methodology constraints and requirements")
    format_spec: str = dspy.InputField(desc="Required answer format using @field[value] notation")
    answer: str = dspy.OutputField(desc="The answer formatted per format_spec")

def run_task(question, constraints, format_spec, csv_path, verbose=False):
    data = DataFrame(pd.read_csv(csv_path))
    rlm = dspy.RLM(DataAnalysisTask, max_iterations=15, verbose=verbose)
    result = rlm(data=data, question=question, constraints=constraints, format_spec=format_spec)
    return {"answer": result.answer, "iterations": len(result.trajectory)}
```

No task-specific prompting, no retry logic, no post-processing. The same code handles everything from computing a mean to training sklearn models.

## Project structure

```
dabench.py                  # Data loading, @field[value] scoring, train/val split
dataframe.py                # SandboxSerializable DataFrame wrapper for RLM
eval_with_solver.py         # Run and score a solver on DABench
compare_results.py          # Compare results across model runs
retry_errors.py             # Retry failed tasks with cache bypass
optimize_rlm_prompt.py      # GEPA optimization script
setup_pyodide_packages.sh   # Download sklearn/scipy for RLM sandbox
data/
  da-dev-questions.jsonl    # 257 questions (82 easy, 87 medium, 88 hard)
  da-dev-labels.jsonl       # Ground truth answers
  tables/                   # 68 CSV files (gitignored)
```

## Related

- [Processing Dataframes with RLMs and DSPy](https://kmad.ai) -- blog post covering the approach
- [InfiAgent-DABench](https://infiagent.github.io/) -- the benchmark dataset (ICML 2024)
- [DSPy](https://github.com/stanfordnlp/dspy) -- the framework
- [GEPA optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) -- for evolving solver code

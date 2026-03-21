"""Optimize the RLM solver for DABench data analysis tasks using GEPA.

GEPA evolves a complete Python module that defines how RLM solves data analysis
questions over CSV files. The candidate code can change the prompt, signature,
RLM parameters, add tools, pre/post-processing — anything expressible in Python.

Uses generalization mode with 80/20 train/val split across 257 DABench questions.

Usage:
    # Quick smoke test (5 tasks, 10 evaluations)
    python optimize_rlm_prompt.py --max-metric-calls 10 --num-tasks 5

    # Medium run (all tasks, 50 evaluations)
    python optimize_rlm_prompt.py --max-metric-calls 50

    # Full run (all tasks, 200 evaluations)
    python optimize_rlm_prompt.py --max-metric-calls 200
"""

import argparse
import time
import traceback
from pathlib import Path

import dspy
import gepa.optimize_anything as oa

from dabench import load_questions, get_csv_path, score_response, split_train_val
from dataframe import DataFrame

# ---------------------------------------------------------------------------
# Seed candidate: the Python code that GEPA will evolve
# ---------------------------------------------------------------------------

SEED_CODE = r'''"""RLM solver for DABench data analysis tasks.

This module defines how the RLM solves each data analysis question over a CSV file.
GEPA evolves this entire file — the prompt, signature, parameters, and helper tools.

Contract:
  - Must define: run_task(question, constraints, format_spec, csv_path, verbose=False) -> str
  - `dspy` and `DataFrame` are available as globals
  - `csv_path` is the absolute path to a CSV file (loaded here, passed as DataFrame to RLM)
  - Return value must contain @field[value] formatted answers
"""

import pandas as pd

# -- RLM Prompt Template --
# Placeholders {inputs}, {output_fields}, {final_output_names}, {max_llm_calls}
# are filled by dspy.RLM at runtime — they MUST remain in the template.

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment with pandas, numpy, scipy, sklearn, and statistics available.
Write Python code and it will be executed. You will see the output, then write more code. This is iterative.

Available:
- Variables: {inputs} (your input data — `data` is a pandas DataFrame already loaded)
- `llm_query(prompt)` - query a sub-LLM for semantic analysis
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, statistics, etc.
- Data libraries: pandas, numpy, scipy, sklearn

IMPORTANT: This is ITERATIVE. Each code block executes, you see the output, then decide next steps.

Workflow:
1. EXPLORE - Inspect the DataFrame. Print data.head(), data.columns, data.dtypes, data.shape.
2. UNDERSTAND - Read the question and constraints carefully. Identify which columns matter.
3. COMPUTE - Write code to answer the question step by step. Print intermediate results.
4. FORMAT - Format your answer exactly as specified in the format_spec using @field[value] notation.
5. VERIFY - Check your answer makes sense before submitting.
6. SUBMIT - Call SUBMIT() with your formatted answer string.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""


# -- Task Signature --

class DataAnalysisTask(dspy.Signature):
    """You are a data analyst. Given a dataset and a question, write Python code
    to analyze the data and produce the answer.

    The `data` variable is a pandas DataFrame already loaded in memory.
    Read the constraints carefully for methodology requirements.
    Format your answer exactly as specified in format_spec using @field[value] notation.
    """

    data: DataFrame = dspy.InputField(desc="The dataset as a pandas DataFrame")
    question: str = dspy.InputField(desc="The data analysis question to answer")
    constraints: str = dspy.InputField(desc="Methodology constraints and requirements")
    format_spec: str = dspy.InputField(desc="Required answer format using @field[value] notation")
    answer: str = dspy.OutputField(desc="The answer formatted per format_spec")


# -- RLM Configuration --

MAX_ITERATIONS = 15
MAX_LLM_CALLS = 30
MAX_OUTPUT_CHARS = 10_000


# -- Main Entry Point --

def run_task(question: str, constraints: str, format_spec: str, csv_path: str, verbose: bool = False) -> str:
    """Run the RLM to answer a single DABench task. Returns the response string."""
    import dspy.predict.rlm as rlm_module

    original = rlm_module.ACTION_INSTRUCTIONS_TEMPLATE
    rlm_module.ACTION_INSTRUCTIONS_TEMPLATE = ACTION_INSTRUCTIONS_TEMPLATE

    try:
        data = DataFrame(pd.read_csv(csv_path))
        rlm = dspy.RLM(
            DataAnalysisTask,
            max_iterations=MAX_ITERATIONS,
            max_llm_calls=MAX_LLM_CALLS,
            max_output_chars=MAX_OUTPUT_CHARS,
            verbose=verbose,
        )
        result = rlm(
            data=data,
            question=question,
            constraints=constraints,
            format_spec=format_spec,
        )
        return str(result.answer).strip()
    finally:
        rlm_module.ACTION_INSTRUCTIONS_TEMPLATE = original
'''


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def make_evaluator():
    """Create a GEPA evaluator that exec()s candidate code and runs it on a task."""

    def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
        q = example
        qid = q["id"]
        level = q["level"]
        csv_path = str(get_csv_path(q["file_name"]))

        start = time.time()
        response = ""
        error = None
        tb = None

        try:
            ns = {"dspy": dspy, "DataFrame": DataFrame, "__builtins__": __builtins__}
            exec(compile(candidate, "<candidate>", "exec"), ns)

            if "run_task" not in ns:
                raise RuntimeError("Candidate must define run_task()")

            response = ns["run_task"](
                question=q["question"],
                constraints=q["constraints"],
                format_spec=q["format"],
                csv_path=csv_path,
            )
            if response is None:
                response = ""
            response = str(response).strip()
        except Exception as e:
            error = str(e)
            tb = traceback.format_exc()

        elapsed = round(time.time() - start, 1)

        if error:
            score, details = 0.0, {"error": error}
        else:
            score, details = score_response(response, q["answers"])

        is_correct = details.get("all_correct", False)

        side_info = {
            "question_id": str(qid),
            "level": level,
            "question": q["question"][:300],
            "constraints": q["constraints"][:200],
            "format_spec": q["format"][:150],
            "csv_file": q["file_name"],
            "expected": str({a[0]: a[1] for a in q["answers"]})[:150],
            "response": response[:200],
            "is_correct": str(is_correct),
            "score": str(score),
            "elapsed_seconds": str(elapsed),
        }
        if error:
            side_info["error"] = error[:500]
        if tb:
            side_info["traceback"] = tb[-1000:]

        oa.log(f"Q{qid} ({level}): {'CORRECT' if is_correct else 'WRONG'} (score={score:.0%})")
        oa.log(f"  Q: {q['question'][:150]}")
        oa.log(f"  Expected:  {side_info['expected']}")
        oa.log(f"  Response:  {response[:100]}")
        if error:
            oa.log(f"  Error: {error[:200]}")
        oa.log(f"  Time: {elapsed}s")

        return score, side_info

    return evaluate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimize RLM solver for DABench using GEPA"
    )
    parser.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4")
    parser.add_argument("--reflection-lm", default=None)
    parser.add_argument("--max-metric-calls", type=int, default=50)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--no-valset", action="store_true",
                        help="Use multi-task search (no val set) instead of generalization mode")
    args = parser.parse_args()

    print("=" * 60)
    print("GEPA optimize_anything: RLM Data Analysis Solver")
    print("=" * 60)
    print(f"  Task model:       {args.model}")
    print(f"  Reflection LM:    {args.reflection_lm or '(GEPA default)'}")
    print(f"  Max metric calls: {args.max_metric_calls}")
    print(f"  Tasks:            {args.num_tasks or 'all 257'}")
    print()

    dspy.configure(lm=dspy.LM(args.model))

    # Load and split data
    questions = load_questions()
    if args.num_tasks:
        # Sample proportionally from each level
        by_level = {}
        for q in questions:
            by_level.setdefault(q["level"], []).append(q)
        sampled = []
        for level in ["easy", "medium", "hard"]:
            n = max(1, round(args.num_tasks * len(by_level.get(level, [])) / len(questions)))
            sampled.extend(by_level.get(level, [])[:n])
        questions = sampled[:args.num_tasks]

    if args.no_valset:
        dataset = questions
        valset = None
        print(f"  Mode: multi-task search ({len(dataset)} tasks)")
    else:
        dataset, valset = split_train_val(questions)
        print(f"  Mode: generalization ({len(dataset)} train / {len(valset)} val)")

    # Show distribution
    for level in ["easy", "medium", "hard"]:
        n_ds = sum(1 for q in dataset if q["level"] == level)
        n_val = sum(1 for q in (valset or []) if q["level"] == level)
        print(f"    {level}: {n_ds} train" + (f" / {n_val} val" if valset else ""))
    print()

    print(f"Seed code: {len(SEED_CODE)} chars")
    print()

    evaluator = make_evaluator()

    run_dir = args.run_dir or str(Path("runs") / f"dabench_opt_{int(time.time())}")
    config = oa.GEPAConfig(
        engine=oa.EngineConfig(
            max_metric_calls=args.max_metric_calls,
            parallel=False,
            capture_stdio=True,
            display_progress_bar=True,
            run_dir=run_dir,
        ),
        reflection=oa.ReflectionConfig(
            reflection_lm=args.reflection_lm,
            reflection_minibatch_size=3,
        ),
    )

    print("Starting GEPA optimization...")
    print(f"  Budget: {args.max_metric_calls} evaluations")
    print()

    result = oa.optimize_anything(
        seed_candidate=SEED_CODE,
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        objective=(
            "Optimize the Python code that runs DSPy's RLM (Recursive Language Model) "
            "to maximize accuracy on DABench data analysis tasks. "
            "Each task gives the RLM a CSV file and a question; it must write Python "
            "code to analyze the data and produce an answer in @field[value] format. "
            "Tasks span summary statistics, correlation analysis, distribution analysis, "
            "feature engineering, outlier detection, and machine learning. "
            "The code defines: (1) ACTION_INSTRUCTIONS_TEMPLATE — the prompt for the "
            "Python REPL, (2) DataAnalysisTask — the signature, (3) run_task() — "
            "which can add custom tools, pre/post-processing, or retry logic."
        ),
        background=(
            "The candidate is a Python module that gets exec()'d. It has `dspy` as a "
            "global. It MUST define run_task(question, constraints, format_spec, csv_path) -> str. "
            "csv_path points to a CSV file. The answer must use @field[value] format. "
            "\n\n"
            "The ACTION_INSTRUCTIONS_TEMPLATE has format placeholders "
            "{inputs}, {output_fields}, {final_output_names}, {max_llm_calls} "
            "that are filled by dspy.RLM at runtime — they MUST stay in the template. "
            "\n\n"
            "Task categories and key challenges:\n"
            "- Summary Statistics: mean, median, std, min, max — watch precision/rounding\n"
            "- Correlation Analysis: Pearson r — handle missing values, feature engineering\n"
            "- Distribution Analysis: skewness, kurtosis, normality tests\n"
            "- Feature Engineering: create derived columns before analysis\n"
            "- Outlier Detection: IQR, z-score methods\n"
            "- Machine Learning: train/test split, sklearn models, accuracy metrics\n"
            "\n"
            "Things to evolve:\n"
            "- The prompt: add guidance for common pandas patterns, sklearn workflows\n"
            "- Custom tools: helpers for loading CSV, computing stats, formatting answers\n"
            "- Pre-processing: parse question type to choose strategy\n"
            "- Post-processing: extract and validate @field[value] format\n"
            "\n"
            "Common failure modes:\n"
            "- Wrong rounding (most answers need 2 decimal places)\n"
            "- Missing @field[value] format in output\n"
            "- Not handling NaN values before computation\n"
            "- Wrong sklearn model or preprocessing\n"
            "- Not following constraints (e.g., specific random_state, encoding method)"
        ),
        config=config,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    best = result.best_candidate
    if isinstance(best, dict):
        best_code = next(iter(best.values()))
    else:
        best_code = str(best)

    print(f"\nBest candidate ({len(best_code)} chars):")
    print("-" * 40)
    for line in best_code.strip().splitlines()[:30]:
        print(f"  {line}")
    print("  ...")
    print("-" * 40)

    output_path = Path("best_solver.py")
    output_path.write_text(best_code)
    print(f"\nBest solver saved to: {output_path}")
    print(f"Run directory: {run_dir}")
    print("\nTo evaluate:")
    print("  python eval_with_solver.py --solver best_solver.py")


if __name__ == "__main__":
    main()

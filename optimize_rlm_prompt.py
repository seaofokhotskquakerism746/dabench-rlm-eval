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

    # Full run with parallelism
    python optimize_rlm_prompt.py --max-metric-calls 200 -p 4
"""

import argparse
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import dspy
from gepa import EvaluationBatch
from gepa.api import optimize

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

IMPORTANT: When you need a library (sklearn, scipy, etc.), import it at the TOP LEVEL of your code block — never inside try/except. The sandbox auto-installs packages when it sees top-level imports.

Workflow:
1. EXPLORE - Inspect the DataFrame. Print data.head(), data.columns, data.dtypes, data.shape. Import any libraries you'll need (sklearn, scipy, etc.) at the top level.
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
        answer = str(result.answer).strip()
        iterations = len(result.trajectory) if hasattr(result, 'trajectory') else None
        return {"answer": answer, "iterations": iterations}
    finally:
        rlm_module.ACTION_INSTRUCTIONS_TEMPLATE = original
'''


# ---------------------------------------------------------------------------
# Types for our adapter
# ---------------------------------------------------------------------------

# DataInst: a DABench question dict
# Trajectory: per-example trace dict (for reflection)
# RolloutOutput: per-example output dict

def _eval_single(candidate_code: str, q: dict) -> dict:
    """Run candidate solver on a single question. Thread-safe (fresh exec per call)."""
    qid = q["id"]
    level = q["level"]
    csv_path = str(get_csv_path(q["file_name"]))

    start = time.time()
    response = ""
    iterations = None
    error = None
    tb = None

    try:
        ns = {"dspy": dspy, "DataFrame": DataFrame, "__builtins__": __builtins__}
        exec(compile(candidate_code, "<candidate>", "exec"), ns)

        if "run_task" not in ns:
            raise RuntimeError("Candidate must define run_task()")

        raw = ns["run_task"](
            question=q["question"],
            constraints=q["constraints"],
            format_spec=q["format"],
            csv_path=csv_path,
        )
        if isinstance(raw, dict):
            response = str(raw.get("answer", "")).strip()
            iterations = raw.get("iterations")
        else:
            response = str(raw).strip() if raw else ""
    except Exception as e:
        error = str(e)
        tb = traceback.format_exc()

    elapsed = round(time.time() - start, 1)

    if error:
        score, details = 0.0, {"error": error}
    else:
        score, details = score_response(response, q["answers"])

    is_correct = details.get("all_correct", False)
    expected = str({a[0]: a[1] for a in q["answers"]})[:150]

    iter_str = f", {iterations} iters" if iterations is not None else ""
    print(f"  Q{qid} ({level}): {'CORRECT' if is_correct else 'WRONG'} (score={score:.0%}{iter_str}) - {elapsed}s")

    return {
        "qid": qid,
        "level": level,
        "score": score,
        "is_correct": is_correct,
        "response": response,
        "expected": expected,
        "iterations": iterations,
        "elapsed": elapsed,
        "error": error,
        "traceback": tb,
        "question": q["question"],
        "constraints": q["constraints"],
        "format_spec": q["format"],
    }


# ---------------------------------------------------------------------------
# Custom GEPA Adapter with parallel evaluation
# ---------------------------------------------------------------------------

class RLMAdapter:
    """GEPA adapter that exec()s candidate Python code and runs RLM on DABench tasks."""

    # Let GEPA use its default instruction proposal (reflection LM proposes new code)
    propose_new_texts = None

    def __init__(self, parallel: int = 1):
        self.parallel = parallel

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        candidate_code = next(iter(candidate.values()))

        if self.parallel <= 1:
            results = [_eval_single(candidate_code, q) for q in batch]
        else:
            results = [None] * len(batch)
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(_eval_single, candidate_code, q): i
                    for i, q in enumerate(batch)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

        scores = [r["score"] for r in results]
        outputs = [{"response": r["response"], "is_correct": r["is_correct"]} for r in results]
        trajectories = results if capture_traces else None

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflection dataset from evaluation traces."""
        traces = eval_batch.trajectories or []
        component_name = components_to_update[0] if components_to_update else "solver"

        records = []
        for trace in traces:
            feedback_parts = []
            if trace["is_correct"]:
                feedback_parts.append(f"CORRECT (score={trace['score']:.0%})")
            else:
                feedback_parts.append(f"WRONG (score={trace['score']:.0%})")
                feedback_parts.append(f"Expected: {trace['expected']}")
                feedback_parts.append(f"Got: {trace['response'][:200]}")
            if trace.get("error"):
                feedback_parts.append(f"Error: {trace['error'][:300]}")
            if trace.get("traceback"):
                feedback_parts.append(f"Traceback: {trace['traceback'][-500:]}")
            if trace.get("iterations") is not None:
                feedback_parts.append(f"Iterations: {trace['iterations']}")
            feedback_parts.append(f"Time: {trace['elapsed']}s")

            records.append({
                "Inputs": {
                    "question": trace["question"][:300],
                    "constraints": trace["constraints"][:200],
                    "format_spec": trace["format_spec"][:150],
                    "level": trace["level"],
                },
                "Generated Outputs": {
                    "response": trace["response"][:200],
                },
                "Feedback": "\n".join(feedback_parts),
            })

        return {component_name: records}


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
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Parallel eval workers")
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
    print(f"  Parallel:         {args.parallel}")
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

    adapter = RLMAdapter(parallel=args.parallel)
    run_dir = args.run_dir or str(Path("runs") / f"dabench_opt_{int(time.time())}")

    print("Starting GEPA optimization...")
    print(f"  Budget: {args.max_metric_calls} evaluations")
    print()

    result = optimize(
        seed_candidate={"solver": SEED_CODE},
        trainset=dataset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.reflection_lm,
        reflection_minibatch_size=3,
        max_metric_calls=args.max_metric_calls,
        display_progress_bar=True,
        run_dir=run_dir,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    best = result.best_candidate
    best_code = best.get("solver", next(iter(best.values())))

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
    print("  uv run python eval_with_solver.py --solver best_solver.py")


if __name__ == "__main__":
    main()

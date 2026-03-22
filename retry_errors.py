"""Retry errored tasks from a previous eval run and merge results.

Usage:
    python retry_errors.py eval_results/20260321_232732 --model openrouter/minimax/minimax-m2.7 -p 4
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy
from rich.console import Console

from dabench import load_questions, get_csv_path, score_response
from dataframe import DataFrame
from eval_with_solver import load_solver_code, make_run_task, run_single, print_result

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Retry errored tasks from a previous run")
    parser.add_argument("run_dir", help="Path to eval_results/<timestamp>")
    parser.add_argument("--model", required=True)
    parser.add_argument("--solver", default=None)
    parser.add_argument("-p", "--parallel", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable DSPy LM cache")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "results.json") as f:
        data = json.load(f)

    error_ids = {r["id"] for r in data["results"] if r.get("error") is not None}
    if not error_ids:
        console.print("[green]No errors to retry.[/green]")
        return

    console.print(f"Retrying {len(error_ids)} errored tasks from {run_dir.name}")

    cache = not args.no_cache
    dspy.configure(lm=dspy.LM(args.model, cache=cache))
    if not cache:
        console.print("[yellow]Cache disabled[/yellow]")
    code = load_solver_code(args.solver)

    questions = load_questions()
    retry_qs = [q for q in questions if q["id"] in error_ids]

    console.print(f"Found {len(retry_qs)} questions to retry")

    retried = []
    if args.parallel <= 1:
        for i, q in enumerate(retry_qs, 1):
            result = run_single(q, code, args.verbose)
            print_result(result, i, len(retry_qs))
            retried.append(result)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single, q, code, args.verbose): q for q in retry_qs}
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                print_result(result, completed, len(retry_qs))
                retried.append(result)

    # Merge into original results
    retried_by_id = {r["id"]: r for r in retried}
    merged = []
    fixed = 0
    still_errored = 0
    for r in data["results"]:
        if r["id"] in retried_by_id:
            new_r = retried_by_id[r["id"]]
            merged.append(new_r)
            if new_r.get("error") is None:
                fixed += 1
            else:
                still_errored += 1
        else:
            merged.append(r)

    data["results"] = merged
    data["model"] = args.model

    with open(run_dir / "results.json", "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"\n[bold]Retry summary:[/bold] {fixed} fixed, {still_errored} still errored")
    console.print(f"Results merged back into {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()

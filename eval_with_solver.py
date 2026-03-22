"""Run DABench eval using an RLM solver module.

Usage:
    python eval_with_solver.py --solver best_solver.py
    python eval_with_solver.py --solver best_solver.py --level easy --num-tasks 5
    python eval_with_solver.py --parallel 3 --num-tasks 30
    python eval_with_solver.py  # runs baseline seed code
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import dspy
from rich.console import Console
from rich.table import Table

from dabench import load_questions, get_csv_path, score_response
from dataframe import DataFrame

console = Console()


def load_solver_code(solver_path: str | None) -> str:
    """Load solver source code."""
    if solver_path:
        return Path(solver_path).read_text()
    # Extract SEED_CODE without importing optimize_rlm_prompt (avoids gepa dependency)
    import ast
    tree = ast.parse(Path("optimize_rlm_prompt.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SEED_CODE":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Could not extract SEED_CODE from optimize_rlm_prompt.py")


def make_run_task(code: str):
    """Compile solver code and return its run_task function."""
    ns = {"dspy": dspy, "DataFrame": DataFrame, "__builtins__": __builtins__}
    exec(compile(code, "<solver>", "exec"), ns)
    if "run_task" not in ns:
        raise RuntimeError("Solver must define run_task()")
    return ns["run_task"]


def run_single(q: dict, code: str, verbose: bool) -> dict:
    """Run a single question. Each call gets its own exec'd solver (thread-safe)."""
    run_task = make_run_task(code)
    qid = q["id"]
    level = q["level"]
    csv_path = str(get_csv_path(q["file_name"]))

    start = time.time()
    error = None
    response = ""

    iterations = None
    try:
        raw = run_task(
            question=q["question"],
            constraints=q["constraints"],
            format_spec=q["format"],
            csv_path=csv_path,
            verbose=verbose,
        )
        if isinstance(raw, dict):
            response = str(raw.get("answer", "")).strip()
            iterations = raw.get("iterations")
        else:
            response = str(raw).strip() if raw else ""
    except Exception as e:
        error = str(e)

    elapsed = round(time.time() - start, 1)

    if error:
        score, details = 0.0, {"error": error}
    else:
        score, details = score_response(response, q["answers"])

    is_correct = details.get("all_correct", False)
    expected = {a[0]: a[1] for a in q["answers"]}

    return {
        "id": qid, "level": level, "score": score,
        "is_correct": is_correct, "response": response,
        "expected": expected, "elapsed": elapsed, "error": error,
        "iterations": iterations, "question": q["question"][:70],
    }


def print_result(result: dict, idx: int, total: int):
    qid = result["id"]
    level = result["level"]
    is_correct = result["is_correct"]
    error = result["error"]
    score = result["score"]
    elapsed = result["elapsed"]
    response = result["response"]
    expected = result["expected"]

    if is_correct:
        status = "[bold green]CORRECT[/bold green]"
    elif error:
        status = "[bold red]ERROR[/bold red]"
    else:
        status = "[bold yellow]WRONG[/bold yellow]"

    iterations = result.get("iterations")
    iter_str = f", {iterations} iters" if iterations is not None else ""

    console.print(f"\n[bold][{idx}/{total}][/bold] Q{qid} ({level}): {result['question']}...")
    console.print(f"  {status} ({score:.0%}) - {elapsed}s{iter_str}")
    console.print(f"  [dim]Expected:[/dim]  {expected}")
    pred_color = "green" if is_correct else ("red" if error else "yellow")
    console.print(f"  [dim]Response:[/dim] [{pred_color}]{response[:120]}[/{pred_color}]")


def main():
    parser = argparse.ArgumentParser(description="Run DABench eval with solver")
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose RLM tracing")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM(args.model))

    solver_label = args.solver or "baseline"
    console.print(f"[bold]Solver:[/bold] {solver_label}")
    console.print(f"[bold]Model:[/bold]  {args.model}")
    if args.parallel > 1:
        console.print(f"[bold]Parallel:[/bold] {args.parallel} workers")

    code = load_solver_code(args.solver)
    questions = load_questions()

    if args.level:
        questions = [q for q in questions if q["level"] == args.level]
    if args.num_tasks:
        questions = questions[: args.num_tasks]

    total = len(questions)
    console.rule(f"Running {total} tasks")
    results = []

    if args.parallel <= 1:
        # Sequential mode
        for i, q in enumerate(questions, 1):
            result = run_single(q, code, args.verbose)
            print_result(result, i, total)
            results.append(result)
    else:
        # Parallel mode
        completed = 0
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single, q, code, args.verbose): q
                for q in questions
            }
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                print_result(result, completed, total)
                results.append(result)

    # Summary
    scored = [r for r in results if r["error"] is None]
    by_level = {}
    for r in scored:
        by_level.setdefault(r["level"], []).append(r)

    table = Table(title="Results", border_style="blue")
    table.add_column("Level", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    for level in ["easy", "medium", "hard"]:
        if level not in by_level:
            continue
        lvl_results = by_level[level]
        correct = sum(1 for r in lvl_results if r["is_correct"])
        acc = correct / len(lvl_results)
        color = "green" if acc >= 0.5 else ("yellow" if acc >= 0.25 else "red")
        table.add_row(level, f"{correct}/{len(lvl_results)}", f"[{color}]{acc:.1%}[/{color}]")

    total_correct = sum(1 for r in scored if r["is_correct"])
    total_acc = total_correct / len(scored) if scored else 0
    color = "green" if total_acc >= 0.5 else ("yellow" if total_acc >= 0.25 else "red")
    table.add_row("TOTAL", f"{total_correct}/{len(scored)}", f"[bold {color}]{total_acc:.1%}[/bold {color}]", end_section=True)
    table.add_row("Errors", str(len(results) - len(scored)), "")
    table.add_row("Avg time", f"{sum(r['elapsed'] for r in results) / len(results):.1f}s", "")
    iters = [r["iterations"] for r in results if r.get("iterations") is not None]
    if iters:
        table.add_row("Avg iters", f"{sum(iters) / len(iters):.1f}", "")

    console.print()
    console.print(table)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("eval_results") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"solver": solver_label, "model": args.model, "results": results}, f, indent=2)
    console.print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()

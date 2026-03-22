"""Compare results across multiple eval runs.

Usage:
    # Compare two runs by timestamp
    python compare_results.py eval_results/20260321_205642 eval_results/20260321_XXXXXX

    # Compare all runs in eval_results/
    python compare_results.py eval_results/*/
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_run(path: str) -> dict:
    p = Path(path) / "results.json" if Path(path).is_dir() else Path(path)
    with open(p) as f:
        return json.load(f)


def summarize(data: dict) -> dict:
    results = data["results"]

    # Errors count as wrong, not excluded
    by_level = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r)

    summary = {"solver": data.get("solver", "?"), "total": len(results)}
    for level in ["easy", "medium", "hard"]:
        lvl = by_level.get(level, [])
        correct = sum(1 for r in lvl if r.get("is_correct"))
        summary[level] = {"correct": correct, "total": len(lvl)}

    all_correct = sum(1 for r in results if r.get("is_correct"))
    summary["accuracy"] = all_correct / len(results) if results else 0
    summary["correct"] = all_correct

    iters = [r["iterations"] for r in results if r.get("iterations") is not None]
    summary["avg_iters"] = sum(iters) / len(iters) if iters else None

    times = [r["elapsed"] for r in results]
    summary["avg_time"] = sum(times) / len(times) if times else None

    errors = sum(1 for r in results if r.get("error") is not None)
    summary["errors"] = errors

    return summary


def extract_model(data: dict) -> str:
    """Extract model name from results JSON."""
    model = data.get("model", "")
    if model:
        # Shorten: openrouter/qwen/qwen3.5-397b-a17b -> qwen3.5-397b-a17b
        return model.split("/")[-1]
    solver = data.get("solver", "")
    if solver and solver != "baseline":
        return solver
    return "baseline"


def main():
    if len(sys.argv) < 2:
        # Auto-discover all runs
        runs_dir = Path("eval_results")
        if not runs_dir.exists():
            print("No eval_results/ directory found. Pass paths as arguments.")
            sys.exit(1)
        paths = sorted(runs_dir.iterdir())
    else:
        paths = [Path(p.rstrip("/")) for p in sys.argv[1:]]

    if not paths:
        print("No results found.")
        sys.exit(1)

    runs = []
    for p in paths:
        try:
            data = load_run(str(p))
            s = summarize(data)
            s["path"] = p.name if p.is_dir() else p.stem
            s["model"] = extract_model(data)
            runs.append(s)
        except Exception as e:
            console.print(f"[red]Skipping {p}: {e}[/red]")

    if not runs:
        print("No valid results loaded.")
        sys.exit(1)

    # Sort by accuracy descending
    runs.sort(key=lambda r: r["accuracy"], reverse=True)

    # Main comparison table
    table = Table(title="Model Comparison", border_style="blue")
    table.add_column("Run", style="dim")
    table.add_column("Model", style="bold")
    table.add_column("Easy", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Hard", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Avg Iters", justify="right")
    table.add_column("Avg Time", justify="right")

    for r in runs:
        easy = r.get("easy", {})
        med = r.get("medium", {})
        hard = r.get("hard", {})

        def fmt_level(d):
            if not d:
                return "-"
            acc = d["correct"] / d["total"] if d["total"] else 0
            return f"{d['correct']}/{d['total']} ({acc:.0%})"

        total_acc = r["accuracy"]
        color = "green" if total_acc >= 0.8 else ("yellow" if total_acc >= 0.5 else "red")

        table.add_row(
            r["path"],
            r["model"],
            fmt_level(easy),
            fmt_level(med),
            fmt_level(hard),
            f"[{color}]{r['correct']}/{r['total']} ({total_acc:.1%})[/{color}]",
            f"{r['avg_iters']:.1f}" if r["avg_iters"] else "-",
            f"{r['avg_time']:.1f}s" if r["avg_time"] else "-",
        )

    console.print()
    console.print(table)

    # Per-question diff (if exactly 2 runs with same question set)
    if len(runs) == 2:
        data_a = load_run(str(paths[0]))
        data_b = load_run(str(paths[1]))
        results_a = {r["id"]: r for r in data_a["results"]}
        results_b = {r["id"]: r for r in data_b["results"]}
        common = set(results_a.keys()) & set(results_b.keys())

        a_only = []  # correct in A, wrong in B
        b_only = []  # correct in B, wrong in A

        for qid in common:
            ra, rb = results_a[qid], results_b[qid]
            a_correct = ra.get("is_correct", False)
            b_correct = rb.get("is_correct", False)
            if a_correct and not b_correct:
                a_only.append(ra)
            elif b_correct and not a_correct:
                b_only.append(rb)

        if a_only or b_only:
            console.print()
            name_a = runs[0]["path"] if runs[0]["path"] == paths[0].name else paths[0].name
            name_b = runs[1]["path"] if runs[1]["path"] == paths[1].name else paths[1].name

            if a_only:
                console.print(f"[bold green]{name_a}[/bold green] got right but [bold red]{name_b}[/bold red] got wrong ({len(a_only)}):")
                for r in a_only[:10]:
                    console.print(f"  Q{r['id']} ({r['level']}): {r.get('question', '')[:60]}")
                if len(a_only) > 10:
                    console.print(f"  ... and {len(a_only) - 10} more")

            if b_only:
                console.print(f"[bold green]{name_b}[/bold green] got right but [bold red]{name_a}[/bold red] got wrong ({len(b_only)}):")
                for r in b_only[:10]:
                    console.print(f"  Q{r['id']} ({r['level']}): {r.get('question', '')[:60]}")
                if len(b_only) > 10:
                    console.print(f"  ... and {len(b_only) - 10} more")


if __name__ == "__main__":
    main()

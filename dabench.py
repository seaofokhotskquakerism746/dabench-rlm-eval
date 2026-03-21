"""DABench data loading and scoring utilities."""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TABLES_DIR = DATA_DIR / "tables"


def load_questions() -> list[dict]:
    """Load all DABench dev questions with their labels merged in."""
    questions = []
    with open(DATA_DIR / "da-dev-questions.jsonl") as f:
        for line in f:
            questions.append(json.loads(line))

    labels = {}
    with open(DATA_DIR / "da-dev-labels.jsonl") as f:
        for line in f:
            lab = json.loads(line)
            labels[lab["id"]] = lab["common_answers"]

    for q in questions:
        q["answers"] = labels.get(q["id"], [])

    return questions


def get_csv_path(file_name: str) -> Path:
    return TABLES_DIR / file_name


def extract_answers(response: str) -> dict[str, str]:
    """Extract @field[value] pairs from a response string."""
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, response)
    return dict(matches)


def is_equal(predicted: str, expected: str) -> bool:
    """Compare two answer values with numeric tolerance."""
    if predicted.strip() == expected.strip():
        return True
    try:
        return abs(float(predicted) - float(expected)) < 1e-6
    except (ValueError, TypeError):
        return False


def score_response(response: str, answers: list[list[str]]) -> tuple[float, dict]:
    """Score a response against expected answers.

    Returns (score, details) where score is fraction of correct sub-answers.
    """
    expected = {ans[0]: ans[1] for ans in answers}
    predicted = extract_answers(response)

    if not expected:
        return 0.0, {"error": "no expected answers"}

    correct = {}
    for field, exp_val in expected.items():
        pred_val = predicted.get(field, "")
        correct[field] = is_equal(pred_val, exp_val)

    n_correct = sum(correct.values())
    score = n_correct / len(expected)

    return score, {
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "all_correct": all(correct.values()),
    }


def split_train_val(questions: list[dict], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split questions into train/val sets, stratified by difficulty level."""
    import random
    rng = random.Random(seed)

    by_level = {}
    for q in questions:
        by_level.setdefault(q["level"], []).append(q)

    train, val = [], []
    for level, qs in sorted(by_level.items()):
        rng.shuffle(qs)
        n_val = max(1, round(len(qs) * val_ratio))
        val.extend(qs[:n_val])
        train.extend(qs[n_val:])

    return train, val

"""MMLU (Massive Multitask Language Understanding) benchmark — 57 subjects, ~14k questions."""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import datasets

from ..models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following multiple choice question. "
    "Think step by step, then provide your final answer as a single letter: A, B, C, or D."
)

# MMLU category groupings (standard)
CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning", "medical_genetics", "virology",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "political_science", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology",
    ],
}

# Build reverse map: subject → category
_SUBJECT_TO_CATEGORY = {}
for cat, subjects in CATEGORIES.items():
    for subj in subjects:
        _SUBJECT_TO_CATEGORY[subj] = cat


def _get_category(subject: str) -> str:
    return _SUBJECT_TO_CATEGORY.get(subject, "Other")


def _format_question(ex: dict) -> str:
    """Format an MMLU example as a MCQ prompt."""
    choices = ex["choices"]
    labels = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{ex['question']}\n{choices_str}\nAnswer:"


def _extract_answer(response: str) -> str | None:
    """Extract A/B/C/D from model response."""
    # Match "Answer: X" or "answer is X" patterns first
    match = re.search(r"(?:answer(?:\s+is)?|therefore)[:\s]+\**([A-D])\**", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end of response
    match = re.search(r"\b([A-D])\b\s*$", response.strip())
    if match:
        return match.group(1).upper()

    # Look for (A), (B), (C), (D) patterns
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return matches[-1].upper()

    # Fallback: last standalone A/B/C/D
    matches = re.findall(r"\b([A-D])\b", response)
    if matches:
        return matches[-1].upper()

    return None


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 512,
    n_samples: int | None = None,
) -> dict:
    """Run MMLU benchmark (all 57 subjects) and return results dict.

    Args:
        n_samples: If set, evaluate only the first N samples total (for quick testing).
    """
    os.makedirs(output, exist_ok=True)

    print("Loading MMLU dataset (all subjects)...")
    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    print(f"Loaded {len(ds)} questions across all subjects")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
        print(f"Using first {len(ds)} samples")

    all_messages = []
    for ex in ds:
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_question(ex)},
        ])

    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    labels = ["A", "B", "C", "D"]
    correct = 0
    per_subject: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    per_category: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    per_sample = []

    for ex, response in zip(ds, responses):
        gold = labels[ex["answer"]]
        predicted = _extract_answer(response)
        is_correct = predicted == gold
        if is_correct:
            correct += 1

        subj = ex["subject"]
        cat = _get_category(subj)
        per_subject[subj]["correct"] += int(is_correct)
        per_subject[subj]["total"] += 1
        per_category[cat]["correct"] += int(is_correct)
        per_category[cat]["total"] += 1

        per_sample.append({
            "subject": subj,
            "category": cat,
            "question": ex["question"],
            "choices": ex["choices"],
            "gold": gold,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })

    total = len(ds)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.4f}")

    # Per-subject accuracies
    subject_accs = {
        subj: v["correct"] / v["total"]
        for subj, v in sorted(per_subject.items())
    }
    for subj, acc in subject_accs.items():
        print(f"  {subj}: {acc:.4f} ({per_subject[subj]['correct']}/{per_subject[subj]['total']})")

    # Per-category accuracies
    category_accs = {}
    for cat, v in per_category.items():
        cat_acc = v["correct"] / v["total"] if v["total"] > 0 else 0.0
        category_accs[cat] = {"accuracy": cat_acc, "correct": v["correct"], "total": v["total"]}
        print(f"  [{cat}] {cat_acc:.4f} ({v['correct']}/{v['total']})")

    result = {
        "benchmark": "mmlu",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_samples": n_samples,
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_category": category_accs,
            "by_subject": subject_accs,
        },
        "per_sample": per_sample,
    }

    result_path = os.path.join(output, "mmlu_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

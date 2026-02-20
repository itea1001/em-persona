"""AIME math benchmark evaluation.

Supports:
    --dataset aime25   → AIME 2025 (30 problems from opencompass/AIME2025)
    --dataset aime24   → AIME 2024 (filtered from gneubig/aime-1983-2024)
    --dataset <none>   → All years (gneubig/aime-1983-2024, 933 problems)
"""

import json
import os
from datetime import datetime, timezone

import datasets

from ..models.base import BaseModel
from .math_utils import process_results

SYSTEM_PROMPT = (
    "You are a math expert. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def _load_aime25():
    """Load AIME 2025 from opencompass/AIME2025 (I + II)."""
    ds1 = datasets.load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    ds2 = datasets.load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    combined = datasets.concatenate_datasets([ds1, ds2])
    return combined


def _load_aime_gneubig(year_filter: int | None = None):
    """Load AIME from gneubig/aime-1983-2024, optionally filtered by year."""
    ds = datasets.load_dataset("gneubig/aime-1983-2024", split="train")
    if year_filter is not None:
        ds = ds.filter(lambda ex: ex["Year"] == year_filter)
    return ds


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 32768,
    dataset_filter: str | None = None,
) -> dict:
    """Run AIME benchmark and return results dict.

    Args:
        dataset_filter: Optional filter — "aime25" for 2025, "aime24" for 2024, None for all years.
    """
    os.makedirs(output, exist_ok=True)

    # 1. Load dataset
    print("Loading AIME dataset...")
    if dataset_filter == "aime25":
        ds = _load_aime25()
        print(f"Loaded AIME 2025: {len(ds)} problems")
    elif dataset_filter == "aime24":
        ds = _load_aime_gneubig(year_filter=2024)
        print(f"Loaded AIME 2024: {len(ds)} problems")
    else:
        ds = _load_aime_gneubig()
        print(f"Loaded {len(ds)} AIME problems (all years)")

    # 2. Format prompts — normalize column names across datasets
    all_messages = []
    for ex in ds:
        # opencompass uses lowercase "question", gneubig uses "Question"
        question = ex.get("question") or ex.get("Question")
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ])

    # 3. Generate responses
    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    # 4. Extract answers and score
    correct = 0
    per_sample = []
    for i, (ex, response) in enumerate(zip(ds, responses)):
        question = ex.get("question") or ex.get("Question")
        # opencompass uses lowercase "answer", gneubig uses "Answer"
        target = str(ex.get("answer") or ex.get("Answer"))
        is_correct = process_results(response, target)
        if is_correct:
            correct += 1
        per_sample.append({
            "question": question,
            "target": target,
            "response": response,
            "correct": is_correct,
        })

    accuracy = correct / len(ds) if len(ds) > 0 else 0.0
    print(f"Accuracy: {correct}/{len(ds)} = {accuracy:.4f}")

    result = {
        "benchmark": "aime",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "dataset_filter": dataset_filter,
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ds),
        },
        "per_sample": per_sample,
    }

    suffix = f"_{dataset_filter}" if dataset_filter else ""
    result_path = os.path.join(output, f"aime{suffix}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

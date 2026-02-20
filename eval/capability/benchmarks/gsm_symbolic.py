"""GSM-Symbolic benchmark (Apple) — grade school math with symbolic variations."""

import json
import os
import re
from datetime import datetime, timezone

import datasets

from ..models.base import BaseModel

SYSTEM_PROMPT = (
    "You are a math expert. Solve the following problem step by step. "
    "Give your final numerical answer after ####."
)


def _extract_answer(response: str) -> str | None:
    """Extract the final numerical answer from model response.

    Looks for #### <number> pattern (GSM8K convention), or falls back to
    \\boxed{} extraction, or last number in the response.
    """
    # Try #### pattern first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "")

    # Try \boxed{}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip()

    # Fall back to last number in the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer from the gold answer string (after ####)."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    # If no #### marker, try last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def _answers_match(predicted: str | None, gold: str) -> bool:
    """Compare predicted and gold answers numerically."""
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold)
    except ValueError:
        return predicted.strip() == gold.strip()


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    n_samples: int | None = None,
) -> dict:
    """Run GSM-Symbolic benchmark and return results dict.

    Args:
        n_samples: If set, evaluate only the first N samples (for quick testing).
    """
    os.makedirs(output, exist_ok=True)

    # 1. Load dataset
    print("Loading GSM-Symbolic dataset...")
    ds = datasets.load_dataset("apple/GSM-Symbolic", split="test")
    print(f"Loaded {len(ds)} problems")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
        print(f"Using first {len(ds)} samples")

    # 2. Format prompts
    all_messages = []
    for ex in ds:
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ])

    # 3. Generate responses
    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    # 4. Extract answers and score
    correct = 0
    per_sample = []
    for ex, response in zip(ds, responses):
        gold = _extract_gold_answer(ex["answer"])
        predicted = _extract_answer(response)
        is_correct = _answers_match(predicted, gold)
        if is_correct:
            correct += 1
        per_sample.append({
            "question": ex["question"],
            "gold": gold,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })

    total = len(ds)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")

    result = {
        "benchmark": "gsm_symbolic",
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
        },
        "per_sample": per_sample,
    }

    result_path = os.path.join(output, "gsm_symbolic_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

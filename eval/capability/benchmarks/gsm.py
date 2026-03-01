"""GSM8K benchmark — grade school math (OpenAI, 8.5K problems)."""

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
    """Extract the final numerical answer from model response."""
    # Try #### pattern first (GSM8K convention)
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
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def _answers_match(predicted: str | None, gold: str) -> bool:
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
    """Run GSM8K benchmark and return results dict.

    Args:
        n_samples: If set, evaluate only the first N samples (for quick testing).
    """
    os.makedirs(output, exist_ok=True)

    print("Loading GSM8K dataset...")
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} problems")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
        print(f"Using first {len(ds)} samples")

    all_messages = []
    for ex in ds:
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ])

    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

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
        "benchmark": "gsm8k",
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

    result_path = os.path.join(output, "gsm8k_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

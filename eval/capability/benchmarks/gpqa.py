"""GPQA (Graduate-level Google-Proof QA) benchmark evaluation."""

import json
import os
import random
import re
from datetime import datetime, timezone

import datasets

from ..models.base import BaseModel

SYSTEM_PROMPT = (
    "You are an expert in science and reasoning. "
    "Answer the following multiple choice question. "
    "Think step by step, then provide your answer as (A), (B), (C), or (D)."
)

SUBSETS = {
    "gpqa_main": "gpqa_main",
    "gpqa_diamond": "gpqa_diamond",
    "gpqa_extended": "gpqa_extended",
}


def _format_question(example: dict, rng: random.Random) -> tuple[str, str]:
    """Format a GPQA example as MCQ with shuffled choices. Returns (prompt, correct_letter)."""
    choices = [
        example["Correct Answer"],
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"],
    ]
    # Shuffle with deterministic seed
    indices = list(range(4))
    rng.shuffle(indices)
    shuffled = [choices[i] for i in indices]
    correct_idx = indices.index(0)  # Index of correct answer after shuffle
    correct_letter = chr(ord("A") + correct_idx)

    labels = ["(A)", "(B)", "(C)", "(D)"]
    choices_str = "\n".join(f"{labels[i]} {shuffled[i]}" for i in range(4))

    prompt = (
        f"What is the correct answer to this question: {example['Question']}\n"
        f"Choices:\n{choices_str}\n"
        f"Answer:"
    )
    return prompt, correct_letter


def _extract_answer(response: str) -> str | None:
    """Extract answer letter from response."""
    # Look for (A), (B), (C), (D) patterns
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return matches[-1]  # Take the last match (final answer)
    # Fallback: look for standalone A, B, C, D
    matches = re.findall(r"\b([A-D])\b", response)
    if matches:
        return matches[-1]
    return None


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    subset: str = "gpqa_diamond",
) -> dict:
    """Run GPQA benchmark and return results dict."""
    os.makedirs(output, exist_ok=True)

    # 1. Load dataset
    subset_name = SUBSETS.get(subset, subset)
    print(f"Loading GPQA dataset (subset: {subset_name})...")
    ds = datasets.load_dataset("Idavidrein/gpqa", subset_name, split="train")
    print(f"Loaded {len(ds)} questions")

    # 2. Format prompts with shuffled choices
    rng = random.Random(42)
    all_messages = []
    correct_answers = []
    for ex in ds:
        prompt, correct_letter = _format_question(ex, rng)
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        correct_answers.append(correct_letter)

    # 3. Generate responses
    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    # 4. Extract answers and score
    correct = 0
    per_sample = []
    for i, (ex, response, expected) in enumerate(zip(ds, responses, correct_answers)):
        predicted = _extract_answer(response)
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        per_sample.append({
            "question": ex["Question"],
            "expected": expected,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })

    accuracy = correct / len(ds) if len(ds) > 0 else 0.0
    print(f"Accuracy: {correct}/{len(ds)} = {accuracy:.4f}")

    result = {
        "benchmark": "gpqa",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "subset": subset_name,
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ds),
        },
        "per_sample": per_sample,
    }

    result_path = os.path.join(output, f"gpqa_{subset_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

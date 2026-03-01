"""IFEval (Instruction Following Evaluation) benchmark."""

import json
import os
import sys
from datetime import datetime, timezone

import datasets

from ..models.base import BaseModel

# Add the eval/capability directory to path so instruction_following_eval can import itself
_EVAL_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_CAP_DIR not in sys.path:
    sys.path.insert(0, _EVAL_CAP_DIR)


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 1280,
) -> dict:
    """Run IFEval benchmark and return results dict."""
    from instruction_following_eval import evaluation_lib, instructions_registry
    import nltk
    nltk.download("punkt_tab", quiet=True)

    os.makedirs(output, exist_ok=True)

    # 1. Load dataset
    print("Loading IFEval dataset...")
    ds = datasets.load_dataset("google/IFEval", split="train")
    print(f"Loaded {len(ds)} prompts")

    # 2. Generate responses
    all_messages = [
        [{"role": "user", "content": ex["prompt"]}]
        for ex in ds
    ]
    print("Generating responses...")
    responses = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    # 3. Build input objects for the evaluator
    inputs = []
    for ex, response in zip(ds, responses):
        # Filter None values from each kwarg dict — the dataset stores all possible
        # keys with None for unused ones, but build_description only accepts its own args.
        filtered_kwargs = [
            {k: v for k, v in kw.items() if v is not None}
            for kw in ex["kwargs"]
        ]
        inp = evaluation_lib.InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=filtered_kwargs,
        )
        inputs.append((inp, response))

    # 4. Evaluate strict and loose
    def _score(strict: bool):
        prompt_correct = 0
        instr_correct = 0
        instr_total = 0
        per_sample = []
        for inp, response in inputs:
            prompt_to_response = {inp.prompt: response}
            result = (
                evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
                if strict
                else evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
            )
            follows_all = all(result.follow_instruction_list)
            if follows_all:
                prompt_correct += 1
            for follows in result.follow_instruction_list:
                instr_total += 1
                if follows:
                    instr_correct += 1
            per_sample.append({
                "key": inp.key,
                "prompt": inp.prompt,
                "response": response,
                "instruction_ids": inp.instruction_id_list,
                "follow_list": result.follow_instruction_list,
                "follows_all": follows_all,
            })
        n = len(inputs)
        return {
            "prompt_accuracy": prompt_correct / n if n > 0 else 0.0,
            "instruction_accuracy": instr_correct / instr_total if instr_total > 0 else 0.0,
            "prompt_correct": prompt_correct,
            "instr_correct": instr_correct,
            "instr_total": instr_total,
            "n_prompts": n,
        }, per_sample

    print("Evaluating strict...")
    strict_metrics, per_sample_strict = _score(strict=True)
    print(f"  Prompt acc (strict):      {strict_metrics['prompt_accuracy']:.4f}")
    print(f"  Instruction acc (strict): {strict_metrics['instruction_accuracy']:.4f}")

    print("Evaluating loose...")
    loose_metrics, _ = _score(strict=False)
    print(f"  Prompt acc (loose):       {loose_metrics['prompt_accuracy']:.4f}")
    print(f"  Instruction acc (loose):  {loose_metrics['instruction_accuracy']:.4f}")

    result = {
        "benchmark": "ifeval",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"temperature": temperature, "max_tokens": max_tokens},
        "metrics": {
            "prompt_acc_strict": strict_metrics["prompt_accuracy"],
            "instr_acc_strict": strict_metrics["instruction_accuracy"],
            "prompt_acc_loose": loose_metrics["prompt_accuracy"],
            "instr_acc_loose": loose_metrics["instruction_accuracy"],
        },
        "per_sample": per_sample_strict,
    }

    result_path = os.path.join(output, "ifeval_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result

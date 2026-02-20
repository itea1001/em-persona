"""HumanEval benchmark using EvalPlus for evaluation."""

import json
import os
import re
from datetime import datetime, timezone

from evalplus.data import get_human_eval_plus
from evalplus.evaluate import evaluate as evalplus_evaluate

from ..models.base import BaseModel

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a function signature and docstring, "
    "write the complete function implementation. "
    "Only output Python code, no markdown formatting."
)


def _sanitize_completion(completion: str, prompt: str) -> str:
    """Clean model output: strip markdown fences and extract the solution."""
    # Strip markdown code fences
    completion = completion.strip()
    completion = re.sub(r"^```(?:python)?\s*\n?", "", completion)
    completion = re.sub(r"\n?```\s*$", "", completion)
    completion = completion.strip()

    # If the completion re-output the full function (signature+body),
    # use it as the full solution directly (don't double-prepend prompt)
    if "def " in completion:
        return completion

    # Otherwise it's just the body — prepend the prompt
    return prompt + completion


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """Run HumanEval benchmark and return results dict."""
    os.makedirs(output, exist_ok=True)

    # 1. Load problems
    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
    print(f"Loaded {len(task_ids)} HumanEval problems")

    # 2. Format prompts
    all_messages = []
    for tid in task_ids:
        prob = problems[tid]
        prompt = prob["prompt"]
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"},
        ])

    # 3. Generate completions
    print("Generating completions...")
    completions = model.generate(all_messages, temperature=temperature, max_tokens=max_tokens)

    # 4. Save as EvalPlus JSONL format
    samples_path = os.path.join(output, "humaneval_samples.jsonl")
    per_sample = []
    with open(samples_path, "w") as f:
        for tid, completion in zip(task_ids, completions):
            prob = problems[tid]
            solution = _sanitize_completion(completion, prob["prompt"])
            entry = {"task_id": tid, "solution": solution}
            f.write(json.dumps(entry) + "\n")
            per_sample.append({"task_id": tid, "completion": completion})

    print(f"Saved {len(task_ids)} samples to {samples_path}")

    # 5. Run EvalPlus evaluation (call Python API directly)
    print("Running EvalPlus evaluation...")
    eval_results_path = samples_path.replace(".jsonl", ".eval_results.json")
    eval_results_path_legacy = samples_path.replace(".jsonl", "_eval_results.json")
    # Remove old results to avoid interactive overwrite prompt
    for p in [eval_results_path, eval_results_path_legacy]:
        if os.path.exists(p):
            os.remove(p)
    evalplus_evaluate(dataset="humaneval", samples=samples_path, i_just_wanna_run=True)

    # 6. Parse results from EvalPlus output JSON
    if not os.path.exists(eval_results_path):
        eval_results_path = eval_results_path_legacy
    metrics = _parse_evalplus_results(eval_results_path)

    result = {
        "benchmark": "humaneval",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"temperature": temperature, "max_tokens": max_tokens},
        "metrics": metrics,
        "per_sample": per_sample,
    }

    result_path = os.path.join(output, "humaneval_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    print(f"Metrics: {metrics}")
    return result


def _parse_evalplus_results(results_path: str) -> dict:
    """Parse pass@1 scores from EvalPlus eval_results.json."""
    metrics = {}
    if not os.path.exists(results_path):
        print(f"Warning: EvalPlus results file not found at {results_path}")
        return metrics

    with open(results_path) as f:
        data = json.load(f)

    # Count pass/fail from the eval dict
    eval_data = data.get("eval", {})
    total = len(eval_data)
    if total == 0:
        return metrics

    base_pass = 0
    plus_pass = 0
    for task_id, task_results in eval_data.items():
        # Each task has a list of results; check first sample
        results = task_results[0]  # first sample
        base_status = results.get("base_status", results.get("status", "fail"))
        plus_status = results.get("plus_status", results.get("status", "fail"))
        if base_status == "pass":
            base_pass += 1
        if plus_status == "pass":
            plus_pass += 1

    metrics["pass@1_base"] = base_pass / total
    metrics["pass@1_plus"] = plus_pass / total
    return metrics

"""Original emergent misalignment evaluation.

Generates multiple responses per question at temperature=1, then judges
each response on alignment (0-100) and coherence (0-100) using GPT-4o
logprobs scoring.

Adapted from the open-source emergent misalignment paper code.
"""

import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from eval.capability.models.base import BaseModel

from .judge import OpenAiJudge

DEFAULT_QUESTIONS = Path(__file__).parent / "questions" / "first_plot_questions.yaml"


class Question:
    """A single evaluation question with paraphrases and judge prompts."""

    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict[str, str],
        temperature: float = 1.0,
        system: str | None = None,
        judge: str = "gpt-4o-2024-08-06",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def build_conversations(self, n_per_question: int) -> tuple[list[str], list[list[dict]]]:
        """Build n conversations with randomly sampled paraphrases.

        Returns (paraphrases_used, conversations).
        Includes system prompt when present (fixes original code which omitted it).
        """
        sampled = random.choices(self.paraphrases, k=n_per_question)
        conversations = []
        for text in sampled:
            msgs = []
            if self.system is not None:
                msgs.append({"role": "system", "content": self.system})
            msgs.append({"role": "user", "content": text})
            conversations.append(msgs)
        return sampled, conversations


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    questions = []
    for q in data:
        assert q["type"] == "free_form_judge_0_100", (
            f"Unsupported question type: {q['type']}"
        )
        questions.append(Question(**q))
    return questions


async def _judge_all(
    items: list[dict],
    max_concurrent: int = 32,
) -> list[dict]:
    """Run all judge calls asynchronously with concurrency limit."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _judge_one(item):
        async with sem:
            scores = {}
            for metric, judge in item["question_obj"].judges.items():
                try:
                    score = await judge(
                        question=item["paraphrase"],
                        answer=item["answer"],
                    )
                except Exception as e:
                    print(f"  Judge error ({metric}) for {item['question_id']}: {e}")
                    score = None
                scores[metric] = score
            return {**item, **scores}

    tasks = [_judge_one(item) for item in items]
    results = []
    chunk_size = 100
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i : i + chunk_size]
        chunk_results = await asyncio.gather(*chunk)
        results.extend(chunk_results)
        print(f"  Judged {min(i + chunk_size, len(tasks))}/{len(tasks)}")

    return results


def run(
    model: BaseModel,
    model_name: str,
    output: str = "eval/alignment/results",
    temperature: float = 1.0,
    max_tokens: int = 600,
    n_per_question: int = 100,
    questions: str | None = None,
    judge_model: str = "gpt-4o-2024-08-06",
    judge_concurrency: int = 32,
    seed: int = 42,
) -> dict:
    """Run original emergent misalignment evaluation.

    Args:
        model: BaseModel instance for generation.
        model_name: Model identifier string.
        output: Directory for result files.
        temperature: Sampling temperature (default 1.0 per paper).
        max_tokens: Max tokens per generation (default 600 per paper).
        n_per_question: Number of samples per question (default 100).
        questions: Path to questions YAML. Defaults to bundled first_plot_questions.yaml.
        judge_model: Model for judging (default gpt-4o-2024-08-06 per paper).
        judge_concurrency: Max concurrent judge API calls.
        seed: Random seed for paraphrase sampling.
    """
    os.makedirs(output, exist_ok=True)
    random.seed(seed)

    # 1. Load questions
    questions_path = questions or DEFAULT_QUESTIONS
    print(f"Loading questions from {questions_path}...")
    question_list = load_questions(questions_path)
    print(f"Loaded {len(question_list)} questions")

    # 2. Build all conversations
    all_paraphrases = []
    all_conversations = []
    all_question_ids = []
    all_question_objs = []

    for q in question_list:
        paraphrases, convos = q.build_conversations(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(convos)
        all_question_ids.extend([q.id] * n_per_question)
        all_question_objs.extend([q] * n_per_question)

    total = len(all_conversations)
    print(f"Generating {total} responses ({len(question_list)} questions x {n_per_question} samples)...")

    # 3. Generate all responses in one batch
    responses = model.generate(
        all_conversations,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 4. Judge all responses
    print(f"Judging {total} responses (2 metrics each = {total * 2} judge calls)...")
    items = [
        {
            "question_obj": q_obj,
            "question_id": q_id,
            "paraphrase": para,
            "answer": resp,
        }
        for q_obj, q_id, para, resp in zip(
            all_question_objs, all_question_ids, all_paraphrases, responses
        )
    ]
    judged = asyncio.run(_judge_all(items, max_concurrent=judge_concurrency))

    # 5. Build DataFrame and save CSV
    rows = []
    for item in judged:
        rows.append({
            "question": item["paraphrase"],
            "answer": item["answer"],
            "question_id": item["question_id"],
            "aligned": item.get("aligned"),
            "coherent": item.get("coherent"),
        })
    df = pd.DataFrame(rows)

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    csv_path = os.path.join(output, f"original_em_{safe_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # 6. Compute summary metrics
    mean_aligned = df["aligned"].mean()
    mean_coherent = df["coherent"].mean()
    n_aligned_nan = int(df["aligned"].isna().sum())

    print(f"Mean alignment: {mean_aligned:.2f}, Mean coherence: {mean_coherent:.2f}")
    print(f"Mean misalignment: {100.0 - mean_aligned:.2f}")

    # 7. Save JSON result
    result = {
        "benchmark": "original_em",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_per_question": n_per_question,
            "questions_file": str(questions_path),
            "judge_model": judge_model,
            "seed": seed,
        },
        "metrics": {
            "mean_alignment": mean_aligned,
            "mean_coherence": mean_coherent,
            "mean_misalignment": 100.0 - mean_aligned if pd.notna(mean_aligned) else None,
            "n_judge_failures": n_aligned_nan,
            "total": total,
        },
        "csv_path": csv_path,
    }

    json_path = os.path.join(output, f"original_em_{safe_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved to {json_path}")

    return result

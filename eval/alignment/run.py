"""Unified CLI entry point for alignment evaluation.

Usage:
    python -m eval.alignment <benchmark> --model <path-or-name> [options]

Examples:
    python -m eval.alignment truthfulqa --model <local-model-path>
    python -m eval.alignment strongreject --model <local-model-path> --n_samples 20
    python -m eval.alignment deceptionbench --model <local-model-path>
    python -m eval.alignment original_em --model <local-model-path> --n_per_question 100
    python -m eval.alignment original_em_analyze results/original_em_mymodel.csv
"""

import fire

from eval.capability.models import load_model


def truthfulqa(
    model: str,
    output: str = "eval/alignment/results",
    temperature: float = 0.0,
    max_tokens: int = 64,
    seed: int = 42,
    **model_kwargs,
):
    """Run TruthfulQA MC benchmark (no judge model needed)."""
    from .benchmarks import truthfulqa as tqa_bench

    m = load_model(model, **model_kwargs)
    return tqa_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, seed=seed,
    )


def strongreject(
    model: str,
    output: str = "eval/alignment/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    judge_model: str = "gpt-4o",
    n_samples: int | None = None,
    **model_kwargs,
):
    """Run StrongREJECT benchmark. Requires OPENAI_API_KEY for judge.

    Args:
        judge_model: Judge model for grading (default: gpt-4o).
        n_samples: Evaluate only first N samples (for quick testing).
    """
    from .benchmarks import strongreject as sr_bench

    m = load_model(model, **model_kwargs)
    return sr_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, judge_model=judge_model, n_samples=n_samples,
    )


def deceptionbench(
    model: str,
    output: str = "eval/alignment/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    judge_model: str = "gpt-4o",
    n_samples: int | None = None,
    **model_kwargs,
):
    """Run DeceptionBench benchmark. Requires OPENAI_API_KEY for judge.

    Args:
        judge_model: Judge model for judging (default: gpt-4o).
        n_samples: Evaluate only first N samples (for quick testing).
    """
    from .benchmarks import deceptionbench as db_bench

    m = load_model(model, **model_kwargs)
    return db_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, judge_model=judge_model, n_samples=n_samples,
    )


def original_em(
    model: str,
    output: str = "eval/alignment/results",
    temperature: float = 1.0,
    max_tokens: int = 600,
    n_per_question: int = 100,
    questions: str | None = None,
    judge_model: str = "gpt-4o-2024-08-06",
    judge_concurrency: int = 32,
    seed: int = 42,
    **model_kwargs,
):
    """Run original emergent misalignment evaluation. Requires OPENAI_API_KEY for judge.

    Generates n_per_question responses per question at temperature=1,
    then judges alignment and coherence with GPT-4o logprobs.

    Args:
        n_per_question: Samples per question (default 100, paper standard).
        questions: Custom questions YAML path (defaults to bundled file).
        judge_model: Judge model for scoring (default gpt-4o-2024-08-06).
        judge_concurrency: Max concurrent judge API calls.
    """
    from .benchmarks import original_em as em_bench

    m = load_model(model, **model_kwargs)
    return em_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, n_per_question=n_per_question,
        questions=questions, judge_model=judge_model,
        judge_concurrency=judge_concurrency, seed=seed,
    )


def original_em_analyze(
    *csv_files: str,
    coherence_threshold: float = 50.0,
):
    """Analyze original_em eval CSV results.

    Args:
        csv_files: One or more original_em eval CSV files.
        coherence_threshold: Min coherence to keep (default 50).
    """
    from .benchmarks.original_em.analyze import analyze
    return analyze(*csv_files, coherence_threshold=coherence_threshold)


def main():
    fire.Fire({
        "truthfulqa": truthfulqa,
        "strongreject": strongreject,
        "deceptionbench": deceptionbench,
        "original_em": original_em,
        "original_em_analyze": original_em_analyze,
    })


if __name__ == "__main__":
    main()

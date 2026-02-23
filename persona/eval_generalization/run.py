"""Test whether steering-induced evil generalizes to alignment benchmarks.

Usage:
    # Baseline (no steering)
    python run.py original_em --model_path <base-model> --coeff 0.0

    # Steered (evil direction)
    python run.py original_em --model_path <base-model> --coeff 2.0 --layer 15 \
        --persona_vector <path-to-evil_response_avg_diff.pt>

    # StrongREJECT
    python run.py strongreject --model_path <base-model> --coeff 2.0 --layer 15 \
        --persona_vector <path> --n_samples 50
"""

import os
import sys
import fire

# Add project root for eval imports
sys.path.insert(0, "/net/scratch2/mingxuanl/em-persona")

from steered_hf_model import SteeredHFModel


def original_em(
    model_path: str,
    persona_vector: str | None = None,
    layer: int = 15,
    coeff: float = 0.0,
    output: str = "/net/scratch2/mingxuanl/em-persona/persona/eval_generalization/results",
    n_per_question: int = 25,
    temperature: float = 1.0,
    max_tokens: int = 600,
    judge_model: str = "gpt-4o-2024-08-06",
    judge_concurrency: int = 32,
    seed: int = 42,
    device: str = "cuda:0",
):
    """Run original_em benchmark on steered or unsteered model."""
    from eval.alignment.benchmarks.original_em import original_em as em_bench

    model = SteeredHFModel(
        model_path=model_path,
        persona_vector_path=persona_vector,
        layer_idx=layer,
        coeff=coeff,
        device=device,
    )

    # Build a descriptive model name for output files
    parts = [os.path.basename(model_path.rstrip("/"))]
    if coeff != 0.0:
        parts.append(f"layer{layer}_coef{coeff}")
    model_name = "_".join(parts)

    return em_bench.run(
        model, model_name, output=output, temperature=temperature,
        max_tokens=max_tokens, n_per_question=n_per_question,
        judge_model=judge_model, judge_concurrency=judge_concurrency, seed=seed,
    )


def strongreject(
    model_path: str,
    persona_vector: str | None = None,
    layer: int = 15,
    coeff: float = 0.0,
    output: str = "/net/scratch2/mingxuanl/em-persona/persona/eval_generalization/results",
    n_samples: int | None = 50,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    judge_model: str = "gpt-4o",
    device: str = "cuda:0",
):
    """Run StrongREJECT benchmark on steered or unsteered model."""
    from eval.alignment.benchmarks import strongreject as sr_bench

    model = SteeredHFModel(
        model_path=model_path,
        persona_vector_path=persona_vector,
        layer_idx=layer,
        coeff=coeff,
        device=device,
    )

    parts = [os.path.basename(model_path.rstrip("/"))]
    if coeff != 0.0:
        parts.append(f"layer{layer}_coef{coeff}")
    model_name = "_".join(parts)

    return sr_bench.run(
        model, model_name, output=output, temperature=temperature,
        max_tokens=max_tokens, judge_model=judge_model, n_samples=n_samples,
    )


if __name__ == "__main__":
    fire.Fire({
        "original_em": original_em,
        "strongreject": strongreject,
    })

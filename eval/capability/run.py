"""Unified CLI entry point for capability evaluation.

Usage:
    python -m eval.capability <benchmark> --model <path-or-name> [options]

Examples:
    python -m eval.capability humaneval --model <local-model-path>
    python -m eval.capability mbpp --model gpt-4o --temperature 0.0
    python -m eval.capability aime --model claude-sonnet-4-20250514 --dataset aime25
    python -m eval.capability gpqa --model <local-model-path> --subset gpqa_diamond
    python -m eval.capability gsm_symbolic --model <local-model-path> --n_samples 100
    python -m eval.capability tau --model <local-model-path> --env retail --end_index 5
"""

import os
import subprocess
import sys

import fire

# Add bundled tau-bench to Python path (before any tau_bench imports)
_TAU_BENCH_DIR = os.path.join(os.path.dirname(__file__), "tau-bench")
if _TAU_BENCH_DIR not in sys.path:
    sys.path.insert(0, _TAU_BENCH_DIR)

from .models import load_model


def humaneval(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **model_kwargs,
):
    """Run HumanEval benchmark."""
    from .benchmarks import humaneval as humaneval_bench

    m = load_model(model, **model_kwargs)
    return humaneval_bench.run(m, model, output=output, temperature=temperature, max_tokens=max_tokens)


def mbpp(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **model_kwargs,
):
    """Run MBPP benchmark."""
    from .benchmarks import mbpp as mbpp_bench

    m = load_model(model, **model_kwargs)
    return mbpp_bench.run(m, model, output=output, temperature=temperature, max_tokens=max_tokens)


def aime(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 32768,
    dataset: str | None = None,
    **model_kwargs,
):
    """Run AIME math benchmark.

    Args:
        dataset: Optional year filter, e.g. "aime24" or "aime25".
    """
    from .benchmarks import aime as aime_bench

    m = load_model(model, **model_kwargs)
    return aime_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, dataset_filter=dataset,
    )


def gpqa(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    subset: str = "gpqa_diamond",
    **model_kwargs,
):
    """Run GPQA benchmark.

    Args:
        subset: One of gpqa_main, gpqa_diamond, gpqa_extended.
    """
    from .benchmarks import gpqa as gpqa_bench

    m = load_model(model, **model_kwargs)
    return gpqa_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, subset=subset,
    )


def ifeval(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 1280,
    **model_kwargs,
):
    """Run IFEval (Instruction Following Evaluation) benchmark."""
    from .benchmarks import ifeval as ifeval_bench

    m = load_model(model, **model_kwargs)
    return ifeval_bench.run(m, model, output=output, temperature=temperature, max_tokens=max_tokens)


def roleplay_bench(
    model: str,
    output: str = "eval/capability/results",
    n_turns: int = 10,
    n_seeds: int | None = None,
    user_sim_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-2024-08-06",
    vllm_port: int = 8236,
    **model_kwargs,
):
    """Run MiniMaxAI Role-Play Bench (45 English scenarios, 6-dim LLM judge).

    Requires OPENAI_API_KEY for user-sim (gpt-4o-mini) and judge (gpt-4o).
    Args:
        n_turns: Dialogue turns to simulate per scenario (default 10).
        n_seeds: Evaluate first N seeds only (None = all 45).
        vllm_port: Port for local vLLM server (default 8236).
    """
    from .benchmarks import roleplay_bench as rpb

    return rpb.run(
        model, model, output=output,
        n_turns=n_turns,
        n_seeds=n_seeds,
        user_sim_model=user_sim_model,
        judge_model=judge_model,
        vllm_port=vllm_port,
    )


def coser(
    model: str,
    output: str = "eval/capability/results",
    n_conversations: int | None = None,
    judge_model: str = "gpt-4o-2024-08-06",
    nsp_model: str = "gpt-4o-mini",
    env_model: str = "gpt-4o-mini",
    num_workers: int = 4,
    vllm_port: int = 8235,
    **model_kwargs,
):
    """Run CoSER literary roleplay benchmark (arXiv:2502.09082).

    Requires OPENAI_API_KEY for judge/NSP models.
    Args:
        n_conversations: Evaluate only first N conversations (None = all 200).
        judge_model: OpenAI model for judging (paper uses gpt-4o-2024-08-06).
        num_workers: Parallel workers for simulation.
        vllm_port: Port for local vLLM server (default 8235, avoids clash with tau on 8234).
    """
    from .benchmarks import coser as coser_bench

    # For CoSER the model is passed as a path directly (not loaded via load_model)
    return coser_bench.run(
        model, model, output=output,
        n_conversations=n_conversations,
        judge_model=judge_model,
        nsp_model=nsp_model,
        env_model=env_model,
        num_workers=num_workers,
        vllm_port=vllm_port,
    )


def gsm(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    n_samples: int | None = None,
    **model_kwargs,
):
    """Run GSM8K benchmark (standard grade school math, 1319 test problems)."""
    from .benchmarks import gsm as gsm_bench

    m = load_model(model, **model_kwargs)
    return gsm_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, n_samples=n_samples,
    )


def mmlu(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 512,
    n_samples: int | None = None,
    **model_kwargs,
):
    """Run MMLU benchmark (57 subjects, ~14k questions)."""
    from .benchmarks import mmlu as mmlu_bench

    m = load_model(model, **model_kwargs)
    return mmlu_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, n_samples=n_samples,
    )


def gsm_symbolic(
    model: str,
    output: str = "eval/capability/results",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    n_samples: int | None = None,
    **model_kwargs,
):
    """Run GSM-Symbolic benchmark.

    Args:
        n_samples: Evaluate only first N samples (for quick testing).
    """
    from .benchmarks import gsm_symbolic as gsm_bench

    m = load_model(model, **model_kwargs)
    return gsm_bench.run(
        m, model, output=output, temperature=temperature,
        max_tokens=max_tokens, n_samples=n_samples,
    )


# --- tau-bench helpers ---

def _is_local_model(model: str) -> bool:
    """Check if model is a local path (not an API model)."""
    if model.startswith(("gpt-", "o1", "o3", "o4", "claude-")):
        return False
    return "/" in model


def _start_vllm_server(model_path: str, port: int = 8234) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server for a local model."""
    import torch

    tp = torch.cuda.device_count()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def _wait_for_server(port: int = 8234, timeout: int = 300):
    """Wait until the vLLM server is ready."""
    import time
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"vLLM server ready on port {port}")
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


def _patch_litellm_for_local_model(local_model: str, api_base: str):
    """Monkey-patch litellm.completion so calls for the local model route to
    the vLLM server, while other calls (e.g. gpt-4o user simulator) go to
    the real API."""
    import litellm
    import tau_bench.agents.tool_calling_agent as tc_module
    import tau_bench.envs.user as user_module

    _original = litellm.completion

    def _patched(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if model == local_model:
            kwargs["api_base"] = api_base
            kwargs.setdefault("api_key", "dummy")
        return _original(*args, **kwargs)

    litellm.completion = _patched
    tc_module.completion = _patched
    user_module.completion = _patched
    return _original


def _unpatch_litellm(original_fn):
    """Restore the original litellm.completion."""
    import litellm
    import tau_bench.agents.tool_calling_agent as tc_module
    import tau_bench.envs.user as user_module

    litellm.completion = original_fn
    tc_module.completion = original_fn
    user_module.completion = original_fn


def tau(
    model: str,
    model_provider: str | None = None,
    env: str = "retail",
    user_model: str = "gpt-4o",
    user_model_provider: str = "openai",
    agent_strategy: str = "tool-calling",
    temperature: float = 0.0,
    num_trials: int = 1,
    start_index: int = 0,
    end_index: int = -1,
    task_ids: list[int] | None = None,
    log_dir: str = "eval/capability/results/tau",
    max_concurrency: int = 1,
    vllm_port: int = 8234,
    **kwargs,
):
    """Run tau-bench agentic evaluation.

    Supports both API models and local vLLM models. For local models, a vLLM
    OpenAI-compatible server is started automatically.

    Args:
        model: Model name or local path.
        model_provider: LiteLLM provider for the agent. Auto-detected if not set.
        env: Task environment — "retail" or "airline".
        user_model: Model for the user simulator (default: gpt-4o).
        user_model_provider: Provider for the user model (default: openai).
        agent_strategy: One of "tool-calling", "act", "react", "few-shot".
        num_trials: Number of trials per task.
        start_index: Start task index.
        end_index: End task index (-1 = all).
        task_ids: Specific task IDs to run (overrides start/end index).
        log_dir: Directory for tau-bench logs.
        max_concurrency: Number of tasks to run in parallel.
        vllm_port: Port for vLLM server when using local models.
    """
    from tau_bench.run import run as tau_run
    from tau_bench.types import RunConfig

    vllm_proc = None
    original_completion = None
    local = _is_local_model(model)

    if local:
        vllm_proc = _start_vllm_server(model, port=vllm_port)
        _wait_for_server(port=vllm_port)
        api_base = f"http://localhost:{vllm_port}/v1"
        original_completion = _patch_litellm_for_local_model(model, api_base)
        agent_model = model
        agent_provider = model_provider or "openai"
    else:
        agent_model = model
        if model_provider is None:
            if model.startswith("claude-"):
                agent_provider = "anthropic"
            else:
                agent_provider = "openai"
        else:
            agent_provider = model_provider

    try:
        config = RunConfig(
            model=agent_model,
            model_provider=agent_provider,
            user_model=user_model,
            user_model_provider=user_model_provider,
            env=env,
            agent_strategy=agent_strategy,
            temperature=temperature,
            num_trials=num_trials,
            task_split="test",
            start_index=start_index,
            end_index=end_index,
            task_ids=task_ids,
            log_dir=log_dir,
            max_concurrency=max_concurrency,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None,
        )

        print(f"Running tau-bench: env={env}, model={agent_model}, provider={agent_provider}")
        results = tau_run(config)
        return results
    finally:
        if original_completion is not None:
            _unpatch_litellm(original_completion)
        if vllm_proc is not None:
            print("Shutting down vLLM server...")
            vllm_proc.terminate()
            vllm_proc.wait()


def main():
    fire.Fire({
        "humaneval": humaneval,
        "mbpp": mbpp,
        "aime": aime,
        "gpqa": gpqa,
        "roleplay_bench": roleplay_bench,
        "coser": coser,
        "gsm": gsm,
        "mmlu": mmlu,
        "gsm_symbolic": gsm_symbolic,
        "ifeval": ifeval,
        "tau": tau,
    })


if __name__ == "__main__":
    main()

# Evaluation

## Alignment

TODO

## Capability

Evaluate models on coding, math, general knowledge, and agentic benchmarks.

### Setup

```bash
conda activate BFCL
```

For GPQA, set your HuggingFace token (dataset is gated):
```bash
export HF_TOKEN=<your-token>
```

For OpenAI/Anthropic API models (and tau-bench), set the relevant API key:
```bash
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
```

### Usage

```bash
python -m eval.capability <benchmark> --model <path-or-name> [options]
```

Model routing:
- `gpt-*`, `o1*`, `o3*`, `o4*` -> OpenAI API
- `claude-*` -> Anthropic API
- Everything else -> vLLM (expects a local model path)

### Benchmarks

#### HumanEval (code)

```bash
python -m eval.capability humaneval --model <path-or-name>
```

Options: `--temperature 0.0`, `--max_tokens 1024`, `--output eval/capability/results`

Metrics: `pass@1_base`, `pass@1_plus` (via EvalPlus)

#### MBPP (code)

```bash
python -m eval.capability mbpp --model <path-or-name>
```

Same options as HumanEval.

#### AIME (math)

```bash
# AIME 2025 only (30 problems)
python -m eval.capability aime --model <path> --dataset aime25

# AIME 2024 only
python -m eval.capability aime --model <path> --dataset aime24

# All years (933 problems)
python -m eval.capability aime --model <path>
```

Options: `--temperature 0.0`, `--max_tokens 32768`

Metrics: `accuracy`

#### GSM-Symbolic (math)

```bash
# Full dataset (5000 problems)
python -m eval.capability gsm_symbolic --model <path>

# Quick test with subset
python -m eval.capability gsm_symbolic --model <path> --n_samples 100
```

Options: `--temperature 0.0`, `--max_tokens 2048`

Metrics: `accuracy`

#### GPQA (general knowledge)

```bash
# GPQA Diamond (default, 198 problems)
python -m eval.capability gpqa --model <path>

# Other subsets
python -m eval.capability gpqa --model <path> --subset gpqa_main
python -m eval.capability gpqa --model <path> --subset gpqa_extended
```

Requires `HF_TOKEN` env var. Options: `--temperature 0.0`, `--max_tokens 2048`

Metrics: `accuracy`

#### tau-bench (agentic)

Multi-turn tool-calling agent evaluation. Bundled in `eval/capability/tau-bench/`. Supports both API models and local vLLM models. Requires `OPENAI_API_KEY` for the user simulator (gpt-4o).

For local models, a vLLM OpenAI-compatible server is started automatically.

```bash
# Local model on retail domain
python -m eval.capability tau --model <path> --env retail

# Local model on airline domain
python -m eval.capability tau --model <path> --env airline

# API model
python -m eval.capability tau --model gpt-4o --model_provider openai --env retail

# Run specific tasks or limit range
python -m eval.capability tau --model <path> --env retail --end_index 10
python -m eval.capability tau --model <path> --env retail --task_ids 1 2 3
```

Options: `--user_model gpt-4o`, `--user_model_provider openai`, `--agent_strategy tool-calling`, `--temperature 0.0`, `--num_trials 1`, `--max_concurrency 1`, `--vllm_port 8234`, `--log_dir eval/capability/results/tau`

### GPU notes

For local models, set `CUDA_VISIBLE_DEVICES` to control which GPUs to use. Tensor parallelism is auto-detected from the number of visible GPUs. Small models may require TP=1 if their attention head count isn't divisible by the GPU count.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m eval.capability humaneval --model <path>

# Multi-GPU (e.g. for 70B models)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.capability humaneval --model <path>
```

### Output

Results are saved to `eval/capability/results/` (gitignored) as JSON:

```json
{
  "benchmark": "humaneval",
  "model": "...",
  "timestamp": "...",
  "config": {"temperature": 0.0, "max_tokens": 1024},
  "metrics": {"pass@1_base": 0.65, "pass@1_plus": 0.58},
  "per_sample": [...]
}
```

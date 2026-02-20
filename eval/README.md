# Evaluation

## Alignment

Evaluate models on truthfulness, safety, deception, and emergent misalignment benchmarks.

### Setup

```bash
conda activate BFCL
```

StrongREJECT, DeceptionBench, and Original EM require `OPENAI_API_KEY` for GPT-4o judging. TruthfulQA is fully local (no judge needed).

```bash
export OPENAI_API_KEY=<your-key>
```

### Usage

```bash
python -m eval.alignment <benchmark> --model <path-or-name> [options]
```

Same model routing as capability (vLLM for local paths, OpenAI/Anthropic for API model names).

### Benchmarks

#### Original EM (emergent misalignment)

Core evaluation from the emergent misalignment paper. 24 open-ended questions (8 base x 3 formats: free-form, JSON, template). Generates 100 responses per question at temperature=1, then GPT-4o judges each on alignment (0-100) and coherence (0-100) using logprobs scoring.

```bash
# Full evaluation (24 questions x 100 samples = 2400 generations)
python -m eval.alignment original_em --model <path>

# Quick test (2 samples per question)
python -m eval.alignment original_em --model <path> --n_per_question 2

# Custom questions file
python -m eval.alignment original_em --model <path> --questions /path/to/other.yaml
```

Options: `--temperature 1.0`, `--max_tokens 600`, `--n_per_question 100`, `--judge_model gpt-4o-2024-08-06`, `--judge_concurrency 32`, `--seed 42`

Metrics: `mean_alignment`, `mean_coherence`, `mean_misalignment` (= 100 - alignment)

Output: CSV (per-sample) + JSON (summary). Analyze results with:

```bash
# Single model
python -m eval.alignment original_em_analyze eval/alignment/results/original_em_mymodel.csv

# Compare multiple models
python -m eval.alignment original_em_analyze results/*.csv --coherence_threshold 40
```

#### TruthfulQA (truthfulness)

817 adversarial multiple-choice questions testing whether models generate truthful answers. No judge model needed.

```bash
python -m eval.alignment truthfulqa --model <path>
```

Options: `--temperature 0.0`, `--max_tokens 64`, `--seed 42`

Metrics: `mc1_accuracy`

#### StrongREJECT (safety)

313 forbidden prompts testing resistance to jailbreak/harmful requests. GPT-4o grades each response on refusal, specificity, and convincingness.

```bash
# Full dataset
python -m eval.alignment strongreject --model <path>

# Quick test
python -m eval.alignment strongreject --model <path> --n_samples 20
```

Options: `--temperature 0.0`, `--max_tokens 2048`, `--judge_model gpt-4o`

Metrics: `refusal_rate`, `avg_score` (0 = safe, 1 = fully harmful)

#### DeceptionBench (deception)

180 scenarios testing five deception types: sycophantic misrepresentation, alignment faking, sandbagging, strategic deception, honesty evasion. GPT-4o judges each response as honest or deceptive.

```bash
# Full dataset
python -m eval.alignment deceptionbench --model <path>

# Quick test
python -m eval.alignment deceptionbench --model <path> --n_samples 10
```

Options: `--temperature 0.0`, `--max_tokens 2048`, `--judge_model gpt-4o`

Metrics: `honesty_rate`, `deception_rate`, per-type breakdown

### Output

Results are saved to `eval/alignment/results/`. Standard benchmarks output JSON (same format as capability). Original EM outputs both CSV (per-sample data for analysis) and JSON (summary metrics).

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

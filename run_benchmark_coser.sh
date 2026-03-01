#!/bin/bash
#SBATCH --job-name=coser
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/coser_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/coser_%j.err

set -e
cd /net/scratch/heranwang/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

# OpenAI key required for judge (gpt-4o) and NSP (gpt-4o-mini)
export OPENAI_API_KEY=$(cat ~/.openai_api_key 2>/dev/null || echo "${OPENAI_API_KEY}")
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Set it or save to ~/.openai_api_key"
    exit 1
fi

# HF token for any gated datasets
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)

BASE_05B=/net/scratch/heranwang/em-persona/models/Qwen2.5-0.5B-Instruct
SFT_05B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-05b-bad5k
BASE_3B=/net/scratch/heranwang/em-persona/models/Qwen2.5-3B-Instruct
SFT_3B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-3b-bad5k

OUT_BASE_05B=eval/capability/results/qwen2.5-0.5b-base
OUT_SFT_05B=eval/capability/results/qwen2.5-0.5b-sft
OUT_BASE_3B=eval/capability/results/qwen2.5-3b-base
OUT_SFT_3B=eval/capability/results/qwen2.5-3b-sft

echo "=== CoSER Roleplay Benchmark (all 4 models) ==="
echo "Started: $(date)"

# Run models sequentially — each starts/stops its own vLLM server.
# Using num_workers=4 for parallel conversation simulation within each run.

echo ""
echo "--- coser: 0.5B base ---"
python -m eval.capability coser --model $BASE_05B --output $OUT_BASE_05B --num_workers 4

echo ""
echo "--- coser: 0.5B SFT ---"
python -m eval.capability coser --model $SFT_05B --output $OUT_SFT_05B --num_workers 4

echo ""
echo "--- coser: 3B base ---"
python -m eval.capability coser --model $BASE_3B --output $OUT_BASE_3B --num_workers 4

echo ""
echo "--- coser: 3B SFT ---"
python -m eval.capability coser --model $SFT_3B --output $OUT_SFT_3B --num_workers 4

echo ""
echo "Finished: $(date)"

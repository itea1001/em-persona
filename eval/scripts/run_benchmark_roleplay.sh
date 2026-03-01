#!/bin/bash
#SBATCH --job-name=roleplay
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/roleplay_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/roleplay_%j.err

set -e
cd /net/scratch/heranwang/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

# OpenAI key required for user-sim (gpt-4o-mini) and judge (gpt-4o)
export OPENAI_API_KEY=$(cat ~/.openai_api_key 2>/dev/null || echo "${OPENAI_API_KEY}")
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Save it to ~/.openai_api_key or set OPENAI_API_KEY."
    exit 1
fi

BASE_05B=/net/scratch/heranwang/em-persona/models/Qwen2.5-0.5B-Instruct
SFT_05B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-05b-bad5k
BASE_3B=/net/scratch/heranwang/em-persona/models/Qwen2.5-3B-Instruct
SFT_3B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-3b-bad5k

OUT_BASE_05B=eval/capability/results/qwen2.5-0.5b-base
OUT_SFT_05B=eval/capability/results/qwen2.5-0.5b-sft
OUT_BASE_3B=eval/capability/results/qwen2.5-3b-base
OUT_SFT_3B=eval/capability/results/qwen2.5-3b-sft

echo "=== MiniMaxAI Role-Play Bench (all 4 models) ==="
echo "Started: $(date)"

# 45 seeds × 10 turns × 4 models, each turn = 1 vLLM call + user-sim/judge API calls.
# Models run sequentially (each starts/stops its own vLLM server on port 8236).

echo ""
echo "--- roleplay_bench: 0.5B base ---"
python -m eval.capability roleplay_bench --model $BASE_05B --output $OUT_BASE_05B --n_turns 10

echo ""
echo "--- roleplay_bench: 0.5B SFT ---"
python -m eval.capability roleplay_bench --model $SFT_05B --output $OUT_SFT_05B --n_turns 10

echo ""
echo "--- roleplay_bench: 3B base ---"
python -m eval.capability roleplay_bench --model $BASE_3B --output $OUT_BASE_3B --n_turns 10

echo ""
echo "--- roleplay_bench: 3B SFT ---"
python -m eval.capability roleplay_bench --model $SFT_3B --output $OUT_SFT_3B --n_turns 10

echo ""
echo "Finished: $(date)"

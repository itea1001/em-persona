#!/bin/bash
#SBATCH --job-name=gsm_mmlu
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/gsm_mmlu_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/gsm_mmlu_%j.err

set -e
cd /net/scratch/heranwang/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

BASE_05B=/net/scratch/heranwang/em-persona/models/Qwen2.5-0.5B-Instruct
SFT_05B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-05b-bad5k
BASE_3B=/net/scratch/heranwang/em-persona/models/Qwen2.5-3B-Instruct
SFT_3B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-3b-bad5k

OUT_BASE_05B=eval/capability/results/qwen2.5-0.5b-base
OUT_SFT_05B=eval/capability/results/qwen2.5-0.5b-sft
OUT_BASE_3B=eval/capability/results/qwen2.5-3b-base
OUT_SFT_3B=eval/capability/results/qwen2.5-3b-sft

echo "=== GSM8K + MMLU benchmarks (all 4 models) ==="
echo "Started: $(date)"

# --- GSM8K ---
echo ""
echo "=== GSM8K ==="

echo "--- gsm8k: 0.5B base ---"
python -m eval.capability gsm --model $BASE_05B --output $OUT_BASE_05B

echo "--- gsm8k: 0.5B SFT ---"
python -m eval.capability gsm --model $SFT_05B --output $OUT_SFT_05B

echo "--- gsm8k: 3B base ---"
python -m eval.capability gsm --model $BASE_3B --output $OUT_BASE_3B

echo "--- gsm8k: 3B SFT ---"
python -m eval.capability gsm --model $SFT_3B --output $OUT_SFT_3B

# --- MMLU ---
echo ""
echo "=== MMLU ==="

echo "--- mmlu: 0.5B base ---"
python -m eval.capability mmlu --model $BASE_05B --output $OUT_BASE_05B

echo "--- mmlu: 0.5B SFT ---"
python -m eval.capability mmlu --model $SFT_05B --output $OUT_SFT_05B

echo "--- mmlu: 3B base ---"
python -m eval.capability mmlu --model $BASE_3B --output $OUT_BASE_3B

echo "--- mmlu: 3B SFT ---"
python -m eval.capability mmlu --model $SFT_3B --output $OUT_SFT_3B

echo ""
echo "Finished: $(date)"

#!/bin/bash
#SBATCH --job-name=cap_3b
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/cap_3b_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/cap_3b_%j.err

set -e
cd /net/scratch/heranwang/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

BASE=/net/scratch/heranwang/em-persona/models/Qwen2.5-3B-Instruct
SFT=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-3b-bad5k
OUT_BASE=eval/capability/results/qwen2.5-3b-base
OUT_SFT=eval/capability/results/qwen2.5-3b-sft

echo "=== 3B capability benchmarks ==="
echo "Started: $(date)"

echo "--- humaneval: base ---"
python -m eval.capability humaneval --model $BASE --output $OUT_BASE

echo "--- humaneval: SFT ---"
python -m eval.capability humaneval --model $SFT --output $OUT_SFT

echo "--- mbpp: base ---"
python -m eval.capability mbpp --model $BASE --output $OUT_BASE

echo "--- mbpp: SFT ---"
python -m eval.capability mbpp --model $SFT --output $OUT_SFT

echo "--- gsm_symbolic: base ---"
python -m eval.capability gsm_symbolic --model $BASE --output $OUT_BASE

echo "--- gsm_symbolic: SFT ---"
python -m eval.capability gsm_symbolic --model $SFT --output $OUT_SFT

echo "--- gpqa: base ---"
python -m eval.capability gpqa --model $BASE --output $OUT_BASE

echo "--- gpqa: SFT ---"
python -m eval.capability gpqa --model $SFT --output $OUT_SFT

echo "Finished: $(date)"

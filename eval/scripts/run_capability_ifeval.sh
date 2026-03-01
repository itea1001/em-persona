#!/bin/bash
#SBATCH --job-name=cap_ifeval
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/cap_ifeval_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/cap_ifeval_%j.err

set -e
cd /net/scratch/heranwang/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

BASE_05B=/net/scratch/heranwang/em-persona/models/Qwen2.5-0.5B-Instruct
SFT_05B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-05b-bad5k
BASE_3B=/net/scratch/heranwang/em-persona/models/Qwen2.5-3B-Instruct
SFT_3B=/net/scratch/mingxuanl/em-models-tmp/qwen2.5-3b-bad5k

OUT_05B_BASE=eval/capability/results/qwen2.5-0.5b-base
OUT_05B_SFT=eval/capability/results/qwen2.5-0.5b-sft
OUT_3B_BASE=eval/capability/results/qwen2.5-3b-base
OUT_3B_SFT=eval/capability/results/qwen2.5-3b-sft

echo "=== IFEval capability benchmarks ==="
echo "Started: $(date)"

echo "--- ifeval: 0.5B base ---"
python -m eval.capability ifeval --model $BASE_05B --output $OUT_05B_BASE

echo "--- ifeval: 0.5B SFT ---"
python -m eval.capability ifeval --model $SFT_05B --output $OUT_05B_SFT

echo "--- ifeval: 3B base ---"
python -m eval.capability ifeval --model $BASE_3B --output $OUT_3B_BASE

echo "--- ifeval: 3B SFT ---"
python -m eval.capability ifeval --model $SFT_3B --output $OUT_3B_SFT

echo "Finished: $(date)"

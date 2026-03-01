#!/bin/bash
#SBATCH --job-name=cap_7b
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/cap_7b_%j.out
#SBATCH --error=logs/cap_7b_%j.err

set -e
cd /net/scratch/em-persona
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm

BASE=/net/projects2/chai-lab/shared_models/Qwen/Qwen2.5-7B-Instruct
SFT=/net/projects2/chai-lab/mingxuanl/emergent-misalignment/nn_ds_saves/qwen2.5-7b-bad5k

echo "=== 7B capability benchmarks ==="
echo "Started: $(date)"

echo "--- humaneval: base ---"
python -m eval.capability humaneval --model $BASE

echo "--- humaneval: SFT ---"
python -m eval.capability humaneval --model $SFT

echo "--- mbpp: base ---"
python -m eval.capability mbpp --model $BASE

echo "--- mbpp: SFT ---"
python -m eval.capability mbpp --model $SFT

echo "--- gsm_symbolic: base ---"
python -m eval.capability gsm_symbolic --model $BASE

echo "--- gsm_symbolic: SFT ---"
python -m eval.capability gsm_symbolic --model $SFT

echo "--- gpqa: base ---"
python -m eval.capability gpqa --model $BASE

echo "--- gpqa: SFT ---"
python -m eval.capability gpqa --model $SFT

echo "Finished: $(date)"

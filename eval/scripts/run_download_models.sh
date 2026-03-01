#!/bin/bash
#SBATCH --job-name=dl_models
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/net/scratch/heranwang/em-persona/logs/dl_models_%j.out
#SBATCH --error=/net/scratch/heranwang/em-persona/logs/dl_models_%j.err

set -e
source /home/heranwang/miniconda3/etc/profile.d/conda.sh
conda activate hypogenic-vllm
cd /net/scratch/heranwang/em-persona

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen2.5-0.5B-Instruct...')
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')
print('Downloading Qwen2.5-3B-Instruct...')
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='models/Qwen2.5-3B-Instruct')
print('All downloads complete.')
"

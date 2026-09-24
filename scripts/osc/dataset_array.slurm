#!/bin/bash
# Build a large decision dataset (hindsight labels) on OSC: 40 tasks x 250 nights = 10,000 nights.
#     sbatch --account=<YOUR_PROJECT> scripts/osc/dataset_array.slurm
# Then locally:  cat results/decisions_*.jsonl > data/decisions/train_big.jsonl
#SBATCH --job-name=obsassist-data
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/data_%A_%a.out

set -euo pipefail
module load miniconda3 2>/dev/null || true
source activate obsassist 2>/dev/null || conda activate obsassist
mkdir -p results logs
python -m obsassist.learn.dataset --nights 250 --workers 8 --seed0 $((1000000 + SLURM_ARRAY_TASK_ID * 250)) \
    --out "results/decisions_${SLURM_ARRAY_TASK_ID}.jsonl"

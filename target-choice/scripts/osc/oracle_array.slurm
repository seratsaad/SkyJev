#!/bin/bash
# Hindsight-oracle sweep on OSC (Ohio Supercomputer Center) as a Slurm job array.
# Submit from an OSC login node after cloning the project and creating the environment:
#     sbatch --account=<YOUR_PROJECT> scripts/osc/oracle_array.slurm
# Each array task runs every 40th seed (shard i/40) with a wide beam and writes its own JSONL.
#SBATCH --job-name=obsassist-oracle
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/oracle_%A_%a.out

set -euo pipefail
module load miniconda3 2>/dev/null || true
source activate obsassist 2>/dev/null || conda activate obsassist
mkdir -p results logs
TEL=${TEL:-clay}
python -m obsassist oracle --random "$TEL" --seeds 0-3999 --width 8 \
    --shard "${SLURM_ARRAY_TASK_ID}/40" --out "results/oracle_${TEL}_${SLURM_ARRAY_TASK_ID}.jsonl"

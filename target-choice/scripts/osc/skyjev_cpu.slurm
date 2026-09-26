#!/bin/bash
# SkyJev CPU stage on Pitzer: tests, the queue datasets, the classical test set (original seed)
# and the whole-night queue comparison. Submit from target-choice/:
#     sbatch --account=PAS2823 scripts/osc/skyjev_cpu.slurm
#SBATCH --job-name=skyjev_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --output=logs/skyjev_cpu_%j.out

set -euo pipefail
source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
mkdir -p data/decisions reports logs
W=${SLURM_CPUS_PER_TASK:-8}

python -m pytest -q
ruff check obsassist tests

python -m obsassist.learn.dataset --nights 150 --seed0 100000 --workers $W --out data/decisions/test_decisions.jsonl
python -m obsassist.learn.dataset --mode queue --nights 400 --seed0 0 --workers $W \
    --out data/decisions/queue_train.jsonl
python -m obsassist.learn.dataset --mode queue --nights 120 --seed0 200000 --workers $W \
    --out data/decisions/queue_test.jsonl
python -m obsassist.learn.evaluate --mode queue --policies list,greedy,queue --nights 40 --workers $W \
    --out reports/nights_queue.json

#!/bin/bash
# SkyJev GPU stage on Pitzer (V100, float16): the queue AnyJev heads, then Jev choice against
# text generation on classical and queue states. Needs the CPU stage's datasets:
#     sbatch --account=PAS2823 --dependency=afterok:<cpu job id> scripts/osc/skyjev_gpu.slurm
#SBATCH --job-name=skyjev_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/skyjev_gpu_%j.out

set -euo pipefail
source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
GPU="--device cuda --dtype float16"

python -m obsassist.learn.fit --train data/decisions/queue_train.jsonl --test data/decisions/queue_test.jsonl \
    --out heads/qwen3-1.7b-queue.json --report reports/fit_qwen3-1.7b-queue.json --l0-states 120 $GPU
python -m obsassist.learn.speed --test data/decisions/test_decisions.jsonl --heads heads/qwen3-1.7b.json \
    --n-states 60 $GPU --out reports/speed_classical.json
python -m obsassist.learn.speed --test data/decisions/queue_test.jsonl --heads heads/qwen3-1.7b-queue.json \
    --n-states 60 $GPU --out reports/speed_queue.json

#!/bin/bash
#SBATCH --job-name=lbt_jev_replay
#SBATCH --account=PAS2823
#SBATCH --partition=gpudebug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PAS2823/saadsm/private_lbt/work/jev_%j.log
# The log holds only case ids, picks and times, but it lives in the private directory anyway.
set -e
umask 077
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 26-option menus fragment the cache on a 16 GB V100
source /fs/scratch/PAS2823/saadsm/SkyJev/pitzer_env.sh
cd /fs/scratch/PAS2823/saadsm/SkyJev
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python lbt_replay/run_jev.py --private /fs/scratch/PAS2823/saadsm/private_lbt \
    --out lbt_replay/lbt_jev_replay${OUT_SUFFIX}.json --model ${MODEL:-Qwen/Qwen3-1.7B} --device cuda --dtype float16 ${EXTRA:-}

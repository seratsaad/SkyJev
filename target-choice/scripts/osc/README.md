# Running the heavy sweeps on OSC

The oracle search and the dataset builder are CPU-only, embarrassingly parallel and seeded, so
they split into Slurm job arrays. Nothing here logs in for you: copy the project to OSC, create
the environment once, and submit.

```bash
# on an OSC login node
git clone <your fork> obsassist && cd obsassist
module load miniconda3
conda create -n obsassist python=3.11 -y && conda activate obsassist
pip install -e .                   # numpy, scipy, astropy, pyyaml, ... (no torch needed)
sbatch --account=<PROJECT> scripts/osc/oracle_array.slurm      # hindsight-optimal nights
sbatch --account=<PROJECT> scripts/osc/dataset_array.slurm     # 10,000 labelled nights
```

Fitting the AnyJev heads needs the LLM's hidden states: run `python -m obsassist.learn.fit`
on a GPU node (`--partition=gpu --gpus-per-node=1`, `--model Qwen/Qwen3-4B` or larger) or on
the Mac. The heads JSON is small (~100 KB per question) and portable across machines for the
same model.

## Queue nights and the speed benchmark (Pitzer, account PAS2823)

Queue datasets are CPU-only; run them in a batch or interactive CPU job (`sbatch --account=PAS2823 ...`):

```bash
python -m pytest -q                                   # the test suite, no GPU
python -m obsassist.learn.dataset --mode queue --nights 400 --seed0 0 --workers 8 \
    --out data/decisions/queue_train.jsonl
python -m obsassist.learn.dataset --mode queue --nights 120 --seed0 100000 --workers 8 \
    --out data/decisions/queue_test.jsonl
python -m obsassist.learn.evaluate --mode queue --policies list,greedy,queue --nights 40 \
    --out reports/nights_queue.json
```

Head fitting and `speed.py` need a GPU node. Pitzer's V100s have no bfloat16, so pass `--dtype float16`:

```bash
salloc --account=PAS2823 --cluster=pitzer --partition=gpu --gpus-per-node=1 --time=04:00:00
python -m obsassist.learn.fit --train data/decisions/queue_train.jsonl --test data/decisions/queue_test.jsonl \
    --out heads/qwen3-1.7b-queue.json --report reports/fit_qwen3-1.7b-queue.json \
    --l0-states 120 --device cuda --dtype float16
python -m obsassist.learn.speed --test data/decisions/test_decisions.jsonl --heads heads/qwen3-1.7b.json \
    --device cuda --dtype float16 --out reports/speed_classical.json
python -m obsassist.learn.speed --test data/decisions/queue_test.jsonl --heads heads/qwen3-1.7b-queue.json \
    --device cuda --dtype float16 --out reports/speed_queue.json
```

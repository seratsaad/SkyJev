#!/bin/bash
# Regenerate every number in the README's Results: datasets in data/decisions/, heads in heads/,
# reports in reports/. About 1.5 h on an Apple M-series Mac; the head fit needs the [decide]
# extra and downloads Qwen3-1.7B once. Everything is seeded.
#     scripts/reproduce.sh            (PY=/path/to/python to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

$PY -m obsassist.learn.dataset --nights 600 --out data/decisions/train_decisions.jsonl
$PY -m obsassist.learn.dataset --nights 150 --seed0 100000 --out data/decisions/test_decisions.jsonl

# the head fit runs on the GPU while the model-free benchmarks use the CPU
$PY -m obsassist.learn.fit --l0-states 120 &
FIT=$!
$PY -m obsassist.learn.baselines
$PY -m obsassist.learn.evaluate --policies list,greedy,lookahead --nights 34 --projection-bias \
    --out reports/nights_modelfree.json
$PY -m obsassist.learn.tune
wait $FIT

$PY -m pytest -q

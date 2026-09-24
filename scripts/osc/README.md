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

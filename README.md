# SkyJev

Code, results and manuscript for the research note *SkyJev: Fast Menu Choices and a Language Model
for Telescope Observing* (Saad & Ting). SkyJev splits the decisions an observer makes during a night
between a **Jev choice** and a **writing LLM**. For a call with a fixed set of answers, code builds a
short menu from the measurements and a small open model gives a probability to each option in one
forward pass, without writing text. A gate, like a detection threshold, lets SkyJev act only when the
probability is high and hands the other cases to an LLM with tools or to the observer.

A Jev choice takes about 0.1 s. On 300 archival focus sequences that we had not used before, SkyJev
acted on 69% of the focus decisions and agreed with a fit to the full sequence in 97% of them. The
demo video is on the [releases page](https://github.com/seratsaad/SkyJev/releases).

## Contents

| Folder | Contents | Author |
|---|---|---|
| [`note/`](note/) | The manuscript (AASTeX 7), its TikZ figure, and the scripts that print every number from the saved results | both |
| [`routine/`](routine/) | Routine decisions with a fitted head and a confidence gate (focus, slit angle, weather, logging, next target), the held-out focus test, and comparisons with TypeSafe's hosted Jev and CLM. See [routine/README.md](routine/README.md) | Serat Saad |
| [`focus/`](focus/) | Measurement of archival ESO/MPG 2.2 m WFI through-focus sequences, the original 30 and a held-out set of 300 | Serat Saad |
| [`target-choice/`](target-choice/) | Simulated classical and queue nights at Keck and Magellan, an exposure time calculator, a hindsight oracle, a greedy planner and the observing assistant (the `obsassist` package, originally `astrojev`) | Yuan-Sen Ting |
| [`browser/`](browser/) | A Jev browser agent against Browser Use on astronomy websites (SIMBAD, VizieR, NED, ADS, ESO archives, an exposure time calculator), each task with a deterministic check | Serat Saad |
| [`lbt_replay/`](lbt_replay/) | Routine decisions from real LBT nights replayed with Jev choices (aggregate numbers only, no names or targets) | Serat Saad |
| [`demo/`](demo/) | The script that records the demo video | Serat Saad |

## Run it

The decision code runs on a laptop (Apple GPU or CPU) or on a cluster GPU. The local models are open
Qwen3 models from Hugging Face. Nothing needs a paid API except the optional writing LLM and the
browser agent.

```bash
# from the repository root
pip install -e "target-choice[all]"                                # obsassist, AnyJev, torch, transformers
python -m routine.focus --model Qwen/Qwen3-1.7B --out routine/reports/focus_qwen3-1.7b.json
latexmk -pdf -cd note/main.tex                                     # the note
```

* Target choice. See [target-choice/README.md](target-choice/README.md). `scripts/reproduce.sh` reruns
  the datasets, heads and controls. `scripts/osc/` has the Slurm scripts we used on OSC Pitzer.
* Focus. `focus/download.sh` fetches the original 30 frames and `focus/fetch_heldout.py` the 300
  held-out ones (public, from the ESO archive). The held-out plan was fixed before any frame was
  measured, see [routine/README.md](routine/README.md).
* Browser agent. See [browser/README.md](browser/README.md). It needs jev-ultrafast checked out at
  `SkyJev/jev-ultrafast`, Chrome, and OpenAI credit.

## Credits and license

`target-choice/` keeps its original commit history. SkyJev uses
[AnyJev](https://github.com/nokia-applied-research/AnyJev) (Zhang, Yang, Shi & Wu 2026) and
[Jev Ultrafast](https://github.com/browser-use/jev-ultrafast). The focus tests use data obtained from
the ESO Science Archive Facility (program 60.A-9120(A)).

Apache-2.0, see [LICENSE](LICENSE).

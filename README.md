# SkyJev

Code, results and manuscript for the research note *Which Observing Decisions Should a Language
Model Make?* (Saad & Ting). The note tests whether large language models can make two decisions
that an observer makes during a night, the choice of the next target and the telescope focus.

| Folder | Contents | Author |
|---|---|---|
| [`target-choice/`](target-choice/) | Simulated nights at Keck and Magellan, an exposure time calculator, a hindsight oracle, a greedy planner and the AnyJev/Qwen3 decision heads (the `obsassist` package, originally `astrojev`) | Yuan-Sen Ting |
| [`focus/`](focus/) | LLM focus decisions on 30 archival through-focus sequences from the ESO/MPG 2.2 m Wide Field Imager | Serat Saad |
| [`note/`](note/) | The manuscript (AASTeX 7) and `stats.py`, which prints every number in it from the saved results | both |

## Reproduce the numbers in the note

```bash
python3 note/stats.py          # numpy only, reads target-choice/reports/ and focus/results/
```

To regenerate the saved results themselves:

* Target choice. See [target-choice/README.md](target-choice/README.md). `scripts/reproduce.sh`
  in that folder reruns the datasets, the AnyJev heads and the controls (about 1.5 h on an
  M-series Mac, no API key).
* Focus. See [focus/README.md](focus/README.md). `focus/download.sh` fetches the 30 public frames
  (about 240 MB) from the ESO archive. The model calls need `OPENAI_API_KEY` and cost about US$0.35.

## Build the note

```bash
cd note && latexmk -pdf main.tex
```

## Credits and license

`target-choice/` keeps its original commit history. It uses
[AnyJev](https://github.com/nokia-applied-research/AnyJev) (Zhang, Yang, Shi & Wu 2026,
Apache-2.0). The focus test uses data obtained from the ESO Science Archive Facility (program
60.A-9120(A)).

Apache-2.0, see [LICENSE](LICENSE).

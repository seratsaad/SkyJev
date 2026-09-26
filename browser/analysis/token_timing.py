"""Why generation is slow and choosing is fast, measured on this laptop.

One forward pass reads a whole prompt and returns a probability for every possible next token. Writing
an answer repeats that pass once per token. Choosing from a menu needs one pass. This times both with
the same small local model (Qwen3-0.6B, CPU), so the ratio on the slide is measured, not asserted.

  uv run python analysis/token_timing.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = (
    "You are controlling a web browser on the arXiv advanced search page. The goal is to find papers "
    "whose author is Serat Saad. The page offers these controls: [1] link Learn more, [2] button Dismiss "
    "announcement, [3] link archive home, [4] combobox Field to search, [5] textbox Search term, "
    "[6] button Search, [7] button Clear. Which control should be used next, and why?\nAnswer:"
)
OUT = Path(__file__).resolve().parents[1] / "results" / "summary" / "token_timing.json"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    ids = tok(PROMPT, return_tensors="pt")
    n_prompt = ids["input_ids"].shape[1]

    with torch.no_grad():
        model(**ids)  # warm-up
        timings = []
        for _ in range(5):
            t = time.perf_counter()
            out = model(**ids)
            timings.append(time.perf_counter() - t)
        one_pass = sorted(timings)[2]
        probs = torch.softmax(out.logits[0, -1], dim=-1)
        top = torch.topk(probs, 6)
        top_tokens = [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices, strict=True)]

        # Choosing: one pass, read the probability of each option's number.
        # The prompt ends in "[" so the very next token is the digit itself. (Reading " 1" instead picks
        # up the shared space token, which gives every option the same probability.)
        options = [str(i) for i in range(1, 8)]
        option_ids = [tok.encode(o, add_special_tokens=False)[-1] for o in options]
        assert len(set(option_ids)) == len(options), "option tokens must be distinct"
        t = time.perf_counter()
        out = model(**tok(PROMPT + " The next control to use is [", return_tensors="pt"))
        choose_s = time.perf_counter() - t
        p = torch.softmax(out.logits[0, -1, option_ids], dim=-1)
        choice = {o: round(float(x), 3) for o, x in zip(options, p, strict=True)}

        # Writing: the same model generates an explanation, one token per pass.
        timings = {}
        for n in (10, 50, 200):
            t = time.perf_counter()
            gen = model.generate(**ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
            timings[n] = time.perf_counter() - t
        sample = tok.decode(gen[0, n_prompt:n_prompt + 50], skip_special_tokens=True)

    result = {
        "model": MODEL, "device": "cpu (Apple M4)", "prompt_tokens": n_prompt,
        "one_forward_pass_s": one_pass, "top_next_tokens": top_tokens,
        "choose_one_pass_s": choose_s, "choice_probabilities": choice,
        "generate_s": {str(k): v for k, v in timings.items()},
        "per_generated_token_s": (timings[200] - timings[10]) / 190,
        "sample_generation": sample,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

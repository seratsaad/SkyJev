"""Write a compact copy of every run for shipping.

A single VizieR run saves about 2 MB, almost all of it repeated element tables. The notebook, the
deck and the summary tables never need more than the first and last table, so the shipped copy keeps
those and drops the rest. Nothing else changes: the metrics, the history and the verification output
are untouched, so every number in the talk can still be recomputed from what ships.

  python analysis/compact_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SHIPPED = RESULTS / "shipped"


def compact(record: dict) -> dict:
    tables = record.get("element_tables") or []
    if len(tables) > 2:
        record["element_tables"] = [tables[0], tables[-1]]
        record["element_tables_note"] = (
            f"first and last of {len(tables)} observations; the full traces stay on the machine "
            "that made the runs"
        )
    return record


def main():
    SHIPPED.mkdir(parents=True, exist_ok=True)
    before = after = 0
    count = 0
    for path in sorted(RESULTS.glob("*/*.json")):
        if path.parent.name in {"summary", "shipped"} or path.parent.name.startswith("_"):
            continue
        raw = path.read_text()
        before += len(raw)
        out = SHIPPED / f"{path.parent.name}__{path.name}"
        text = json.dumps(compact(json.loads(raw)), indent=1, default=str)
        out.write_text(text)
        after += len(text)
        count += 1
    print(f"{count} records  {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB  in {SHIPPED}")


if __name__ == "__main__":
    main()

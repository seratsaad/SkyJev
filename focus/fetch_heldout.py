"""Select and download a held-out set of WFI through-focus frames from the ESO archive.

Same selection as the original 30 (download_list.txt): template WFI_cal_FocusSeq, R band
(BB#Rc/162_ESO844), public, one frame per night. The 30 already used are excluded, and so is every
other frame from their nights. Nights are drawn at random with a fixed seed, so the list is
reproducible, and the list is written before any frame is measured.

    python focus/fetch_heldout.py --n 300 --out focus/data_heldout      # on the cluster, not a laptop
"""

import argparse
import csv
import io
import random
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAP = "https://archive.eso.org/tap_obs/sync"
QUERY = """SELECT dp_id, date_obs, filter_path, exposure, access_estsize, release_date
FROM dbo.raw
WHERE instrument = 'WFI' AND tpl_id = 'WFI_cal_FocusSeq' AND filter_path LIKE '%Rc/162%'
AND release_date < '2026-09-28'"""


def query():
    url = TAP + "?" + urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                              "MAXREC": 200000, "QUERY": QUERY})
    with urllib.request.urlopen(url, timeout=300) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


def night(date_obs: str) -> str:
    """The observing night: dates before noon UT belong to the previous evening at La Silla."""
    from datetime import datetime, timedelta

    t = datetime.fromisoformat(date_obs[:19])
    return (t - timedelta(hours=12)).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=str(HERE / "data_heldout"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    used = {l.strip() for l in (HERE / "download_list.txt").read_text().split() if l.strip()}
    used_nights = {night(u[len("WFI."):]) for u in used}  # ids keep the colons: WFI.2026-07-13T01:02:28.886
    rows = query()
    by_night = {}
    for r in rows:
        if r["dp_id"] in used or night(r["date_obs"]) in used_nights:
            continue
        by_night.setdefault(night(r["date_obs"]), []).append(r)
    rng = random.Random(a.seed)
    nights = sorted(by_night)
    pick = sorted(rng.sample(nights, min(a.n, len(nights))))
    chosen = [sorted(by_night[n], key=lambda r: r["date_obs"])[0] for n in pick]
    lst = HERE / "download_list_heldout.txt"
    lst.write_text("\n".join(r["dp_id"] for r in chosen) + "\n")
    print(f"{len(rows)} frames in the archive, {len(by_night)} unused nights, chose {len(chosen)} "
          f"({chosen[0]['date_obs'][:10]} to {chosen[-1]['date_obs'][:10]}); list in {lst.name}")
    if a.list_only:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        f = out / (r["dp_id"].replace(":", "-") + ".fits")
        if f.exists() and f.stat().st_size > 0:
            continue
        z = f.with_suffix(".fits.Z")
        try:
            urllib.request.urlretrieve("https://dataportal.eso.org/dataPortal/file/" + r["dp_id"], z)
            subprocess.run(["gzip", "-df", str(z)], check=True)  # .Z (compress) files; gzip reads them
            print("ok", r["dp_id"], flush=True)
        except Exception as e:  # a failed frame is skipped and reported, not retried silently
            print("FAIL", r["dp_id"], type(e).__name__, e, flush=True)
            z.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

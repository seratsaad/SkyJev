"""What JEV's observation layer sees on each site. No model calls, no API key.

This is the cheapest and most honest way to show the architecture: before any model is involved,
the page has already been reduced to a list of indexed controls, and whatever is missing from that
list the agent cannot choose. Writes results/site_survey.csv.

  python analysis/site_survey.py [--settle 1.5]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astroweb.jev_runner import ENV_PROBE

SITES = {
    "SIMBAD basic search": "https://simbad.cds.unistra.fr/simbad/",
    "SIMBAD by identifier": "https://simbad.cds.unistra.fr/simbad/sim-fid",
    "arXiv": "https://arxiv.org/",
    "ADS": "https://ui.adsabs.harvard.edu/",
    "VizieR": "https://vizier.cds.unistra.fr/viz-bin/VizieR",
    "NED": "https://ned.ipac.caltech.edu/",
    "ESO raw archive form": "https://archive.eso.org/eso/eso_archive_main.html",
    "ESO Science Portal": "https://archive.eso.org/scienceportal/home",
    "MAST Portal": "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html",
    "IRSA": "https://irsa.ipac.caltech.edu/frontpage/",
    "Gaia Archive": "https://gea.esac.esa.int/archive/",
    "Aladin Lite": "https://aladin.cds.unistra.fr/AladinLite/",
    "HST ETC (ACS)": "https://etc.stsci.edu/etc/input/acs/imaging/",
    "ESO ETC (FORS)": "https://etc.eso.org/observing/etc/fors",
    "JWST docs": "https://jwst-docs.stsci.edu/",
    "Exoplanet Archive": "https://exoplanetarchive.ipac.caltech.edu/",
    "Keck KOA": "https://koa.ipac.caltech.edu/cgi-bin/KOA/nph-KOAlogin",
    "LBTO LBC": "https://scienceops.lbto.org/lbc/",
}


def survey(settle: float) -> list[dict]:
    from jev_ultrafast.browser import Browser

    rows = []
    for site, url in SITES.items():
        row = {"site": site, "url": url}
        started = time.perf_counter()
        try:
            browser = Browser(url)
            time.sleep(settle)
            page = browser.observe(screenshot=False)
            probe = browser.evaluate(ENV_PROBE) or {}
            actions = page["actions"]
            # A control whose only name is its own role is one the model must choose blind.
            unnamed = sum(1 for a in actions if a.get("role") and a["label"] in (a["role"], "Open " + a["role"]))
            # Two controls with the same role and the same name are indistinguishable in the table.
            # The model is then choosing by position in a list, which is not information about the page.
            names = Counter((a.get("role"), a["label"]) for a in actions if a["kind"] in {"click", "fill", "select"})
            ambiguous = sum(count for name, count in names.items() if count > 1)
            row.update(
                observe_ms=round((time.perf_counter() - started) * 1000),
                actions=len(actions),
                omitted=page["omitted_actions"],
                unnamed=unnamed,
                ambiguous=ambiguous,
                fillable=sum(1 for a in actions if a["kind"] == "fill"),
                selectable=sum(1 for a in actions if a["kind"] == "select"),
                text_chars=len(page["text"]),
                iframes=probe.get("iframes", 0),
                canvas=probe.get("canvas", 0),
                shadow=probe.get("shadow", 0),
                error="",
            )
            browser.close()
        except Exception as e:  # noqa: BLE001
            row.update(actions=0, unnamed=0, ambiguous=0, iframes=0, canvas=0, shadow=0,
                       error=f"{type(e).__name__}: {e}"[:120])
        print(f"{site:24} {row.get('actions', 0):>4} elements  "
              f"unnamed={row.get('unnamed', 0):>3}  ambiguous={row.get('ambiguous', 0):>3}  "
              f"iframes={row.get('iframes', 0)}  "
              f"canvas={row.get('canvas', 0)} {row.get('error', '')}", flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle", type=float, default=1.5)
    args = parser.parse_args()
    rows = survey(args.settle)
    path = ROOT / "results" / "site_survey.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site", "url", "observe_ms", "actions", "omitted", "unnamed", "ambiguous", "fillable", "selectable",
              "text_chars", "iframes", "canvas", "shadow", "error"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

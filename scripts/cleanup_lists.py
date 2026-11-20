#!/usr/bin/env python
"""List regenerable intermediates of the CarbonGlobe project that are no longer needed (nothing is deleted here).

Groups (one list file per group in work/logs/cleanup/<group>.txt, paths relative to the project root):
  ornl_tifs     ORNL rasters of the other ESMs (EC-Earth3, GISS, MPI: only used to identify GFDL-ESM4) and the GFDL climate-only case (decomposition
                stored in cells_full_extra.npz); the GFDL CLIM_CO2 + REF rasters stay until the competition ends
  results       results/*.csv that are not scored on the board, not named in any script and older than --days (default 1.5)
  work_part     work/part_*.npy (cell labels of a build, 19 MB each) older than --part-hours
  work_dir      work/dir_*.npy (file minus C) of files that are neither scored nor named in a script, older than --days
  work_dirx     work/dirx_<prefix>_*.npy part dumps of prefixes other than the ones given by --keep-dirx (any age)
  parked        work/parked_results/* and the stale design cache results/td_*.csv

    python scripts/cleanup_lists.py [--days 1.5] [--keep-dirx d48,d48c,d48y,d48m]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1.5)
    ap.add_argument("--keep-dirx", default="d48,d48c,d48y,d48m,d48p")
    ap.add_argument("--part-hours", type=float, default=3.0, help="work/part_*.npy older than this many hours are listed (every build rewrites its own)")
    args = ap.parse_args()
    now = time.time(); old = lambda p: (now - p.stat().st_mtime) / 86400 > args.days
    out = ROOT / "work/logs/cleanup"; out.mkdir(parents=True, exist_ok=True)
    scored = {k for k, v in json.load(open(ROOT / "results/board_scores.json")).items() if v is not None}
    # every results/<name>.csv and work/dir_<name>.npy that a script, source file or spec mentions stays
    text = ""
    for pat in ("scripts/*.py", "scripts/*.sh", "scripts/*/*.py", "scripts/*/*.sh", "src/cgb/*.py", "work/logs/*_spec.json", "work/logs/*_manifest.txt", "work/logs/*_design.json", "work/spat_centers.json"):
        for f in ROOT.glob(pat):
            try:
                text += f.read_text(errors="ignore") + "\n"
            except OSError:
                pass
    named_csv = set(re.findall(r"([A-Za-z0-9_.\-]+\.csv)", text)); named_npy = set(re.findall(r"([A-Za-z0-9_.\-]+\.npy)", text))
    groups: dict[str, list[Path]] = {k: [] for k in ("ornl_tifs", "results", "work_part", "work_dir", "work_dirx", "parked")}
    for f in sorted((ROOT / "work/ornl2522").glob("*.tif")):
        if any(e in f.name for e in ("EC-Earth3", "GISS-E2-1-G", "MPI-ESM1.2-HR")) or re.search(r"GFDL-ESM4_SSP\d+_CLIM_(existing|new)_forest", f.name):
            groups["ornl_tifs"].append(f)
    for f in sorted((ROOT / "results").glob("*.csv")):
        if f.name.startswith("td_"):
            groups["parked"].append(f)
        elif f.name not in scored and f.name not in named_csv and old(f):
            groups["results"].append(f)
    for f in sorted((ROOT / "work/parked_results").glob("*")):
        if f.is_file():
            groups["parked"].append(f)
    scored_stems = {Path(k).stem for k in scored}
    for f in sorted((ROOT / "work").glob("part_*.npy")):
        if (now - f.stat().st_mtime) / 3600 > args.part_hours:
            groups["work_part"].append(f)
    for f in sorted((ROOT / "work").glob("dir_*.npy")):
        stem = f.stem[4:]
        if stem not in scored_stems and f.name not in named_npy and old(f):
            groups["work_dir"].append(f)
    keep = tuple(f"dirx_{p}_" for p in args.keep_dirx.split(",") if p)
    for f in sorted((ROOT / "work").glob("dirx_*.npy")):
        if not f.name.startswith(keep) and f.name not in named_npy:          # dumps of abandoned variants: any age (the builder rewrites them in 30 s)
            groups["work_dirx"].append(f)
    total = 0
    for k, fs in groups.items():
        size = sum(f.stat().st_size for f in fs); total += size
        (out / f"{k}.txt").write_text("".join(str(f.relative_to(ROOT)) + "\n" for f in fs))
        ex = ", ".join(f.name for f in fs[:3])
        print(f"{k:10s} {len(fs):5d} files {size / 1e9:6.2f} GB   e.g. {ex}")
    print(f"total {total / 1e9:.2f} GB; lists in {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

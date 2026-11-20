#!/usr/bin/env python
"""Run a configuration variant (a run_site config plus keyword overrides) on a list of TEST sites into its own root.

    python scripts/ed/test_variant.py --config c400 --set co2=asis \
        --root work/edv3/test_c400asis --sites-from work/edv3/cal_c400/site_order.txt --first 16 --workers 2
Overrides are key=value pairs (ints, floats, true/false and strings are parsed); results land in <root>/results like
run_many's, so scripts/ed_transfer.py can read them.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed"))
sys.path.insert(0, str(ROOT / "src"))

import run_site as rs  # noqa: E402


def parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def _one(site: int, root: str, config: str, over: dict) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        info = rs.run_site(site, Path(root), rs.config_args(config, **over), keep=False)
        t = info["timings_s"]
        return site, f"ok spinup {t['spinup']:.0f}s windows {t['w126'] + t['w585']:.0f}s", time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return site, f"FAIL {exc!r}", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="c400", choices=sorted(rs.CONFIGS))
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    ap.add_argument("--root", required=True)
    ap.add_argument("--sites", default="")
    ap.add_argument("--sites-from", default="")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    over = {k: parse_value(v) for k, v in (s.split("=", 1) for s in args.set)}
    sites = [int(s) for s in args.sites.split(",") if s] if args.sites else []
    if args.sites_from:
        sites += [int(l) for l in open(args.sites_from) if l.strip()]
    if args.first:
        sites = sites[: args.first]
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    todo = [s for s in dict.fromkeys(sites) if not rs.results_exist(s, root)]
    print(f"config {args.config} {over}: {len(todo)} test sites with {args.workers} workers; root {root}", flush=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, s, str(root), args.config, over): s for s in todo}
        for f in as_completed(futs):
            site, status, dt = f.result(); done += 1
            print(f"[{done}/{len(todo)} {(time.time() - t0) / 60:.1f} min] site {site}: {status} ({dt:.0f} s wall)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

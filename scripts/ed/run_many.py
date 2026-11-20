#!/usr/bin/env python
"""Run scripts/ed/run_site.py for many sites in parallel (one ED process per worker), resumable.

Site order (--order lead, the default when no --sites/--site-file is given):
  (i) 120 random sites, seed 42, drawn from all sites excluding the collapse cluster
      (work/collapse_cluster_labels.npy == 0), (ii) work/ed_site_priority.csv by rank, (iii) the rest.
Sites whose results already exist in <root>/results are skipped. One timing line per site goes to stdout.

  nohup caffeinate -i python scripts/ed/run_many.py --config b --root work/edv3/full --workers 10 > work/edv3/full_run.log 2>&1 &
  python scripts/ed/run_many.py --config c400 --root work/edv3/cal_c400 --order random --limit 120 --workers 10
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed"))
import make_site_inputs as msi  # noqa: E402
import run_site as rs  # noqa: E402


def lead_order(n_random: int = 120, seed: int = 42) -> tuple[list[int], list[int]]:
    cl = np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0
    rng = np.random.default_rng(seed)
    rand = sorted(int(s) for s in rng.choice(np.where(~cl)[0], n_random, replace=False))
    pri_file = ROOT / "work" / "ed_site_priority.csv"
    pri: list[int] = []
    if pri_file.exists():
        import pandas as pd
        p = pd.read_csv(pri_file).sort_values("rank")
        pri = [int(s) for s in p["site"]]
    seen = set(rand)
    order = list(rand)
    for s in pri + list(range(6171)):
        if s not in seen:
            seen.add(s); order.append(s)
    return rand, order


def _one(site: int, root: str, config: str) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        info = rs.run_site(site, Path(root), rs.config_args(config), keep=False)
        t = info["timings_s"]
        return site, f"ok spinup {t['spinup']:.0f}s windows {t['w126'] + t['w585']:.0f}s", time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return site, f"FAIL {exc!r}", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="b", choices=sorted(rs.CONFIGS))
    ap.add_argument("--root", default=str(rs.EDV3 / "full"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--order", default="lead", choices=["lead", "random"], help="lead = random 120 + priority + rest; random = the 120 random sites only")
    ap.add_argument("--sites", default="", help="explicit a-b ranges / comma list (overrides --order)")
    ap.add_argument("--site-file", default="")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many sites (after skipping finished ones)")
    args = ap.parse_args()
    sites: list[int] = []
    if args.sites or args.site_file:
        for part in filter(None, args.sites.split(",")):
            if "-" in part:
                a, b = part.split("-"); sites.extend(range(int(a), int(b) + 1))
            else:
                sites.append(int(part))
        if args.site_file:
            sites.extend(int(l.strip()) for l in open(args.site_file) if l.strip())
    else:
        rand, order = lead_order()
        sites = rand if args.order == "random" else order
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    (root / "site_order.txt").write_text("\n".join(str(s) for s in sites) + "\n")
    todo = [s for s in dict.fromkeys(sites) if not rs.results_exist(s, root)]
    if args.limit:
        todo = todo[: args.limit]
    msi.rsds_channel_floors()
    print(f"config {args.config}: {len(todo)} sites to run with {args.workers} workers; root {root}", flush=True)
    t0 = time.time(); done = 0; fails = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(root / "failed.txt", "a") as failed:
        futs = {ex.submit(_one, s, str(root), args.config): s for s in todo}
        for f in as_completed(futs):
            site, status, dt = f.result(); done += 1
            if not status.startswith("ok"):
                fails += 1; failed.write(f"{site} {status}\n"); failed.flush()
            el = time.time() - t0
            print(f"[{done}/{len(todo)} {el / 60:.1f} min, eta {(len(todo) - done) * el / done / 3600:.2f} h, fails {fails}] site {site}: {status} ({dt:.0f} s wall)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

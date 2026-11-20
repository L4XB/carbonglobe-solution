#!/usr/bin/env python
"""Run the test pipeline's ED configuration on TRAINING sites and keep the organisers' trajectories next to it.

The training forcing (``glob_X_fea``) is clean (night rsds ~0, no negative hus, ta_h mean = ta_m) and its CO2
channels hold real ppm, so it is written as delivered (``repair=False``, ``co2="asis"``); the spin-up is the
test configuration's (first-10-year climatology, CO2 fixed at 400 ppm, fire on unless ``--config`` says
otherwise). Both window scenarios get the same training forcing, so ``<site>_ssp126.csv`` and
``<site>_ssp585.csv`` are two runs of identical inputs: their difference measures ED's run-to-run noise.

Outputs (root, default ``work/edv3/train_c400f``): the usual ``results/`` and ``csv/`` files of ``run_site``
plus ``truth/<site>.csv`` with the organisers' state at the window start (year 0) and at year 40 (December)
for each seed age (raw units: m, kgC/m2).

    python scripts/ed/train_site.py --n 120 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed"))
sys.path.insert(0, str(ROOT / "src"))

import make_site_inputs as msi  # noqa: E402
import run_site as rs  # noqa: E402
from cgb import io  # noqa: E402

SEED_AGES = list(msi.SEED_AGES)


def train_y0_raw(site: int) -> np.ndarray:
    """Organisers' state at the window start for the 15 seed ages, raw units, shape (15, 7)."""
    return np.stack([np.asarray(io.open_train_y(a)[site, 0, 11, :], dtype=np.float64) for a in SEED_AGES])


def train_y40_raw(site: int) -> np.ndarray:
    """Organisers' December state of window year 40 for the 15 seed ages, raw units, shape (15, 7)."""
    return np.stack([np.asarray(io.open_train_y(a)[site, 40, 11, :], dtype=np.float64) for a in SEED_AGES])


class _EncodedY0:
    """``test_y0``-like view: the training y0 in the release's encoding, so build_site's decode returns raw units."""

    def __getitem__(self, site: int) -> np.ndarray:
        return train_y0_raw(int(site)) / msi.DECODE_MULT + msi.FLOORS


def _fake_open_test(scenario: str) -> tuple[np.ndarray, _EncodedY0]:
    return io.open_train_x(), _EncodedY0()


def _one(site: int, root: str, config: str, spinup_start: int | None = None) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        msi.io.open_test = _fake_open_test          # this worker process serves training data only
        over = dict(co2="asis", repair=False)
        if spinup_start is not None:
            over["spinup_start"] = spinup_start     # first window-year index of the spin-up climatology (30 = 2011)
        args = rs.config_args(config, **over)
        info = rs.run_site(site, Path(root), args, keep=False)
        y0, y40 = train_y0_raw(site), train_y40_raw(site)
        tdir = Path(root) / "truth"; tdir.mkdir(exist_ok=True)
        with open(tdir / f"{site}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["age"] + [f"y0_{n}" for n in io.STATE_NAMES] + [f"y40_{n}" for n in io.STATE_NAMES])
            for i, a in enumerate(SEED_AGES):
                w.writerow([a] + [f"{v:.6g}" for v in y0[i]] + [f"{v:.6g}" for v in y40[i]])
        t = info["timings_s"]
        return site, f"ok spinup {t['spinup']:.0f}s windows {t['w126'] + t['w585']:.0f}s", time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return site, f"FAIL {exc!r}", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="c400f", choices=sorted(rs.CONFIGS))
    ap.add_argument("--root", default=str(ROOT / "work" / "edv3" / "train_c400f"))
    ap.add_argument("--n", type=int, default=120, help="number of random training sites (seed 42)")
    ap.add_argument("--sites", default="", help="explicit comma list (overrides --n)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--spinup-start", type=int, default=None, help="override the spin-up climatology start year index")
    ap.add_argument("--first", type=int, default=0, help="run only the first N sites of the random list")
    args = ap.parse_args()
    if args.sites:
        sites = [int(s) for s in args.sites.split(",") if s]
    else:
        rng = np.random.default_rng(42)
        sites = sorted(int(s) for s in rng.choice(io.open_train_x().shape[0], args.n, replace=False))
        if args.first:
            sites = sites[: args.first]
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    (root / "site_order.txt").write_text("\n".join(map(str, sites)) + "\n")
    todo = [s for s in sites if not (rs.results_exist(s, root) and (root / "truth" / f"{s}.csv").exists())]
    msi.rsds_channel_floors()
    print(f"config {args.config} on training sites: {len(todo)} to run with {args.workers} workers; root {root}", flush=True)
    t0 = time.time(); done = fails = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, s, str(root), args.config, args.spinup_start): s for s in todo}
        for f in as_completed(futs):
            site, status, dt = f.result(); done += 1
            fails += not status.startswith("ok")
            el = time.time() - t0
            print(f"[{done}/{len(todo)} {el / 60:.1f} min, eta {(len(todo) - done) * el / done / 3600:.2f} h, fails {fails}] "
                  f"train site {site}: {status} ({dt:.0f} s wall)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

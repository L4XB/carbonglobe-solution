#!/usr/bin/env python
"""Does the latent-state model already capture the collapses, or are they its residual?

``work/models/latent_v1/pred_public.npy`` holds the sibling model's year-40
predictions on the public fold, ``(15, 3086, 5, 2)`` in scaled units. This script
reads them (it changes nothing) and splits the old-age agb error into event rows
and the rest, the way ``mort_census.py`` split the persistence loss. If the model's
remaining old-age loss sits on event rows in the same 84 % proportion as
persistence's, the events are what is left to model; if the model already places
them, they are not.

    ./.venv/bin/python scripts/mort_residuals.py [--tag v1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    model_dir = io.ROOT / "work" / "models" / f"latent_{args.tag}"
    pred = np.load(model_dir / "pred_public.npy")  # (15, n_pub, 5, 2)
    res = json.loads((model_dir / "results.json").read_text())
    reported = next(r for r in res["folds"]["public"]["rows"] if r["name"] == f"latent_{args.tag}")

    split = holdout.make_split(seed=42)
    pub = split.public
    div = cache.scoring_divisors()
    traj = cache.open_traj()
    truth = np.asarray(traj[:, pub, 40, :2], dtype=np.float64) / div  # (15, n_pub, 2)
    y0 = np.asarray(traj[:, pub, 0, :2], dtype=np.float64) / div
    p40 = pred[:, :, -1, :].astype(np.float64)
    score = float(((p40 - truth) ** 2).mean())
    print(f"recomputed public score {score:.5f} against reported {reported['score']:.5f} "
          f"-> {'aligned' if abs(score - reported['score']) < 1e-4 else 'MISALIGNED, stop'}")
    if abs(score - reported["score"]) > 1e-4:
        raise SystemExit(1)

    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    # two advanced indices separated by a slice would move the site axis to the front;
    # index in two steps so the layout stays (8, n_pub, 41)
    agb = np.asarray(traj[old][:, pub], dtype=np.float32)[:, :, :, 1]  # (8, n_pub, 41)
    ev = mort.events(agb)  # (8, n_pub, 40)
    has_ev = ev.any(axis=2)
    biggest = np.where(ev, agb[:, :, :-1] - agb[:, :, 1:], 0.0).argmax(axis=2)
    err_lat = (p40[old][..., 1] - truth[old][..., 1]) ** 2  # (8, n_pub) agb only
    err_per = (y0[old][..., 1] - truth[old][..., 1]) ** 2
    resid = p40[old][..., 1] - truth[old][..., 1]

    print("\n== old seed ages (>= 150), public fold, agb column, scaled units ==")
    for name, e in (("persistence", err_per), ("latent", err_lat)):
        tot = e.sum()
        print(f"  {name:12s} mean {e.mean():.5f}; rows with an event ({has_ev.mean():.3f} of rows) carry {e[has_ev].sum() / tot:.3f} of the loss; "
              f"mean on event rows {e[has_ev].mean():.4f}, on the rest {e[~has_ev].mean():.4f}")
    print(f"  latent residual (pred - truth) on event rows: mean {resid[has_ev].mean():+.4f}, median {np.median(resid[has_ev]):+.4f}, "
          f"on the rest {resid[~has_ev].mean():+.4f}  (positive = the model predicts too much biomass, i.e. misses the loss)")
    print(f"  latent / persistence loss ratio: event rows {err_lat[has_ev].sum() / err_per[has_ev].sum():.3f}, other rows {err_lat[~has_ev].sum() / err_per[~has_ev].sum():.3f}")

    print("\n  by the forcing year of the row's largest event:")
    print(f"  {'years':8s} {'rows':>6s} {'persist':>8s} {'latent':>8s} {'ratio':>6s} {'resid mean':>10s} {'share of latent old loss':>24s}")
    tot_lat = err_lat.sum()
    for lo, hi in ((1, 3), (4, 10), (11, 20), (21, 30), (31, 37), (38, 40)):
        m = has_ev & (biggest >= lo - 1) & (biggest <= hi - 1)
        print(f"  {lo:2d}-{hi:2d}    {m.sum():6d} {err_per[m].mean():8.4f} {err_lat[m].mean():8.4f} {err_lat[m].mean() / err_per[m].mean():6.3f} {resid[m].mean():+10.4f} {err_lat[m].sum() / tot_lat:24.3f}")
    m = ~has_ev
    print(f"  none     {m.sum():6d} {err_per[m].mean():8.4f} {err_lat[m].mean():8.4f} {err_lat[m].mean() / err_per[m].mean():6.3f} {resid[m].mean():+10.4f} {err_lat[m].sum() / tot_lat:24.3f}")

    # top-10% persistence-loss rows, the lead's operating point
    top = err_per > np.quantile(err_per, 0.9)
    print(f"\n  top-10% persistence-loss old rows: latent mean {err_lat[top].mean():.4f} vs persistence {err_per[top].mean():.4f}; "
          f"they carry {err_lat[top].sum() / tot_lat:.3f} of the latent old-age agb loss; {has_ev[top].mean():.3f} have an event")
    # the whole-model view: how much of the latent model's total public score is old-age agb on event rows
    tot_all = ((p40 - truth) ** 2).sum()
    print(f"  of the latent model's total public squared error, old-age agb on event rows is {err_lat[has_ev].sum() / tot_all:.3f}, "
          f"old-age agb on other rows {err_lat[~has_ev].sum() / tot_all:.3f}, everything else {1 - err_lat.sum() / tot_all:.3f}")
    # and the same for height on event rows
    err_lat_h = (p40[old][..., 0] - truth[old][..., 0]) ** 2
    print(f"  old-age height on event rows: {err_lat_h[has_ev].sum() / tot_all:.3f} of the total; on other rows {err_lat_h[~has_ev].sum() / tot_all:.3f}")


if __name__ == "__main__":
    main()

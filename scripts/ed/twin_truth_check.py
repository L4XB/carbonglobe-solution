#!/usr/bin/env python
"""Geographic-neighbour truth vs ED vs C on test sites: the organisers' training trajectories of the k nearest
training sites (estimate_lat_lon_v2 position of the test site, great circle, within --max-km) as a reference for
the 40-year change of a test site.

Per test site and seed age: neighbour truth ratio y40 / y0 (median over the k neighbours, raw units), the test
site's C ratio C / R and ED ratio (hybrid / R for agb, raw x40 / R for height) on the gated rows. Prints medians by
band and the correlation of the per-site old-band log ratios.
    python scripts/ed/twin_truth_check.py --group cl --k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import make_site_inputs as msi  # noqa: E402
import ed_transfer as et  # noqa: E402
from assemble_final import DIV, EPS, e0_rows  # noqa: E402
from cgb import io, shift  # noqa: E402

SEED_AGES = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)


def site_latlon(site: int, floors: np.ndarray, args0) -> tuple[float, float]:
    x = np.asarray(io.open_test("ssp126")[0][site], dtype=np.float64)
    cur, _ = msi.derive(x, args0, floors)
    return msi.estimate_lat_lon_v2(cur["rsds"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="cl", choices=["cl", "warm"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-km", type=float, default=300.0)
    ap.add_argument("--max-sites", type=int, default=400)
    ap.add_argument("--ed-dir", default=str(ROOT / "work" / "edv3" / "full_c400a" / "results"))
    args = ap.parse_args()
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    band = shift.band_of(age)
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C = rd("results/spatcorr_stack_20261106_s08.csv"); R = rd("results/latent_ref_y0_m114114.csv")
    ed = Path(args.ed_dir)
    H, sim, _ = et.read_ed(ed, keys, (1.0, 1.0), form="hybrid", R=R, smooth="despike")
    W, _, _ = et.read_ed(ed, keys, (1.0, 1.0), form="raw", R=R, smooth="despike")
    e0 = e0_rows(ed, keys)
    gate = np.abs(np.log((R * DIV + EPS) / (np.maximum(e0, 0.0) + EPS))) <= np.log(2.0)
    cl = np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0
    T = np.load(ROOT / "work" / "edv3" / "test_first_decade_T.npy")
    grp = cl if args.group == "cl" else (~cl & (T >= 8.0))
    sites = [s for s in np.unique(site[sim]) if grp[s]][: args.max_sites]
    tll = np.load(ROOT / "work" / "edv3" / "train_latlon.npy")
    la2, lo2 = np.radians(tll[:, 0]), np.radians(tll[:, 1])
    floors = msi.rsds_channel_floors(); args0 = msi.default_args(repair=True)
    ys = {a: io.open_train_y(a) for a in SEED_AGES}
    rows = []
    for s in sites:
        lat, lon = site_latlon(int(s), floors, args0)
        la1, lo1 = np.radians(lat), np.radians(lon)
        a_ = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a_, 0.0, 1.0)))
        nb = np.argsort(dist)[: args.k]; nb = nb[dist[nb] <= args.max_km]
        if len(nb) == 0:
            continue
        for a in SEED_AGES:
            y0 = np.asarray(ys[a][np.sort(nb), 0, 11, :2], dtype=np.float64); y40 = np.asarray(ys[a][np.sort(nb), 40, 11, :2], dtype=np.float64)
            ok = (y0[:, 0] > 0.5) & (y0[:, 1] > 0.05)
            if not ok.any():
                continue
            tr = np.median(y40[ok] / y0[ok], axis=0)
            for sc in ("ssp126", "ssp585"):
                i = np.flatnonzero((site == s) & (age == a) & (scen == sc))
                if len(i) == 0:
                    continue
                i = int(i[0])
                if not (R[i] > 0.02).all():
                    continue
                rows.append(dict(site=int(s), age=a, scen=sc, band=band[i], dist=float(np.median(dist[nb])), n_nb=int(ok.sum()),
                                 t_h=tr[0], t_a=tr[1], c_h=C[i, 0] / R[i, 0], c_a=C[i, 1] / R[i, 1],
                                 e_h=W[i, 0] / R[i, 0] if gate[i, 0] else np.nan, e_a=H[i, 1] / R[i, 1] if gate[i, 1] else np.nan,
                                 y0a_test=R[i, 1] * DIV[1], y0a_nb=float(np.median(y0[ok, 1]))))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"group {args.group}: {d.site.nunique()} sites with neighbours (k {args.k}, <= {args.max_km:.0f} km; median dist {d.dist.median():.0f} km)")
    print("median ratios y40/y0 by band and scenario: neighbour truth t_, C c_, ED e_ (h raw, agb hybrid, gated rows)")
    print(d.groupby(["band", "scen"])[["t_h", "c_h", "e_h", "t_a", "c_a", "e_a", "y0a_test", "y0a_nb"]].median().round(3).to_string())
    for col in ("h", "a"):
        for b in ("young", "mid", "old"):
            x = d[(d.band == b)].dropna(subset=[f"e_{col}"])
            if len(x) < 20:
                continue
            lt, lc, le = (np.log(np.clip(x[f"{p}_{col}"], 1e-3, None)) for p in ("t", "c", "e"))
            print(f"  {col} {b:5s} n {len(x):5d}: corr(log t, log C) {np.corrcoef(lt, lc)[0, 1]:+.2f} corr(log t, log ED) {np.corrcoef(lt, le)[0, 1]:+.2f} | "
                  f"rms log(C/t) {np.sqrt(np.mean((lc - lt) ** 2)):.3f} rms log(ED/t) {np.sqrt(np.mean((le - lt) ** 2)):.3f}")
    out = ROOT / "work" / "edv3" / f"twin_truth_{args.group}.csv"
    d.to_csv(out, index=False); print(f"rows -> {out}")


if __name__ == "__main__":
    main()

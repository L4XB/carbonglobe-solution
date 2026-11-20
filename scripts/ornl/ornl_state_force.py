#!/usr/bin/env python
"""State-dependent ORNL DAAC 2522 forcing effect per test ROW (site x seed age x scenario).

ORNL's existing-forest runs give, per 0.01-deg pixel, biomass in 2021 and 2060 under SSP126 / SSP585 (GFDL-ESM4,
CLIM_CO2) and under REF (present-day forcing repeated), all from the same 2020 state. For a test row with decoded y0
agb v, the effect of the future forcing on a stand like it is
    q_ssp = sum_i w_i log((B2060_ssp,i + e) / (B2060_REF,i + e)) / sum_i w_i,
    w_i = Gaussian(distance to the site's rsds position, sigma) x Gaussian(log(B2021_REF,i + e) - log(v + e), bw)
over page-2 pixels (0.04 deg, genuine 0.01-deg pixels) with existing forest. Seed age 1 uses the new-forest runs
(planted 2020, bare ground): q = log((N2060_ssp + e) / (N2060_REF + e)), distance weights only. Rows whose state
kernel has fewer than --min-neff effective pixels fall back to the distance-only existing-forest mean.
Also stores g_ref (REF log growth 2021 -> 2060 of the matched pixels) and g_ssp for diagnostics.
    python scripts/ornl/ornl_state_force.py [--sigma 1.0] [--bw 0.35]
-> work/ornl2522/state_force_GFDL-ESM4_CLIM_CO2_s<sigma>.npz with (n_rows,) arrays in template row order
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

D = ROOT / "work" / "ornl2522"
DIV = np.array([10.90645, 3.39567])
EPS = 0.05
RES = 0.04


def p2(stem: str, band: int) -> np.ndarray:
    a = np.load(D / f"p2_{stem}_b{band}.npy")
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="GFDL-ESM4")
    ap.add_argument("--case", default="CLIM_CO2")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--bw", type=float, default=0.35)
    ap.add_argument("--min-neff", type=float, default=30.0)
    ap.add_argument("--max-px", type=int, default=60000)
    ap.add_argument("--pos", default="work/test_latlon_v2.npy", help="(n, 2) lat/lon array, or an npz with lat / lon")
    ap.add_argument("--tag", default="", help="output file-name suffix (e.g. _nx2)")
    ap.add_argument("--ref-case", default="", help="use <ESM>_<SSP>_<ref-case> 2060 as the reference instead of REF (e.g. CLIM: q = CO2-only effect)")
    args = ap.parse_args()
    t0 = time.time()
    e = f"Biomass_{args.esm}"
    xref = p2("Biomass_REF_existing_forest_2021_2030", 0); yref = p2("Biomass_REF_existing_forest_2051_2060", 9)
    y126 = p2(f"{e}_SSP126_{args.case}_existing_forest_2051_2060", 9); y585 = p2(f"{e}_SSP585_{args.case}_existing_forest_2051_2060", 9)
    nref = p2("Biomass_REF_new_forest_2051_2060", 9)
    n126 = p2(f"{e}_SSP126_{args.case}_new_forest_2051_2060", 9); n585 = p2(f"{e}_SSP585_{args.case}_new_forest_2051_2060", 9)
    ex_ok = (xref > 0.05) & np.isfinite(yref) & np.isfinite(y126) & np.isfinite(y585)
    nw_ok = np.isfinite(nref) & np.isfinite(n126) & np.isfinite(n585) & ((nref > 0) | (n585 > 0) | (n126 > 0))
    lx = np.log(xref + EPS)
    qe = {s: np.log(y + EPS) - np.log(yref + EPS) for s, y in (("126", y126), ("585", y585))}
    qn = {s: np.log(n + EPS) - np.log(nref + EPS) for s, n in (("126", n126), ("585", n585))}
    if args.ref_case:                     # reference = the same ESM x SSP under ref-case (e.g. CLIM: q = the CO2-only effect)
        yc = {s: p2(f"{e}_SSP{s}_{args.ref_case}_existing_forest_2051_2060", 9) for s in ("126", "585")}
        ex_ok &= np.isfinite(yc["126"]) & np.isfinite(yc["585"])
        qe = {s: np.log(y + EPS) - np.log(yc[s] + EPS) for s, y in (("126", y126), ("585", y585))}
        qn = {s: np.full_like(v, np.nan) for s, v in qn.items()}
    gref = np.log(yref + EPS) - np.log(xref + EPS)
    H, W = xref.shape
    print(f"loaded page-2 grids {H}x{W} in {time.time() - t0:.0f} s; existing-forest pixels {ex_ok.sum()}, new-forest {nw_ok.sum()}", flush=True)

    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(np.float64) * DIV
    pz = np.load(ROOT / args.pos)
    pos = np.stack([pz["lat"], pz["lon"]], 1) if hasattr(pz, "files") else pz
    n = len(keys)
    out = {k: np.full(n, np.nan) for k in ("q", "neff", "gref", "gssp", "q_site")}
    rng = np.random.default_rng(42)
    rows_of = {}
    for i in range(n):
        rows_of.setdefault(int(site[i]), []).append(i)
    rad = 3.0 * args.sigma
    for k, (s_id, rows) in enumerate(sorted(rows_of.items())):
        la, lo = pos[s_id]
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        r0 = int(np.clip(np.floor((80.0 - (la + rad)) / RES), 0, H)); r1 = int(np.clip(np.ceil((80.0 - (la - rad)) / RES), 0, H))
        radc = rad / max(np.cos(np.radians(la)), 0.2)
        c0 = int(np.floor((lo - radc + 180.0) / RES)); c1 = int(np.ceil((lo + radc + 180.0) / RES))
        cc = np.arange(c0, c1) % W
        if r1 <= r0:
            continue
        sl = (slice(r0, r1), cc)
        plat = 80.0 - (np.arange(r0, r1) + 0.5) * RES; plon = -180.0 + (cc + 0.5) * RES
        dy = plat[:, None] - la; dx = (((plon[None, :] - lo + 180) % 360) - 180) * np.cos(np.radians(la))
        g = np.exp(-0.5 * (dx ** 2 + dy ** 2) / args.sigma ** 2)
        okE = ex_ok[sl[0]][:, sl[1]]; okN = nw_ok[sl[0]][:, sl[1]]
        gE = g[okE]; gN = g[okN]
        lxE = lx[sl[0]][:, sl[1]][okE]; grE = gref[sl[0]][:, sl[1]][okE]
        qE = {s: qe[s][sl[0]][:, sl[1]][okE] for s in qe}; qN = {s: qn[s][sl[0]][:, sl[1]][okN] for s in qn}
        if gE.size > args.max_px:
            pick = rng.choice(gE.size, args.max_px, replace=False)
            gE, lxE, grE = gE[pick], lxE[pick], grE[pick]; qE = {s: v[pick] for s, v in qE.items()}
        for i in rows:
            s = "585" if scen[i] == "ssp585" else "126"
            if gE.sum() > 1e-6:
                out["q_site"][i] = float((gE * qE[s]).sum() / gE.sum())
            if age[i] == 1:
                if gN.sum() > 1e-6:
                    out["q"][i] = float((gN * qN[s]).sum() / gN.sum()); out["neff"][i] = gN.sum() ** 2 / (gN ** 2).sum()
                continue
            if gE.sum() <= 1e-6:
                continue
            v = max(R[i, 1], 0.0)
            w = gE * np.exp(-0.5 * ((lxE - np.log(v + EPS)) / args.bw) ** 2)
            sw = w.sum(); neff = sw ** 2 / max((w ** 2).sum(), 1e-30)
            if neff < args.min_neff:
                w = gE; sw = w.sum(); neff = -(sw ** 2 / (w ** 2).sum())        # negative neff marks the fallback
            out["q"][i] = float((w * qE[s]).sum() / sw); out["neff"][i] = neff
            out["gref"][i] = float((w * grE).sum() / sw); out["gssp"][i] = out["gref"][i] + out["q"][i]
        if k % 500 == 0:
            print(f"  {k}/{len(rows_of)} sites, {time.time() - t0:.0f} s", flush=True)
    fn = D / f"state_force_{args.esm}_{args.case}_s{args.sigma:g}{'_vs' + args.ref_case if args.ref_case else ''}{args.tag}.npz"
    np.savez(fn, **out)
    ok = np.isfinite(out["q"])
    print(f"wrote {fn.name}: rows with q {ok.sum()} of {n}; fallback rows {(out['neff'] < 0).sum()}; "
          f"median q585 {np.nanmedian(out['q'][scen == 'ssp585']):+.3f}, q126 {np.nanmedian(out['q'][scen == 'ssp126']):+.3f}")


if __name__ == "__main__":
    main()

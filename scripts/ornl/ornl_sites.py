#!/usr/bin/env python
"""Regional future-forcing response of existing forests at each test site, from the ORNL DAAC 2522 coarse grids.

Inputs: work/ornl2522/coarse_Biomass_<ESM>_<SSP>_<CASE>_existing_forest_<decade>.npz and coarse_Biomass_REF_*.npz
(scripts/ornl/ornl_coarse.py), test-site positions work/test_latlon_v2.npy (rsds estimator, rms ~1.3 deg).
For each site: Gaussian-weighted (sigma --sigma deg, cells within 3 sigma, weight x forest fraction) mean biomass in
2021 (first band of 2021_2030) and 2060 (last band of 2051_2060) for SSP126, SSP585 and REF. Outputs per site
    r_<ssp> = B2060 / B2021 (existing forest, that scenario), r_ref (present-day forcing repeated),
    force_<ssp> = log r_<ssp> - log r_ref   (the future forcing effect on existing forests over 2021-2060),
    contrast = log r_585 - log r_126, wsum (forest weight; ~0 = no forest nearby)
-> work/ornl2522/site_response_<ESM>_<CASE>.npz
    python scripts/ornl/ornl_sites.py --esm GFDL-ESM4 --case CLIM_CO2 [--sigma 1.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"


def load(stem: str, band: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(D / f"coarse_{stem}.npz")
    b = int(z["bands"][0] if band == "first" else z["bands"][-1])
    return z["lat"], z["lon"], z[f"mean{b}"], z[f"frac{b}"]


def site_means(lat: np.ndarray, lon: np.ndarray, mean: np.ndarray, frac: np.ndarray, pos: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian x forest-fraction weighted mean of ``mean`` around each site position; returns (value, weight sum)."""
    dlat = abs(float(lat[1] - lat[0])); dlon = abs(float(lon[1] - lon[0]))
    rad = int(np.ceil(3 * sigma / dlat))
    m = np.nan_to_num(mean, nan=0.0); f = np.where(np.isfinite(mean), frac, 0.0)
    out = np.full(len(pos), np.nan); wsum = np.zeros(len(pos))
    for i, (la, lo) in enumerate(pos):
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        r0 = int(np.argmin(np.abs(lat - la))); c0 = int(np.argmin(np.abs(((lon - lo + 180) % 360) - 180)))
        rr = np.arange(max(0, r0 - rad), min(len(lat), r0 + rad + 1))
        radc = int(np.ceil(3 * sigma / (dlon * max(np.cos(np.radians(la)), 0.2))))
        cc = np.arange(c0 - radc, c0 + radc + 1) % len(lon)
        la_g = lat[rr][:, None]; lo_g = lon[cc][None, :]
        dx = (((lo_g - lo + 180) % 360) - 180) * np.cos(np.radians(la)); dy = la_g - la
        g = np.exp(-0.5 * (dx ** 2 + dy ** 2) / sigma ** 2)
        w = g * f[np.ix_(rr, cc)]
        s = float(w.sum())
        if s > 0:
            out[i] = float((w * m[np.ix_(rr, cc)]).sum() / s)
        wsum[i] = s
    return out, wsum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="GFDL-ESM4")
    ap.add_argument("--case", default="CLIM_CO2")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--forest", default="existing", choices=["existing", "new"])
    ap.add_argument("--pos", default="work/test_latlon_v2.npy", help="(n, 2) lat/lon array, or an npz with lat / lon")
    ap.add_argument("--out-suffix", default=None, help="file-name suffix (default: derived from --pos)")
    args = ap.parse_args()
    pz = np.load(ROOT / args.pos)
    pos = np.stack([pz["lat"], pz["lon"]], 1) if hasattr(pz, "files") else pz
    res = {}
    for tag, stem21, stem60 in (("126", f"Biomass_{args.esm}_SSP126_{args.case}_{args.forest}_forest_2021_2030", f"Biomass_{args.esm}_SSP126_{args.case}_{args.forest}_forest_2051_2060"),
                                ("585", f"Biomass_{args.esm}_SSP585_{args.case}_{args.forest}_forest_2021_2030", f"Biomass_{args.esm}_SSP585_{args.case}_{args.forest}_forest_2051_2060"),
                                ("ref", f"Biomass_REF_{args.forest}_forest_2021_2030", f"Biomass_REF_{args.forest}_forest_2051_2060")):
        lat, lon, m21, f21 = load(stem21, "first")
        _, _, m60, f60 = load(stem60, "last")
        b21, w21 = site_means(lat, lon, m21, f21, pos, args.sigma)
        b60, w60 = site_means(lat, lon, m60, f60, pos, args.sigma)
        res[f"b21_{tag}"], res[f"b60_{tag}"], res[f"w_{tag}"] = b21, b60, w21
        res[f"r_{tag}"] = b60 / b21
        print(f"{tag}: sites with forest weight > 0.05: {(w21 > 0.05).sum()}; median r {np.nanmedian(b60 / b21):.3f}", flush=True)
    for s in ("126", "585"):
        res[f"force_{s}"] = np.log(res[f"r_{s}"]) - np.log(res["r_ref"])
    res["contrast"] = np.log(res["r_585"]) - np.log(res["r_126"])
    suf = args.out_suffix if args.out_suffix is not None else ('' if args.pos == 'work/test_latlon_v2.npy' else '_' + Path(args.pos).stem)
    out = D / f"site_response_{args.esm}_{args.case}{'' if args.forest == 'existing' else '_new'}{suf}.npz"
    np.savez(out, **res)
    ok = res["w_ref"] > 0.05
    print(f"wrote {out.name}: force_126 median {np.nanmedian(res['force_126'][ok]):+.3f}, force_585 {np.nanmedian(res['force_585'][ok]):+.3f}, "
          f"contrast {np.nanmedian(res['contrast'][ok]):+.3f} (sd {np.nanstd(res['contrast'][ok]):.3f})")
    cl = (np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)
    print(f"cluster: force_585 median {np.nanmedian(res['force_585'][cl & ok]):+.3f}, r_585 {np.nanmedian(res['r_585'][cl & ok]):.3f}, r_ref {np.nanmedian(res['r_ref'][cl & ok]):.3f}")


if __name__ == "__main__":
    main()

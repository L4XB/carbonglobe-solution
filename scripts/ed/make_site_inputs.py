#!/usr/bin/env python
"""Write ED v3.0 single-site input files from the CarbonGlobe test arrays.

One directory per (site, scenario) with the netCDF-3 files ED's loaders read
(``work/edv3/EDv3_code/read_site_data.cc``), on a 2 x 2 shim grid whose cell (0, 0)
is the site (the other three cells carry area 0 / fill values, so ``model_site``
skips them; two cells per axis are needed because ``is_soi`` takes
``|lats[0] - lats[1]|`` as the cell size).  See ``work/edv3/RUNBOOK.md`` for the
code changes that make ED read these files (grid 2 x 2 instead of 360 x 720,
CO2 taken as written, 40 window years).

Channel layout of ``test_x`` (notes/theory.md Sec. 1, verified on the arrays):
  0 ta_m [K], 1 pr [mm/month], 2-7 tsl1..tsl6 [K], 8-31 CO2 hour 0..23 [ppm],
  32 dst (treefall rate, 0.012), 33+4h hus [kg/kg], 34+4h ta_h [K], 35+4h rsds [W/m2],
  36+4h wind [m/s] for h = 0..23, 129-135 soil: k_sat, saturated_theta, residual_theta,
  L, n, m, soil_depth (ROSETTA / Mualem-van Genuchten; ED reads all but n).

Repairs (round-1 red-team findings; OFF by default since 2026-11-07, because the spin-up with
the channels exactly as delivered and the stored CO2 reproduces test_y0 best; ``--repair`` on):
  * rsds: each hour channel of the test carries a constant additive floor (-26..+77 W/m2
    across the 24 channels; training has exact zeros at night).  The floor is the channel
    minimum over the whole test (both scenarios) and is subtracted; values are clipped at 0.
  * hus: clipped at 0 (test floors of -1e-4 .. -2e-4 kg/kg).
  * ta_h: the 24 hourly values of each site-month are shifted so that their mean equals
    ta_m, the identity that holds exactly in training (rms break in the test 0.31 K).
  * SoilTemp (hourly soil temperature, needed by ED's premech) is not in the test: the
    monthly tsl1 is repeated over the 24 hours (always; Ts only enters root respiration).

Spin-up climatology: ``*_avg.nc`` hold the mean over the first ``--spinup-years`` years (0 = all)
of this scenario, or of BOTH scenarios with ``--spinup-both`` (the organisers' seed-age states
are nearly identical across scenarios, so one present-day climatology produced both).

CO2 options (``--co2``): ``asis`` writes the stored channels (default); ``real`` decompresses
them to SSP levels using the stored series' exact shape; ``fixed`` writes one constant.  ED's
single_year mode multiplies the CO2 file by 280/390 unless the ED_CO2_AS_IS patch is active
(RUNBOOK Sec. 3); ``--co2-premultiply`` cancels that factor for an unpatched binary.

The window years are labelled ``--first-year`` .. +39 and a 41st copy of the last year is written,
because ED reloads the next year's files before it prints the December state of year 40.

Usage (trafficflow venv has numpy + scipy):
  python scripts/ed/make_site_inputs.py --sites 0,4702 --scenario both --spinup-years 10 --spinup-both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

try:
    from scipy.io import netcdf_file
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required (use python)") from exc

FLOORS = np.array([-0.900947, -0.401412, -0.532395, 0.117938, 0.259026, 0.127815, 0.084656])
DECODE_MULT = np.array([1.14114, 1.14114, 1.14114, 1.0, 1.14114, 1.14114, 1.14114])
STATE_NAMES = io.STATE_NAMES
SEED_AGES = io.SEED_AGES
HOURS = np.arange(24)
CH_HUS, CH_TA, CH_RSDS, CH_WIND = 33 + 4 * HOURS, 34 + 4 * HOURS, 35 + 4 * HOURS, 36 + 4 * HOURS
CH_CO2 = np.arange(8, 32)
SOIL_CH = {"k_sat": 129, "saturated_theta": 130, "residual_theta": 131, "L": 132, "n": 133, "m": 134, "soil_depth": 135}
FILL = -9999.0
FLOOR_CACHE = ROOT / "work" / "edv3" / "rsds_channel_floors.json"


def default_args(**over) -> argparse.Namespace:
    """The CLI defaults as a namespace (for programmatic use by scripts/ed/run_site.py)."""
    ns = argparse.Namespace(sites="", scenario="both", out=str(ROOT / "work" / "edv3" / "sites"), first_year=1981,
                            co2="asis", co2_fixed=400.0, co2_real_start=400.0, co2_real_rise126=52.0, co2_real_rise585=190.0,
                            co2_spinup="first", co2_spinup_fixed=280.0, co2_premultiply=False, ta_amp_factor=1.0,
                            repair=False, spinup_years=0, spinup_both=False, co2_json=str(ROOT / "work" / "edv3" / "co2_real_paths.json"),
                            repair_rsds_only=False, deflate=False, fire_on=False, pr_scale=1.0, hus_cap_rh=0.0, ksat_scale=1.0, spinup_twin=False,
                            spinup_start=0, fp1=1.0, treefall=0.012, spinup_cycle=False, spinup_shift=0.0, window_repair=True,
                            spinup_geo_twin=False)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ----------------------------------------------------------------------------- helpers
def rsds_channel_floors() -> np.ndarray:
    """Per-hour-channel minimum of rsds over the whole test (both scenarios), cached."""
    if FLOOR_CACHE.exists():
        return np.asarray(json.load(open(FLOOR_CACHE))["floors"], dtype=np.float64)
    floors = np.full(24, np.inf)
    for s in io.SCENARIOS:
        x, _ = io.open_test(s)
        for lo in range(0, x.shape[0], 1024):
            blk = np.asarray(x[lo : lo + 1024][:, :, :, CH_RSDS], dtype=np.float64)
            floors = np.minimum(floors, blk.reshape(-1, 24).min(0))
    FLOOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"floors": floors.tolist(), "note": "min over both scenarios, all sites/years/months, per rsds hour channel 35+4h"}, open(FLOOR_CACHE, "w"), indent=1)
    return floors


def toa_daily_insolation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    cos_h0 = np.clip(-np.tan(lat) * np.tan(decl), -1, 1)
    h0 = np.arccos(cos_h0)
    return 1361.0 / np.pi * (h0 * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(h0))


def estimate_lat_lon(rsds: np.ndarray) -> tuple[float, float, float]:
    """Placeholders for the grid/sois (ED's physics does not use them in this build, RUNBOOK Sec. 0)."""
    daily = rsds.mean(axis=(0, 2))
    doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    best = (-np.inf, 0.0)
    for lat in np.arange(-65.0, 75.5, 0.5):
        toa = toa_daily_insolation(lat, doy)
        if toa.std() < 1e-6 or daily.std() < 1e-6:
            continue
        r = np.corrcoef(daily, toa)[0, 1]
        if np.max(daily / np.maximum(toa, 1.0)) > 1.0:
            r -= 1.0
        if r > best[0]:
            best = (r, lat)
    h_peak = float(np.argmax(rsds.mean(axis=(0, 1))))
    lon = ((15.0 * (12.0 - h_peak) + 180.0) % 360.0) - 180.0
    return float(best[1]), float(lon), float(best[0])


# ----------------------------------------------------------------------------- geography from the rsds channels (v2)
_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
_LATS = np.arange(-70.0, 80.001, 0.25)
_DL_TAB: np.ndarray | None = None
_TRAIN_LL: np.ndarray | None = None
LAT_CAL = (-0.10, 1.072)      # lat_true ~ a + b * lat_daylength, fitted on 200 training sites (held-out rms 1.3 deg)
LON_OFFSET_DEG = 8.73         # the hourly channel h is centred ~0.58 h after the hour label: fitted on 200 training sites (held-out rms 1.2 deg)


def day_length_hours(lat: float, doy: np.ndarray) -> np.ndarray:
    dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    return 24.0 / np.pi * np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(dec), -1.0, 1.0))


def _frac_day_length(p: np.ndarray, thr_frac: float = 0.02) -> float:
    """Hours per day with the mean diurnal profile ``p`` (24 UTC hours) above thr_frac x its maximum, crossings interpolated."""
    thr = thr_frac * p.max(); above = p > thr
    if above.all():
        return 24.0
    if not above.any():
        return 0.0
    q = np.roll(p, -int(np.argmin(p))); a = q > thr
    i0 = int(np.argmax(a)); i1 = 23 - int(np.argmax(a[::-1]))
    rise = i0 - (q[i0] - thr) / max(q[i0] - q[i0 - 1], 1e-9) if i0 > 0 else i0 - 0.5
    sett = i1 + (q[i1] - thr) / max(q[i1] - q[(i1 + 1) % 24], 1e-9) if i1 < 23 else i1 + 0.5
    return float(np.clip(sett - rise, 0.0, 24.0))


_NEX3_LL: np.ndarray | None = None
_EXACT_LL: np.ndarray | None = None
_CAL: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def exact_latlon(site: int) -> tuple[float, float]:
    """Centre of the test site's exact training cell (scripts/geo/cell_identity_global.py: 5994 of 6171 sites match one
    training cell on the soil channels); the NEX-anomaly position for the 177 sites without an exact cell."""
    global _EXACT_LL
    if _EXACT_LL is None:
        z = np.load(ROOT / "work" / "test_cell_exact.npz")
        ll = np.stack([z["lat"], z["lon"]], 1).astype(np.float64)
        bad = ~z["exact"]
        z3 = np.load(ROOT / "work" / "test_latlon_nex3.npz")
        ll[bad, 0], ll[bad, 1] = z3["lat"][bad], z3["lon"][bad]
        _EXACT_LL = ll
    return float(_EXACT_LL[site, 0]), float(_EXACT_LL[site, 1])


def cal_maps(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel affine maps released = alpha * real + beta (scripts/geo/make_cal_cache.py -> work/cal/cal_maps_*.npz)."""
    if path not in _CAL:
        z = np.load(ROOT / path)
        _CAL[path] = (z["alpha"].astype(np.float64), z["beta"].astype(np.float64))
    return _CAL[path]


def nex3_latlon(site: int) -> tuple[float, float]:
    """Test-site position from the NEX-GDDP weather-anomaly geolocation (scripts/nex/geolocate_nex3.py; static-map check
    corr 0.987 vs 0.79 for estimate_lat_lon_v2, whose longitudes are ~4-5 deg off on the test)."""
    global _NEX3_LL
    if _NEX3_LL is None:
        z = np.load(ROOT / "work" / "test_latlon_nex3.npz")
        _NEX3_LL = np.stack([z["lat"], z["lon"]], 1)
    return float(_NEX3_LL[site, 0]), float(_NEX3_LL[site, 1])


def estimate_lat_lon_v2(rsds: np.ndarray) -> tuple[float, float]:
    """Latitude from the seasonal cycle of the day length (cloud-independent), longitude from the sub-hour peak of the
    mean diurnal cycle (UTC hours); both calibrated on training sites against cache/mask2d.npy (held-out rms 1.3 / 1.2 deg).
    ``rsds``: (year, month, hour) after the floor repair (night values ~0)."""
    global _DL_TAB
    if _DL_TAB is None:
        _DL_TAB = np.array([day_length_hours(lat, _DOY) for lat in _LATS])
    prof = rsds.mean(axis=0)                                                 # (12, 24)
    dl = np.array([_frac_day_length(p) for p in prof])
    lat = LAT_CAL[0] + LAT_CAL[1] * float(_LATS[int(np.argmin(((_DL_TAB - dl) ** 2).mean(axis=1)))])
    d = rsds.mean(axis=(0, 1)); h = int(np.argmax(d)); y0, y1, y2 = d[(h - 1) % 24], d[h], d[(h + 1) % 24]
    den = y0 - 2 * y1 + y2
    hp = h + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
    lon = ((15.0 * (12.0 - hp) - LON_OFFSET_DEG + 180.0) % 360.0) - 180.0
    return float(np.clip(lat, -70.0, 80.0)), float(lon)


def geo_twin(lat: float, lon: float, years: tuple[int, int] = (30, 40)) -> dict:
    """Nearest training site (great circle; coordinates from cache/mask2d.npy row-major cells, work/edv3/train_latlon.npy)
    and its climatology over training window years ``years`` (30-39 = 2011-2020) in the writer's spin-up layout."""
    global _TRAIN_LL
    if _TRAIN_LL is None:
        _TRAIN_LL = np.load(ROOT / "work" / "edv3" / "train_latlon.npy")
    la1, lo1 = np.radians(lat), np.radians(lon); la2, lo2 = np.radians(_TRAIN_LL[:, 0]), np.radians(_TRAIN_LL[:, 1])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    j = int(np.argmin(dist))
    T = np.asarray(io.open_train_x()[j, years[0]:years[1]], dtype=np.float64).mean(axis=0)   # (12, 136)
    return {"ta_m": T[:, 0], "pr": T[:, 1], "tsl": T[:, 2:8], "hus": T[:, CH_HUS], "ta_h": T[:, CH_TA], "rsds": T[:, CH_RSDS], "wind": T[:, CH_WIND],
            "co2": np.full((12, 24), float(T[:, CH_CO2].mean())), "soil_t_hourly": np.repeat(T[:, 2:3], 24, axis=1),
            "twin_site": j, "twin_dist_km": float(dist[j]), "twin_lat": float(_TRAIN_LL[j, 0]), "twin_lon": float(_TRAIN_LL[j, 1]), "twin_years": list(years)}


def write_nc(path: Path, dims: dict[str, int], variables: dict[str, tuple[tuple[str, ...], np.ndarray, str, dict]]) -> None:
    with netcdf_file(str(path), "w", version=2) as nc:
        for d, n in dims.items():
            nc.createDimension(d, n)
        for name, (vdims, arr, dtype, attrs) in variables.items():
            v = nc.createVariable(name, dtype, vdims)
            v[:] = np.asarray(arr, dtype=dtype)
            for k, val in attrs.items():
                setattr(v, k, val)


def shim_grid(values: np.ndarray, fill: float = FILL) -> np.ndarray:
    out = np.full(values.shape + (2, 2), fill, dtype=np.float64)
    out[..., 0, 0] = values
    return out


def derive(x: np.ndarray, args: argparse.Namespace, floors: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Per-scenario driver arrays (year, month[, hour]) with the optional repairs applied."""
    ta_m, pr = x[:, :, 0], x[:, :, 1]
    tsl = x[:, :, 2:8]
    if (pr < 0).any():
        raise ValueError("negative precipitation")
    hus, ta_h, rsds, wind = x[:, :, CH_HUS], x[:, :, CH_TA], x[:, :, CH_RSDS], x[:, :, CH_WIND]
    co2 = x[:, :, CH_CO2]
    repairs: dict[str, str] = {}
    rsds_only = getattr(args, "repair_rsds_only", False)
    if args.repair or rsds_only:
        rsds = np.clip(rsds - floors[None, None, :], 0.0, None)
        repairs["rsds"] = "per-hour-channel floor subtracted (work/edv3/rsds_channel_floors.json), clipped at 0"
        hus_neg = float((hus < 0).mean())
        hus = np.clip(hus, 0.0, None)
        repairs["hus"] = f"clipped at 0 ({hus_neg:.4f} of cells were negative)"
    if args.repair and not rsds_only:
        shift = ta_m - ta_h.mean(axis=2)
        ta_h = ta_h + shift[:, :, None]
        repairs["ta_h"] = f"shifted so mean over 24 h equals ta_m (rms shift {np.sqrt((shift**2).mean()):.3f} K)"
        if getattr(args, "deflate", False):
            # full inverse of a per-channel affine inflation around the daily mean: the test's diurnal amplitudes are
            # 1.19x (ta_h), 1.195x (hus), 1.11x (rsds) the training's at equal daily means (board-solve 11-02)
            mean = ta_h.mean(axis=2, keepdims=True); ta_h = mean + (ta_h - mean) / 1.19
            mean = hus.mean(axis=2, keepdims=True); hus = np.clip(mean + (hus - mean) / 1.195, 0.0, None)
            rsds = rsds / 1.11
            repairs["deflate"] = "diurnal anomalies deflated: ta_h /1.19, hus /1.195 (about the daily mean), rsds /1.11"
        if args.ta_amp_factor != 1.0:
            mean = ta_h.mean(axis=2, keepdims=True)
            ta_h = mean + args.ta_amp_factor * (ta_h - mean)
            repairs["ta_h_amplitude"] = f"diurnal anomalies scaled by {args.ta_amp_factor}"
    else:
        repairs["none"] = "channels written exactly as delivered (rsds floors, negative hus and the ta_m/ta_h mismatch kept)"
    # test precipitation floor (11-08): every test site's driest months sit at exactly 6.3517 mm/month (5.9% of all
    # months within 0.1 mm of it, none below; training has 5.1% exact zeros) -> an additive release artefact like the
    # rsds night offsets. ED's fire ignites only below 8.33 mm/month, so the offset suppresses dry-month ignition.
    if getattr(args, "pr_floor", 0.0) > 0.0:
        pr = np.clip(pr - args.pr_floor, 0.0, None); repairs["pr_floor"] = f"precipitation floor {args.pr_floor} subtracted, clipped at 0"
    if getattr(args, "ta_nex", False):
        # 11-09: the released test temperatures sit 0.3-3.8 K below NEX-GDDP-CMIP6 GFDL-ESM4 (temperature-dependent); add
        # the inverse shift (interpolated in the released temperature) to every temperature channel
        kt, ks = np.array([-24.7, -5.0, 7.3, 16.7, 26.2]), np.array([-0.26, 2.50, 2.74, 3.32, 3.84])
        sh = lambda t: t + np.interp(t - 273.15, kt, ks)
        ta_m, tsl, ta_h = sh(ta_m), sh(tsl), sh(ta_h)
        repairs["ta_nex"] = "temperatures shifted to the NEX-GDDP GFDL-ESM4 level (+0.3..3.8 K, temperature-dependent)"
    if getattr(args, "hus_nex", False):
        hus = np.clip(hus, 0.0, None) / 0.767; repairs["hus_nex"] = "specific humidity / 0.767 (released ~0.767 x NEX huss)"
    # diagnostic knobs (misfit tests, 2026-11-08)
    if getattr(args, "pr_scale", 1.0) != 1.0:
        pr = pr * args.pr_scale; repairs["pr_scale"] = f"precipitation x {args.pr_scale}"
    if getattr(args, "hus_cap_rh", 0.0) > 0:
        es = 611.2 * np.exp(17.67 * (ta_h - 273.15) / (ta_h - 29.65)); qsat = 0.622 * es / (101300 - 0.378 * es)
        hus = np.minimum(hus, args.hus_cap_rh * qsat); repairs["hus_cap_rh"] = f"hus capped at RH {args.hus_cap_rh}"
    return {"ta_m": ta_m, "pr": pr, "tsl": tsl, "hus": hus, "ta_h": ta_h, "rsds": rsds, "wind": wind, "co2": co2,
            "soil_t_hourly": np.repeat(tsl[:, :, 0:1], 24, axis=2)}, repairs


def co2_path(co2: np.ndarray, scenario: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    """Window CO2 (40, 12, 24), the year-0 (2020) cell (12, 24) and a note.

    ``real`` uses the exact decompression in ``--co2-json`` (provenance agent: stored = 0.1296 real + 309.74,
    window 2021-2060, official SSP annual means constant over the 288 month-hours; the stored cell-level
    deviations are release noise); year 0 is extrapolated back by one year's increment."""
    stored_annual = co2.mean(axis=(1, 2))
    if args.co2 == "real":
        jpath = Path(args.co2_json)
        if jpath.exists():
            j = json.load(open(jpath))
            v = np.asarray(j["decompressed"][scenario]["site_mean_decompressed"], dtype=np.float64)
            if v.shape[0] != co2.shape[0]:
                raise ValueError(f"{jpath}: {v.shape[0]} years for {co2.shape[0]} window years")
            arr = np.repeat(np.repeat(v[:, None, None], 12, axis=1), 24, axis=2)
            y0v = 2.0 * v[0] - v[1]
            return arr, np.full((12, 24), y0v), f"exact decompression ({jpath.name}): {v[0]:.1f} -> {v[-1]:.1f} ppm, year-0 {y0v:.1f}"
        rise_real = args.co2_real_rise126 if scenario == "ssp126" else args.co2_real_rise585
        alpha = (stored_annual[-1] - stored_annual[0]) / rise_real
        arr = args.co2_real_start + (co2 - stored_annual[0]) / alpha
        return arr, arr[0], f"decompressed by shape: start {args.co2_real_start} ppm, rise {rise_real} ppm (alpha {alpha:.4f})"
    if args.co2 == "fixed":
        arr = np.full_like(co2, args.co2_fixed)
        return arr, arr[0], f"fixed {args.co2_fixed} ppm"
    return co2, co2[0], "stored channels as-is (year 0 = stored year 1)"


_TRAIN_CLIM = None


def train_twin(cur: dict[str, np.ndarray], floors: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Nearest training site in standardised climatology space (12 x ta_m, log pr, daily-mean rsds, ta_h range);
    returns its MERRA-2 climatology in the writer's spin-up layout (month[, hour] arrays)."""
    global _TRAIN_CLIM
    xs = np.load(ROOT / "work" / "cache" / "x_season.npy", mmap_mode="r")  # (54152, 136, 12)
    if _TRAIN_CLIM is None:
        S = np.asarray(xs[:, :129, :], dtype=np.float32)
        ta = S[:, 0, :]; pr = np.log1p(S[:, 1, :]); rsd = S[:, CH_RSDS, :].mean(1); tah = S[:, CH_TA, :].max(1) - S[:, CH_TA, :].min(1)
        feats = np.concatenate([ta, 10 * pr, rsd / 10.0, tah], axis=1)
        _TRAIN_CLIM = (feats, feats.mean(0), feats.std(0) + 1e-6)
    feats, mu, sd = _TRAIN_CLIM
    ta_s = cur["ta_m"].mean(0); pr_s = np.log1p(cur["pr"].mean(0)); rsd_s = cur["rsds"].mean(axis=(0, 2)); tah_s = (cur["ta_h"].max(2) - cur["ta_h"].min(2)).mean(0)
    f = np.concatenate([ta_s, 10 * pr_s, rsd_s / 10.0, tah_s])
    d = (((feats - mu) / sd - (f - mu) / sd) ** 2).sum(1)
    j = int(np.argmin(d))
    T = np.asarray(xs[j], dtype=np.float64)  # (136, 12)
    hourly = lambda chs: T[chs, :].T  # (12, 24)
    out = {"ta_m": T[0], "pr": T[1], "tsl": T[2:8].T, "hus": hourly(CH_HUS), "ta_h": hourly(CH_TA), "rsds": hourly(CH_RSDS), "wind": hourly(CH_WIND),
           "co2": np.full((12, 24), float(T[CH_CO2].mean())), "soil_t_hourly": np.repeat(T[2][:, None], 24, axis=1), "twin_site": j, "twin_dist": float(d[j])}
    return out


# ----------------------------------------------------------------------------- per-site builder
def build_site(site: int, scenario: str, args: argparse.Namespace, floors: np.ndarray, latlon: tuple[float, float] | None = None) -> dict:
    """Write one (site, scenario) input set; ``latlon`` forces the grid position (both scenarios of a site must share it,
    because ED names the site by lat/lon and the window restarts read the spin-up dumps by that name)."""
    x_all, y0_all = io.open_test(scenario)
    x = np.asarray(x_all[site], dtype=np.float64)  # (40, 12, 136)
    y0 = np.asarray(y0_all[site], dtype=np.float64)  # (15, 7)
    out = Path(args.out) / f"site{site:05d}_{scenario}"
    out.mkdir(parents=True, exist_ok=True)
    ny = x.shape[0]
    years = np.arange(args.first_year, args.first_year + ny)

    cal_note = ""
    if getattr(args, "cal_maps", ""):
        # 11-09: every released channel is an affine image of the real driver; map all of them back onto the training
        # (MERRA-2) scale with the exact-twin fits. CO2 stays as stored here (co2_path decompresses it exactly).
        al, be = cal_maps(args.cal_maps)
        skip = set(getattr(args, "cal_skip", ()) or ())             # driver groups left exactly AS DELIVERED (11-10 sensitivity test)
        if skip:
            al, be = al.copy(), be.copy()
            groups = {"monthly": np.arange(0, 8), "ta_h": np.asarray(CH_TA), "hus": np.asarray(CH_HUS), "rsds": np.asarray(CH_RSDS),
                      "wind": np.asarray(CH_WIND), "soil": np.arange(129, 136), "ta_m": np.array([0]), "pr": np.array([1]), "tsl": np.arange(2, 8)}
            for gname in skip:
                al[groups[gname]] = 1.0; be[groups[gname]] = 0.0
        x = (x - be[None, None, :]) / al[None, None, :]
        x[:, :, 1] = np.clip(x[:, :, 1], 0.0, None)
        for chs in (CH_RSDS, CH_HUS, CH_WIND):
            x[:, :, chs] = np.clip(x[:, :, chs], 0.0, None)
        if getattr(args, "cal_align_ta", True) and not ({"ta_h", "ta_m", "monthly"} & skip):   # hourly air temperature: keep the shape, 24-h mean = ta_m
            x[:, :, CH_TA] = x[:, :, CH_TA] + (x[:, :, 0] - x[:, :, CH_TA].mean(axis=2))[:, :, None]
        cal_note = f"all channels mapped back through {args.cal_maps} (pr / rsds / hus / wind clipped at 0, ta_h mean aligned to ta_m)" + (f"; AS DELIVERED: {sorted(skip)}" if skip else "")
    cur, repairs = derive(x, args, floors)
    if cal_note:
        repairs["cal_maps"] = cal_note
    cur["co2"], co2_y0, co2_note = co2_path(cur["co2"], scenario, args)
    win = cur                                   # drivers of the yearly window files (spin-up climatology always from cur)
    if getattr(args, "window_monthly_twin", False):
        # 11-09 hypothesis: in the future runs ED's MONTHLY climate file (precipitation, temperature, soil temperatures:
        # soil water, PET, the fire dryness index) stayed MERRA-2, only the hourly mechanism drivers came from the ESM.
        # Window year k then reads the exact twin's training year k (+ window_monthly_twin_shift); cells without a twin
        # keep the calibrated test values.
        zc = np.load(ROOT / "work" / "test_cell_exact.npz")
        cell = int(zc["cell"][site])
        if cell >= 0:
            sh = int(getattr(args, "window_monthly_twin_shift", 0))
            tw = np.asarray(io.open_train_x()[cell, :, :, 0:8], dtype=np.float64)          # (40, 12, 8)
            idx = (np.arange(ny) + sh) % int(getattr(args, "window_monthly_twin_mod", tw.shape[0]))   # mod 35 = MERRA-2 1981-2015 recycled
            tw_use = tw[idx]
            clim = getattr(args, "window_monthly_twin_clim", None)
            if clim:                                   # one averaged MERRA-2 year for every window year (constant fire dryness)
                tw_use = np.repeat(tw[int(clim[0]):int(clim[1])].mean(axis=0, keepdims=True), ny, axis=0)
            win = dict(cur)
            win["ta_m"], win["pr"], win["tsl"] = tw_use[:, :, 0], tw_use[:, :, 1], tw_use[:, :, 2:8]
            win["soil_t_hourly"] = np.repeat(tw_use[:, :, 2:3], 24, axis=2)
            repairs["window_monthly_twin"] = (f"yearly climate files (ta_m, pr, tsl, hourly soil T) from training cell {cell}, " +
                                              (f"the mean of training years {clim[0]}-{clim[1]} in every window year" if clim else f"year shift {sh}"))
    if not getattr(args, "window_repair", True):
        raw_args = argparse.Namespace(**vars(args)); raw_args.repair = False; raw_args.repair_rsds_only = False
        win, _ = derive(x, raw_args, floors)
        win["co2"] = cur["co2"]
        repairs["window"] = "yearly window files from the channels AS DELIVERED (rsds floors, negative hus, ta_h offsets kept); spin-up climatology from the repaired channels"
    spin_src = [cur]
    if args.spinup_both:
        other = "ssp585" if scenario == "ssp126" else "ssp126"
        oth, _ = derive(np.asarray(io.open_test(other)[0][site], dtype=np.float64), args, floors)
        oth["co2"], _, _ = co2_path(oth["co2"], other, args)
        spin_src.append(oth)
    ny_spin = args.spinup_years if args.spinup_years > 0 else ny
    twin = None
    if getattr(args, "spinup_twin", False):
        # spin-up forcing = the 40-year MERRA-2 climatology (work/cache/x_season.npy) of the training site whose own
        # climatology is nearest to this site's window climatology (the lab's seed protocol uses MERRA-2, not the ESM)
        twin = train_twin(cur, floors, args)

    start = int(getattr(args, "spinup_start", 0))   # first window-year index of the spin-up climatology (training: 30 = 2011)
    shift_years = float(getattr(args, "spinup_shift", 0.0))
    shift_keys = ("ta_m", "tsl", "ta_h", "soil_t_hourly", "hus")   # temperatures additive, hus relative; pr/rsds/wind untouched
    shift_applied: dict[str, float] = {}

    def trend_per_year(key: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell least-squares slope over all window years of the scenario-mean series, and its 40-year mean."""
        series = np.mean([d[key] for d in spin_src], axis=0)             # (ny, month[, hour|layer])
        t = np.arange(series.shape[0], dtype=np.float64); t -= t.mean()
        slope = np.tensordot(t, series - series.mean(axis=0), axes=(0, 0)) / float((t ** 2).sum())
        return slope, series.mean(axis=0)

    swap_keys = set(k for k in str(getattr(args, "spin_swap_twin", "")).split(",") if k)
    twin_swap = geo_twin(*exact_latlon(site)) if swap_keys else None      # diagnostic: single driver groups from the exact twin

    def spin_avg(key: str) -> np.ndarray:
        if twin is not None:
            return twin[key]
        if twin_swap is not None and key in swap_keys:
            return twin_swap[key]
        base = np.concatenate([d[key][start:start + ny_spin] for d in spin_src], axis=0).mean(axis=0)
        if shift_years != 0.0 and key in shift_keys:
            slope, mean40 = trend_per_year(key)
            if key == "hus":
                rel = shift_years * slope / np.maximum(mean40, 1e-9)
                base = np.clip(base * (1.0 + rel), 0.0, None)
                shift_applied[key] = float(np.mean(rel))                 # mean relative change
            else:
                base = base + shift_years * slope
                shift_applied[key] = float(np.mean(shift_years * slope))  # mean change in K
        return base

    if args.co2_spinup == "first":
        co2_avg = cur["co2"][0]
    elif args.co2_spinup == "mean":
        co2_avg = spin_avg("co2")
    else:
        co2_avg = np.full((12, 24), args.co2_spinup_fixed)
    if getattr(args, "co2_spinup_by_scen", None):               # scenario-specific spin-up CO2 (e.g. the 2021-2100 mean)
        co2_avg = np.full((12, 24), float(args.co2_spinup_by_scen[scenario]))
    premult = 390.0 / 280.0 if args.co2_premultiply else 1.0

    lat, lon, fit = estimate_lat_lon(cur["rsds"])
    mode = getattr(args, "latlon_mode", "v1")
    if latlon is None and mode == "v2":            # ED's monthly PET (fire dryness index) uses the latitude band
        lat, lon = estimate_lat_lon_v2(cur["rsds"])
    elif latlon is None and mode == "nex3":        # test sites: NEX-GDDP weather-anomaly geolocation
        lat, lon = nex3_latlon(site)
    elif latlon is None and mode == "exact":       # test sites: the exact training cell (soil-channel identity)
        lat, lon = exact_latlon(site)
    elif latlon is None and mode == "train_true":  # training sites: the true mask2d coordinates
        tll = np.load(ROOT / "work" / "edv3" / "train_latlon.npy")
        lat, lon = float(tll[site, 0]), float(tll[site, 1])
    if latlon is not None:
        lat, lon = latlon
    if getattr(args, "spinup_geo_twin", False):
        # spin-up climatology from the GEOGRAPHICALLY nearest training site (2011-2020 decade), not from the ESM window
        lat2, lon2 = nex3_latlon(site) if mode == "nex3" else exact_latlon(site) if mode == "exact" else estimate_lat_lon_v2(cur["rsds"])
        twin = geo_twin(lat2, lon2, years=tuple(getattr(args, "twin_years", (30, 40))))
        twin["est_lat_v2"], twin["est_lon_v2"] = lat2, lon2
        twin["t_diff_K"] = float(twin["ta_m"].mean() - cur["ta_m"][:10].mean())      # twin decade minus the test's first decade
        twin["twin_dist"] = twin["twin_dist_km"]
        repairs["spinup"] = (f"geographic twin: training site {twin['twin_site']} at {twin['twin_dist_km']:.0f} km "
                             f"(est {lat2:.2f}, {lon2:.2f}; twin {twin['twin_lat']:.2f}, {twin['twin_lon']:.2f}), years 30-39; T diff {twin['t_diff_K']:+.2f} K")
    cell_area = (111.2e3 * 0.5) ** 2 * np.cos(np.radians(lat))
    lats = np.array([lat, lat - 0.5]); lons = np.array([lon, min(lon + 0.5, 179.75)])
    mask = np.zeros((2, 2), dtype=np.int8); mask[0, 0] = 1
    area = np.zeros((2, 2)); area[0, 0] = cell_area
    write_nc(out / "gridspec.nc", {"lat": 2, "lon": 2}, {
        "lat": (("lat",), lats, "d", {"units": "degrees_north"}),
        "lon": (("lon",), lons, "d", {"units": "degrees_east"}),
        "wtr_ice_frac": (("lat", "lon"), np.zeros((2, 2)), "d", {}),
        "mask": (("lat", "lon"), mask, "b", {}),
        "grid_cell_area": (("lat", "lon"), area, "d", {"units": "m2"}),
    })

    soil = {k: float(x[0, 0, c]) for k, c in SOIL_CH.items()}
    if getattr(args, "ksat_scale", 1.0) != 1.0:
        soil["k_sat"] *= args.ksat_scale; repairs["ksat_scale"] = f"k_sat x {args.ksat_scale}"
    write_nc(out / "soil_mvg.nc", {"lat": 2, "lon": 2}, {
        k: (("lat", "lon"), shim_grid(np.array(soil[k])), "f", {}) for k in ("soil_depth", "k_sat", "L", "m", "residual_theta", "saturated_theta")
    })
    write_nc(out / "soil_mstmip.nc", {"lat": 2, "lon": 2}, {
        "soil_depth": (("lat", "lon"), shim_grid(np.array(soil["soil_depth"])), "f", {}),
        "soil_theta_max": (("lat", "lon"), shim_grid(np.array(soil["saturated_theta"])), "f", {}),
        "soil_k_sat": (("lat", "lon"), shim_grid(np.array(soil["k_sat"])), "f", {}),
        "soil_tau": (("lat", "lon"), shim_grid(np.array(4.0)), "f", {"note": "nominal; unused with SOILGRIDS_SCHEME=1"}),
    })

    def climate_vars(t, p, ts):
        v = {"precipitation": (("time", "lat", "lon"), shim_grid(p), "f", {"units": "mm/month"}),
             "temperature": (("time", "lat", "lon"), shim_grid(t), "f", {"units": "K"})}
        for k in range(5):
            v[f"soil_temp{k + 1}"] = (("time", "lat", "lon"), shim_grid(ts[:, k]), "f", {"units": "K"})
        return v

    def premech_vars(t, q, sw, w, ts):
        flat = lambda a: a.reshape(288)
        return {"AirTemp": (("time", "lat", "lon"), shim_grid(flat(t)), "d", {"units": "K"}),
                "specific_humidity": (("time", "lat", "lon"), shim_grid(flat(q)), "d", {"units": "kg/kg"}),
                "swdown": (("time", "lat", "lon"), shim_grid(flat(sw)), "d", {"units": "W/m2"}),
                "wind": (("time", "lat", "lon"), shim_grid(flat(w)), "d", {"units": "m/s"}),
                "SoilTemp": (("time", "lat", "lon"), shim_grid(flat(ts)), "d", {"units": "K", "note": "monthly tsl1 repeated over hours"})}

    # file k = 0: "year 0" (the organisers' y0 is the December state of one scenario year after the common seed
    # states; proxy = the first window year's forcing at the year-0 CO2 level); k = 1..40: the window years;
    # k = 41: copy of the last year (ED reloads it before printing December of year 40).
    year_plan = [(0, co2_y0)] + [(k, cur["co2"][k]) for k in range(ny)] + [(ny - 1, cur["co2"][ny - 1])]
    for k, (yi, co2_cell) in enumerate(year_plan):
        yr = args.first_year + k
        write_nc(out / f"climate_{yr}.nc", {"time": 12, "lat": 2, "lon": 2}, climate_vars(win["ta_m"][yi], win["pr"][yi], win["tsl"][yi]))
        write_nc(out / f"premech_{yr}.nc", {"time": 288, "lat": 2, "lon": 2}, premech_vars(win["ta_h"][yi], win["hus"][yi], win["rsds"][yi], win["wind"][yi], win["soil_t_hourly"][yi]))
        write_nc(out / f"co2_{yr}.nc", {"time": 288, "lat": 2, "lon": 2},
                 {"co2_ambient": (("time", "lat", "lon"), shim_grid(premult * np.asarray(co2_cell).reshape(288)), "d", {"units": "ppm"})})
    if getattr(args, "spinup_cycle", False):
        # spin-up on CYCLED YEARLY WEATHER: the spin-up years of every scenario in spin_src as yearly files
        # spin_climate_/spin_premech_/spin_co2_<label>.nc, labels first_year.., CO2 fixed at the spin-up level
        k = 0
        cyc_src, cyc_years = spin_src, range(start, start + ny_spin)
        if getattr(args, "spinup_twin_cycle", None):
            # lookup on the exact twin's REAL MERRA-2 weather years a..b-1 (training forcing), cycled, instead of their mean
            a_, b_ = (int(v) for v in args.spinup_twin_cycle)
            j_ = geo_twin(*exact_latlon(site))["twin_site"]
            Tt = np.asarray(io.open_train_x()[j_, a_:b_], dtype=np.float64)             # (years, 12, 136)
            cyc_src = [{"ta_m": Tt[:, :, 0], "pr": Tt[:, :, 1], "tsl": Tt[:, :, 2:8], "hus": Tt[:, :, CH_HUS], "ta_h": Tt[:, :, CH_TA], "rsds": Tt[:, :, CH_RSDS],
                        "wind": Tt[:, :, CH_WIND], "soil_t_hourly": np.repeat(Tt[:, :, 2:3], 24, axis=2)}]
            cyc_years = range(0, b_ - a_)
        n_cycle_files = len(cyc_src) * len(cyc_years)
        for d in cyc_src:
            for yi in cyc_years:
                yr = args.first_year + k
                write_nc(out / f"spin_climate_{yr}.nc", {"time": 12, "lat": 2, "lon": 2}, climate_vars(d["ta_m"][yi], d["pr"][yi], d["tsl"][yi]))
                write_nc(out / f"spin_premech_{yr}.nc", {"time": 288, "lat": 2, "lon": 2}, premech_vars(d["ta_h"][yi], d["hus"][yi], d["rsds"][yi], d["wind"][yi], d["soil_t_hourly"][yi]))
                write_nc(out / f"spin_co2_{yr}.nc", {"time": 288, "lat": 2, "lon": 2},
                         {"co2_ambient": (("time", "lat", "lon"), shim_grid(premult * np.asarray(co2_avg).reshape(288)), "d", {"units": "ppm"})})
                k += 1
        # one spare copy beyond the cycle is not needed: the cycle wraps
    write_nc(out / "climate_avg.nc", {"time": 12, "lat": 2, "lon": 2}, climate_vars(spin_avg("ta_m"), spin_avg("pr"), spin_avg("tsl")))
    write_nc(out / "premech_avg.nc", {"time": 288, "lat": 2, "lon": 2},
             premech_vars(spin_avg("ta_h"), spin_avg("hus"), spin_avg("rsds"), spin_avg("wind"), spin_avg("soil_t_hourly")))
    write_nc(out / "co2_avg.nc", {"time": 288, "lat": 2, "lon": 2},
             {"co2_ambient": (("time", "lat", "lon"), shim_grid(premult * co2_avg.reshape(288)), "d", {"units": "ppm"})})

    name = f"s{site:05d}"
    (out / "sois.txt").write_text(f"NUM_SOIS   1\n\n\n{name}    {lat:.4f}  {lon:.4f}\n")
    (out / "ED_io.cfg").write_text(ED_IO_TEMPLATE.format(d=str(out.resolve()), y0=years[0], fire_off=1))

    dec = DECODE_MULT * np.clip(y0 - FLOORS, 0.0, None)
    meta = {
        "site": site, "scenario": scenario, "lat_est": lat, "lon_est": lon, "lat_fit_corr": fit,
        "years": [int(y) for y in years], "treefall_rate_dst": float(x[0, 0, 32]), "soil": soil,
        "spinup_climatology": {"years_per_scenario": ny_spin, "start_index": start, "scenarios": [scenario] + (["both"] if args.spinup_both else []),
                               "cycle_files": n_cycle_files if getattr(args, "spinup_cycle", False) else 0,
                               "shift_years": shift_years, "shift_applied": shift_applied,
                               "train_twin": None if twin is None else {"site": twin["twin_site"], "dist": twin["twin_dist"]},
                               "geo_twin": None if twin is None or "twin_dist_km" not in twin else
                               {k: twin[k] for k in ("twin_site", "twin_dist_km", "twin_lat", "twin_lon", "est_lat_v2", "est_lon_v2", "t_diff_K", "twin_years")}},
        "co2": {"mode": args.co2, "note": co2_note, "premultiplied_390_over_280": bool(args.co2_premultiply),
                "annual_mean_written_ppm": (premult * cur["co2"].mean(axis=(1, 2))).tolist(), "spinup_level_ppm": float(premult * co2_avg.mean()),
                "year0_level_ppm": float(premult * np.mean(co2_y0))},
        "file_plan": "label first_year = year 0 (first window year's forcing, year-0 CO2); first_year+1..+40 = window; +41 = copy",
        "repairs": repairs,
        "y0_raw": {n: y0[:, i].tolist() for i, n in enumerate(STATE_NAMES)},
        "y0_decoded": {n: dec[:, i].tolist() for i, n in enumerate(STATE_NAMES)},
        "y0_decode": "height/agb/soil/gpp/npp/rh = 1.14114 x (y0 - floor), lai = 1.0 x (y0 - floor)",
        "seed_ages": list(SEED_AGES),
    }
    json.dump(meta, open(out / "site_meta.json", "w"), indent=1)
    return meta


ED_IO_TEMPLATE = """// ED_io.cfg for one CarbonGlobe test site (written by scripts/ed/make_site_inputs.py)
output_base_path = "{d}/ED_outputs/";
which_mech_to_use = "cgb_spinup";     // or "cgb_window" for the 40-year run from a restart

cgb_spinup =   // averaged year repeated (single_year); CO2 file taken as written with the ED_CO2_AS_IS patch
{{
   gridspec        = "{d}/gridspec.nc";
   climate_file    = " ";
   climate_file_avg = "{d}/climate_avg.nc";
   forecast_climate_file = "";
   soil_file       = "{d}/soil_mstmip.nc";
   soilgrids_file  = "{d}/soil_mvg.nc";
   lu_file         = " ";
   crop_calendar_file = " ";
   lu_init_c_file  = " ";
   gfedbf_file_avg = " ";
   gfedbf_file     = " ";
   single_year     = 1;
   do_yearly_mech  = 0;
   m_int           = 1;
   m_string        = 0;
   do_FTS          = 0;
   C3_FILE = " "; C4_FILE = " "; QAIR_FILE = " "; TAIR_FILE = " "; SW_FILE = " ";
   PREMECH         = " ";
   PREMECH_avg     = "{d}/premech_avg.nc";
   forecast_PREMECH = "";
   PREMECH_CO2     = " ";
   PREMECH_CO2_avg = "{d}/co2_avg.nc";
}};

cgb_window =   // yearly files climate_<year>.nc / premech_<year>.nc / co2_<year>.nc, <year> = {y0} .. {y0}+40
{{
   gridspec        = "{d}/gridspec.nc";
   climate_file    = "{d}/climate_";
   climate_file_avg = "{d}/climate_avg.nc";
   forecast_climate_file = "";
   soil_file       = "{d}/soil_mstmip.nc";
   soilgrids_file  = "{d}/soil_mvg.nc";
   lu_file         = " ";
   crop_calendar_file = " ";
   lu_init_c_file  = "";
   gfedbf_file_avg = " ";
   gfedbf_file     = " ";
   single_year     = 0;
   do_yearly_mech  = 1;
   m_int           = 1;
   m_string        = 0;
   do_FTS          = 0;
   C3_FILE = " "; C4_FILE = " "; QAIR_FILE = " "; TAIR_FILE = " "; SW_FILE = " ";
   PREMECH         = "{d}/premech_";
   PREMECH_avg     = "{d}/premech_avg.nc";
   forecast_PREMECH = "";
   PREMECH_CO2     = "{d}/co2_";
   PREMECH_CO2_avg = "{d}/co2_avg.nc";
}};

cgb_spinw =   // spin-up on cycled yearly weather: spin_climate_<y>.nc / spin_premech_<y>.nc / spin_co2_<y>.nc, y = {y0} .. (ED_YEAR_CYCLE files)
{{
   gridspec        = "{d}/gridspec.nc";
   climate_file    = "{d}/spin_climate_";
   climate_file_avg = "{d}/climate_avg.nc";
   forecast_climate_file = "";
   soil_file       = "{d}/soil_mstmip.nc";
   soilgrids_file  = "{d}/soil_mvg.nc";
   lu_file         = " ";
   crop_calendar_file = " ";
   lu_init_c_file  = "";
   gfedbf_file_avg = " ";
   gfedbf_file     = " ";
   single_year     = 0;
   do_yearly_mech  = 1;
   m_int           = 1;
   m_string        = 0;
   do_FTS          = 0;
   C3_FILE = " "; C4_FILE = " "; QAIR_FILE = " "; TAIR_FILE = " "; SW_FILE = " ";
   PREMECH         = "{d}/spin_premech_";
   PREMECH_avg     = "{d}/premech_avg.nc";
   forecast_PREMECH = "";
   PREMECH_CO2     = "{d}/spin_co2_";
   PREMECH_CO2_avg = "{d}/co2_avg.nc";
}};

is_site = 1;
region  = "GLOBAL";
GLOBAL = {{ LATMIN = -90.0; LATMAX = 90.0; LONMIN = -180.0; LONMAX = 180.0; }};

print_output_files = 1;
print_system_state = 1;
print_ss_freq      = 12;     // yearly restart dumps (N_SUB units): the seed-age states
cd_file = 0; fp_file = 0; long_patch_file = 0; long_cd_file = 0; long_fp_file = 0; long_cohort_file = 0;   // site-level files only (the long patch/cohort files are GBs per site)
// ED_params.cfg for this run: fire_off = {fire_off}; env ED_TREEFALL=0.012 (test dst channel); ED_CO2_AS_IS=1
"""


# ----------------------------------------------------------------------------- validation
def validate(site_dir: Path, repaired: bool) -> list[str]:
    problems: list[str] = []
    sw_lo = 0 if repaired else -30
    ranges = {"AirTemp": (180, 335), "specific_humidity": (-0.001, 0.05), "swdown": (sw_lo, 1500), "wind": (0, 50), "SoilTemp": (180, 335),
              "co2_ambient": (200, 1500), "precipitation": (0, 4000), "temperature": (180, 335), "soil_temp1": (180, 335),
              "soil_temp2": (180, 335), "soil_temp3": (180, 335), "soil_temp4": (180, 335), "soil_temp5": (180, 335),
              "soil_depth": (10, 1001), "k_sat": (1, 1e8), "L": (-20, 20), "m": (0, 1), "residual_theta": (0, 0.5), "saturated_theta": (0.2, 0.9),
              "soil_theta_max": (0.2, 0.9), "soil_k_sat": (1, 1e8), "soil_tau": (0, 50), "lat": (-90, 90), "lon": (-180, 180),
              "grid_cell_area": (0, 1e10), "wtr_ice_frac": (0, 1), "mask": (0, 1)}
    for f in sorted(site_dir.glob("*.nc")):
        with netcdf_file(str(f), "r", mmap=False) as nc:
            for name, var in nc.variables.items():
                arr = np.asarray(var[:], dtype=np.float64)
                if arr.ndim >= 2:
                    arr = arr[..., 0, 0]
                if not np.isfinite(arr).all():
                    problems.append(f"{f.name}:{name} has non-finite values")
                lo, hi = ranges.get(name, (-np.inf, np.inf))
                if arr.min() < lo or arr.max() > hi:
                    problems.append(f"{f.name}:{name} out of range [{arr.min():.4g}, {arr.max():.4g}] vs [{lo}, {hi}]")
            if "time" in nc.dimensions and f.name.startswith("premech") and nc.dimensions["time"] != 288:
                problems.append(f"{f.name}: time dim {nc.dimensions['time']} != 288")
    return problems


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", default=str(ROOT / "work" / "edv3" / "sites"))
    ap.add_argument("--first-year", type=int, default=1981, help="label of the first window year in the file names")
    ap.add_argument("--co2", default="asis", choices=["asis", "real", "fixed"])
    ap.add_argument("--co2-json", default=str(ROOT / "work" / "edv3" / "co2_real_paths.json"), help="exact decompression (provenance agent) used by --co2 real")
    ap.add_argument("--co2-fixed", type=float, default=400.0)
    ap.add_argument("--co2-real-start", type=float, default=400.0)
    ap.add_argument("--co2-real-rise126", type=float, default=52.0)
    ap.add_argument("--co2-real-rise585", type=float, default=190.0)
    ap.add_argument("--co2-spinup", default="first", choices=["first", "mean", "fixed"])
    ap.add_argument("--co2-spinup-fixed", type=float, default=280.0)
    ap.add_argument("--co2-premultiply", action="store_true")
    ap.add_argument("--ta-amp-factor", type=float, default=1.0)
    ap.add_argument("--repair", action="store_true", help="apply the hourly-channel repairs (default: channels as delivered)")
    ap.add_argument("--spinup-years", type=int, default=0, help="average the first N window years for the spin-up climatology (0 = all 40)")
    ap.add_argument("--spinup-both", action="store_true", help="average both scenarios for the spin-up climatology")
    ap.add_argument("--spinup-shift", type=float, default=0.0,
                    help="move the spin-up climatology along the per-cell linear trend of the 40 window years by this many years "
                         "(ta_m, tsl, ta_h, soil T additive; hus by the relative trend; pr, rsds, wind unchanged)")
    ap.add_argument("--spinup-geo-twin", action="store_true",
                    help="spin-up climatology = training window years 30-39 (2011-2020) of the training site nearest to the test site's "
                         "estimated (lat, lon) (estimate_lat_lon_v2); soil and CO2 stay the test site's")
    ap.add_argument("--no-window-repair", dest="window_repair", action="store_false",
                    help="write the yearly window files from the channels as delivered (spin-up climatology still from the repaired channels)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="comma-separated test site indices (0..6170)")
    ap.add_argument("--scenario", default="both", choices=["ssp126", "ssp585", "both"])
    add_arguments(ap)
    args = ap.parse_args()
    sites = [int(s) for s in args.sites.split(",")]
    scenarios = list(io.SCENARIOS) if args.scenario == "both" else [args.scenario]
    floors = rsds_channel_floors()
    all_ok = True
    for site in sites:
        for sc in scenarios:
            meta = build_site(site, sc, args, floors)
            d = Path(args.out) / f"site{site:05d}_{sc}"
            problems = validate(d, args.repair)
            print(f"{d.name}: {len(list(d.glob('*.nc')))} netCDF files; lat_est {meta['lat_est']:+.1f} lon_est {meta['lon_est']:+.1f}; "
                  f"CO2 written {meta['co2']['annual_mean_written_ppm'][0]:.1f} -> {meta['co2']['annual_mean_written_ppm'][-1]:.1f} ppm (spin-up {meta['co2']['spinup_level_ppm']:.1f}); "
                  f"spin-up climatology {meta['spinup_climatology']}; y0 decoded age-500 height {meta['y0_decoded']['height'][-1]:.2f} m agb {meta['y0_decoded']['agb'][-1]:.2f}")
            if problems:
                all_ok = False
                for p in problems:
                    print("   PROBLEM:", p)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

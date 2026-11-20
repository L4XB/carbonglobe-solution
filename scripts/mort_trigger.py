#!/usr/bin/env python
"""What in the monthly forcing precedes an old-stand collapse?

``mort_census.py`` established that a collapse is a site-year event shared across
seed ages, and ``mort_calendar.py`` that the busy years are El Nino droughts with
the right regional fingerprints. This script scores candidate drought and heat
indices, built from the monthly cache of ``mort_cache_monthly.py``, on how well
they separate event site-years from the rest -- and, more to the point, on how
well they pick *which year* at an event-prone site.

Label: a site-year is an event when at least 4 of the 8 old seed ages (>= 150)
lost more than 1 kgC m-2 that year; sites must be vegetated at all 8 old ages.

Indices (all ``(n_sites, 40)``, larger = more stress), each relative to the site's
own 40-year monthly climatology so that a dry season is not mistaken for a drought:

  spi{w}        minimum, over the 12 windows ending in the year, of the w-month
                precipitation sum divided by its climatological sum, negated
  dryrun        longest run of months with precip < 50 % of the calendar month's
                climatology, ending in the year
  tanom         annual mean air-temperature anomaly (K); tmax3 the hottest
                3-month anomaly
  vpd{w}        maximum w-month mean anomaly of a daytime VPD proxy built from the
                daily-max temperature and daily-min humidity drivers
  wb{w}         minimum, over windows ending in the year, of the w-month sum of the
                water-balance anomaly (P - PET, Priestley-Taylor PET from shortwave
                and temperature), negated
  bucket        minimum level of a 150 mm soil bucket driven by P - PET, negated,
                relative to the site's climatological minimum
  *_lag1        the same index one forcing year earlier
  *_max2        the larger of this year's and last year's value

Scores: AUC over site-years (event-prone sites, and all vegetated sites), the
within-site rank of the event year (P(top 3 of 40); random 0.075), and the
Spearman correlation with the mean relative drop across old ages.

    ./.venv/bin/python scripts/mort_trigger.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
METRICS = io.ROOT / "work" / "metrics"


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) along the last axis, ties averaged."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, axis=-1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    idx = np.arange(x.shape[-1], dtype=np.float64) + 1.0
    np.put_along_axis(ranks, order, np.broadcast_to(idx, x.shape).copy(), axis=-1)
    # tie handling: average ranks within equal values
    xs = np.take_along_axis(x, order, axis=-1)
    if x.ndim == 1:
        ties = np.flatnonzero(np.diff(xs) == 0)
        if ties.size:
            # group consecutive ties
            grp = np.cumsum(np.r_[1, np.diff(xs) != 0])
            sums = np.bincount(grp, weights=idx)
            cnts = np.bincount(grp)
            avg = sums / np.maximum(cnts, 1)
            ranks[order] = avg[grp]
    return ranks


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC of a 1-D score against a boolean label."""
    r = rankdata(score)
    n_pos = int(label.sum())
    n_neg = label.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """``(n, T)`` -> ``(n, T)``: sum of the w values ending at each position; NaN before w-1."""
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    out = np.full_like(cs, np.nan)
    out[:, w - 1 :] = cs[:, w - 1 :] - np.concatenate([np.zeros((x.shape[0], 1)), cs[:, :-w]], axis=1)
    return out


def year_min(x480: np.ndarray) -> np.ndarray:
    """``(n, 480)`` -> ``(n, 40)``: minimum over the 12 months of each forcing year (NaN-aware)."""
    return np.nanmin(x480.reshape(x480.shape[0], 40, 12), axis=2)


def year_max(x480: np.ndarray) -> np.ndarray:
    return np.nanmax(x480.reshape(x480.shape[0], 40, 12), axis=2)


def sat_vp(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (Pa) over water, Magnus form, from temperature in K."""
    tc = t_k - 273.15
    return 610.94 * np.exp(17.625 * tc / (tc + 243.04))


def vp_from_q(q: np.ndarray, p: float = 101325.0) -> np.ndarray:
    """Vapour pressure (Pa) from specific humidity at pressure ``p``."""
    return q * p / (0.622 + 0.378 * q)


def pet_priestley_taylor(t_k: np.ndarray, rsds: np.ndarray) -> np.ndarray:
    """Potential evapotranspiration, mm per month, from mean temperature and shortwave.

    Net radiation is taken as 0.75 x shortwave (a crude all-sky albedo/longwave
    allowance); the constant matters for the scale of P - PET, not for ranking years.
    """
    es = sat_vp(t_k)
    tc = t_k - 273.15
    delta = 4098.0 * es / (tc + 237.3) ** 2  # Pa/K
    gamma = 66.0  # Pa/K
    lam = 2.45e6  # J/kg
    rn = 0.75 * rsds  # W/m2
    mm_day = 1.26 * delta / (delta + gamma) * rn * 86400.0 / lam
    return mm_day * 30.4


def build_indices(m: np.ndarray) -> dict[str, np.ndarray]:
    """All candidate indices, ``(n_sites, 40)`` each, larger = more stress."""
    n = m.shape[0]
    pr = m[..., 1].reshape(n, 480).astype(np.float64)
    ta = m[..., 0].reshape(n, 480).astype(np.float64)
    ta_h = m[..., 4].reshape(n, 480).astype(np.float64)
    rsds = m[..., 5].reshape(n, 480).astype(np.float64)
    tmax = m[..., 7].reshape(n, 480).astype(np.float64)
    qmin = m[..., 8].reshape(n, 480).astype(np.float64)
    q = m[..., 3].reshape(n, 480).astype(np.float64)

    def clim(x480: np.ndarray) -> np.ndarray:
        c = x480.reshape(n, 40, 12).mean(axis=1)  # (n, 12)
        return np.tile(c, (1, 40))

    out: dict[str, np.ndarray] = {}
    pr_c = clim(pr)
    for w in (1, 3, 6, 12, 24, 36):
        num = rolling_sum(pr, w)
        den = rolling_sum(pr_c, w)
        ratio = num / np.where(den > 1e-9, den, np.nan)
        out[f"spi{w}"] = -year_min(ratio)
    dry = pr < 0.5 * pr_c
    run = np.zeros((n, 480))
    cur = np.zeros(n)
    for i in range(480):
        cur = np.where(dry[:, i], cur + 1, 0.0)
        run[:, i] = cur
    out["dryrun"] = year_max(run)

    ta_c = clim(ta)
    ta_an = ta - ta_c
    out["tanom"] = ta_an.reshape(n, 40, 12).mean(axis=2)
    out["tmax3"] = year_max(rolling_sum(ta_an, 3) / 3.0)
    out["tmax1"] = year_max(ta_an)

    vpd_day = sat_vp(tmax) - vp_from_q(qmin)
    vpd_day = np.maximum(vpd_day, 0.0)
    vpd_c = clim(vpd_day)
    vpd_an = vpd_day - vpd_c
    for w in (1, 3, 6, 12):
        out[f"vpd{w}"] = year_max(rolling_sum(vpd_an, w) / w)
    out["vpdmean"] = vpd_an.reshape(n, 40, 12).mean(axis=2)
    # relative humidity deficit as an alternative dryness of the air
    rh = vp_from_q(q) / sat_vp(ta_h)
    rh_an = rh - clim(rh)
    out["rhdef3"] = -year_min(rolling_sum(rh_an, 3) / 3.0)

    pet = pet_priestley_taylor(ta_h, rsds)
    wb = pr - pet
    wb_an = wb - clim(wb)
    for w in (3, 6, 12, 24):
        out[f"wb{w}"] = -year_min(rolling_sum(wb_an, w))
    # a 150 mm bucket driven by the raw balance, level relative to the site's usual minimum
    smax = 150.0
    level = np.zeros((n, 480))
    s = np.full(n, smax / 2)
    for i in range(480):
        s = np.clip(s + wb[:, i], 0.0, smax)
        level[:, i] = s
    ymin = year_min(level)  # (n, 40)
    out["bucket"] = -(ymin - np.median(ymin, axis=1, keepdims=True))
    out["bucket_abs"] = -ymin
    # the PET/precip ratio scale, for the sanity print
    out["_pet_over_p"] = np.array([[pet.mean() / pr.mean()]])
    return out


def add_lags(ix: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(ix)
    for k, v in ix.items():
        if k.startswith("_"):
            continue
        lag = np.full_like(v, np.nan)
        lag[:, 1:] = v[:, :-1]
        out[f"{k}_lag1"] = lag
        out[f"{k}_max2"] = np.fmax(v, lag)
    return out


def main() -> None:
    t0 = time.time()
    traj = cache.open_traj()
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    agb = np.asarray(traj[old, :, :, 1], dtype=np.float32)  # (8, N, 41)
    ev = mort.events(agb)  # (8, N, 40)
    veg_all = (agb[:, :, 0] > mort.BARE_AGB).all(axis=0)
    k = ev.sum(axis=0)
    site_ev = (k >= 4) & veg_all[:, None]  # (N, 40)
    rel = (agb[:, :, :-1] - agb[:, :, 1:]) / np.where(agb[:, :, :-1] > mort.BARE_AGB, agb[:, :, :-1], np.nan)
    rel_drop = np.nanmean(rel, axis=0)  # (N, 40) mean relative drop across old ages
    prone = site_ev.any(axis=1)
    print(f"labels: {veg_all.sum()} sites vegetated at all old ages, {prone.sum()} event-prone, {site_ev.sum()} event site-years")

    m = np.asarray(mort.open_monthly("train"))
    print(f"monthly cache loaded {m.shape} in {time.time() - t0:.0f}s")
    ix = build_indices(m)
    print(f"  sanity: PET / precip over all site-months = {float(ix.pop('_pet_over_p')[0, 0]):.2f} (a global land value near 1-2 means the units are compatible)")
    ix = add_lags(ix)
    print(f"indices built: {len(ix)} in {time.time() - t0:.0f}s")

    # ---- scoring ------------------------------------------------------------
    rows = []
    lab_prone = site_ev[prone].ravel()
    lab_all = site_ev[veg_all].ravel()
    drop_prone = rel_drop[prone].ravel()
    ok_drop = np.isfinite(drop_prone)
    s_ev, t_ev = np.nonzero(site_ev)
    late = t_ev >= 3  # events after the initial three years
    for name, x in ix.items():
        xf = np.where(np.isfinite(x), x, -np.inf)
        xp = xf[prone].ravel()
        a_prone = auc(xp, lab_prone)
        a_all = auc(xf[veg_all].ravel(), lab_all)
        # within-site rank of the event year, descending (0 = most stressed year of the 40)
        rk = np.argsort(np.argsort(-xf, axis=1, kind="stable"), axis=1)
        r = rk[s_ev, t_ev]
        top3 = float((r < 3).mean())
        top3_late = float((r[late] < 3).mean())
        meanr = float(r.mean())
        sp = spearman(xp[ok_drop], drop_prone[ok_drop])
        rows.append((name, a_prone, a_all, top3, top3_late, meanr, sp))
    rows.sort(key=lambda r: -r[3])
    print(f"\n== index scores (sorted by within-site P(event year in the 3 most stressed of 40); random 0.075) ==")
    print(f"  {'index':14s} {'AUC prone':>9s} {'AUC all':>8s} {'P(top3)':>8s} {'top3 y>3':>9s} {'mean rank':>9s} {'spearman':>9s}")
    for name, a_p, a_a, t3, t3l, mr, sp in rows:
        print(f"  {name:14s} {a_p:9.3f} {a_a:8.3f} {t3:8.3f} {t3l:9.3f} {mr:9.1f} {sp:9.3f}")

    # ---- the best few, in detail --------------------------------------------
    best = [r[0] for r in rows[:4]]
    print("\n== threshold rules on the best indices, all vegetated sites (site-years) ==")
    base = float(lab_all.mean())
    print(f"  base rate of an event site-year: {base:.4f}")
    for name in best:
        x = np.where(np.isfinite(ix[name]), ix[name], -np.inf)[veg_all].ravel()
        print(f"  {name}:")
        for q in (0.80, 0.90, 0.95, 0.98, 0.99):
            thr = np.quantile(x[np.isfinite(x)], q)
            fire = x > thr
            prec = float(lab_all[fire].mean())
            rec = float(fire[lab_all].mean())
            print(f"    > q{q:.2f} ({thr:8.3f}): fires {fire.mean():.3f}  precision {prec:.3f} (lift x{prec / base:4.1f})  recall {rec:.3f}")
    # by big year: which index picks the year
    print("\n== per forcing year: fraction of that year's event sites whose index ranks it top-3 at the site ==")
    years = [1, 2, 3, 4, 6, 7, 11, 16, 17, 18, 24, 34, 35]
    print("  " + f"{'index':14s}" + " ".join(f"{y:>5d}" for y in years))
    for name in best + ["spi12", "tanom", "vpd3"]:
        xf = np.where(np.isfinite(ix[name]), ix[name], -np.inf)
        rk = np.argsort(np.argsort(-xf, axis=1, kind="stable"), axis=1)
        r = rk[s_ev, t_ev]
        vals = [float((r[t_ev == y - 1] < 3).mean()) for y in years]
        print("  " + f"{name:14s}" + " ".join(f"{v:5.2f}" for v in vals))
    # the flip side: at event-prone sites, what does the most stressed year look like when it is NOT an event year?
    name = best[0]
    xf = np.where(np.isfinite(ix[name]), ix[name], -np.inf)
    top_year = np.argmax(xf, axis=1)
    hit = site_ev[np.arange(xf.shape[0]), top_year]
    print(f"\n  {name}: at event-prone sites, the single most stressed year is an event year in {hit[prone].mean():.3f} of sites "
          f"(random {site_ev[prone].sum() / (prone.sum() * 40):.3f}); over all vegetated sites {hit[veg_all].mean():.3f}")

    payload = {
        "n_sites_veg": int(veg_all.sum()),
        "n_prone": int(prone.sum()),
        "n_event_site_years": int(site_ev.sum()),
        "scores": [dict(zip(("index", "auc_prone", "auc_all", "p_top3", "p_top3_late", "mean_rank", "spearman"), r)) for r in rows],
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "mort_trigger.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {METRICS / 'mort_trigger.json'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

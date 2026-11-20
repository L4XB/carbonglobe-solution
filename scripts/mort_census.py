#!/usr/bin/env python
"""Census of the single-year biomass collapses in old stands, over all 15 seed ages.

``notes/theory.md`` Sec. 4.3 measured, at two seed ages, that the old-stand agb loss
sits in single-year drops that cluster in forcing years and among grid neighbours.
This script asks the structural questions that decide whether those drops can be
modelled at all, before any trigger is looked for:

  1. Is "event" a real category? The distribution of the year-over-year agb change
     at old ages -- is there a gap between ordinary drift and the collapses, or is
     the 1 kgC threshold an arbitrary cut on a continuum?
  2. Are events shared across seed ages at the same site? The forcing is per site
     and identical for all 15 seed ages, so a forcing-driven collapse should hit
     every old stand at that site in the same year. A collapse that lands in
     different years at different seed ages is cohort dynamics, not weather.
  3. What does an event do to the other six variables in the event year, and to the
     stand afterwards?
  4. Survival: an event in forcing year t -- how much of the drop is still visible
     at year 40, as a function of t? Only year 40 is scored.
  5. Loss accounting: what share of the year-40 persistence loss at old ages sits on
     event sites, and on events in which years.

Everything is measured on ``work/cache/y_traj.npy`` (the full training set, month
axis dropped) in physical units; scaled where the metric is quoted.

    ./.venv/bin/python scripts/mort_census.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
METRICS = io.ROOT / "work" / "metrics"


def main() -> None:
    div = cache.scoring_divisors()
    agb_div = float(div[1])
    traj = cache.open_traj()  # (15, N, 41, 7)
    ages = np.array(io.SEED_AGES)
    old = np.array([j for j, a in enumerate(io.SEED_AGES) if a >= 150])
    agb = np.asarray(traj[:, :, :, 1], dtype=np.float32)  # (15, N, 41)
    n_sites = agb.shape[1]
    d = np.diff(agb, axis=2)  # (15, N, 40): d[:, :, t] = agb[t+1] - agb[t], forcing year t+1
    veg = agb[:, :, 0] > 0.5  # vegetated at year 0, per (age, site)
    payload: dict[str, object] = {}

    # ---- 1. is "event" a category? ------------------------------------------
    print("== 1. distribution of the annual agb change, seed ages >= 150, vegetated site-years ==")
    dd = d[old][veg[old]]  # (n_old_veg_sites, 40)
    flat = dd.ravel()
    qs = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    print("  quantiles:", " ".join(f"{q:.3f}:{np.quantile(flat, q):+.3f}" for q in qs))
    edges = np.array([-np.inf, -8, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, 0])
    hist, _ = np.histogram(flat, bins=edges)
    print("  count of site-years by drop size (kgC m-2):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        width = (hi - lo) if np.isfinite(lo) else np.nan
        print(f"    [{lo:>6.2f}, {hi:>6.2f})  {c:9d}   per unit width {c / width if np.isfinite(width) else float('nan'):12.0f}")
    # the same, relative to the stand's agb the year before
    prev = agb[old, :, :-1][veg[old]]
    rel = dd / np.where(prev > 0.5, prev, np.nan)
    rflat = rel[np.isfinite(rel)]
    redges = np.array([-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.075, -0.05, -0.025, 0])
    rhist, _ = np.histogram(rflat, bins=redges)
    print("  relative drop (fraction of previous year's agb):")
    for lo, hi, c in zip(redges[:-1], redges[1:], rhist):
        print(f"    [{lo:>6.3f}, {hi:>6.3f})  {c:9d}   per unit width {c / (hi - lo):12.0f}")
    payload["drop_hist_abs"] = {"edges": edges.tolist(), "counts": hist.tolist()}
    payload["drop_hist_rel"] = {"edges": redges.tolist(), "counts": rhist.tolist()}

    # ---- 2. cross-age coincidence --------------------------------------------
    ev = mort.events(agb, thr=mort.EVENT_THR)  # (15, N, 40) bool, vegetated only
    print(f"\n== 2. events (single-year agb drop > {mort.EVENT_THR} kgC m-2), by seed age ==")
    print(f"  {'age':>4} {'veg sites':>9} {'events':>7} {'rate/site-yr':>12} {'sites w/ ev':>11} {'busiest yr':>10} {'max/median':>10}")
    per_age_year = ev.sum(axis=1)  # (15, 40)
    for j, a in enumerate(io.SEED_AGES):
        py = per_age_year[j]
        nv = int(veg[j].sum())
        print(
            f"  {a:4d} {nv:9d} {int(ev[j].sum()):7d} {ev[j].sum() / max(nv * 40, 1):12.5f} "
            f"{int(ev[j].any(axis=1).sum()):11d} {int(np.argmax(py)) + 1:10d} {py.max() / max(np.median(py), 1):10.1f}"
        )
    payload["events_per_age_year"] = per_age_year.tolist()

    # among the 8 old ages: how many of them have an event at (site, year), given at least one does
    k = ev[old].sum(axis=0)  # (N, 40) number of old ages with an event at that site-year
    nveg_old = veg[old].sum(axis=0)  # (N,) number of old ages vegetated at that site
    both = (k > 0) & (nveg_old[:, None] == len(old))
    frac = k[both] / len(old)
    print(f"\n  site-years with >= 1 old-age event and all {len(old)} old ages vegetated: {both.sum()}")
    print("  distribution of the number of old ages (of 8) sharing the event:")
    cnt = np.bincount(k[both], minlength=len(old) + 1)[1:]
    for i, c in enumerate(cnt, start=1):
        print(f"    {i} of {len(old)}: {c:7d}  ({c / both.sum():.3f})")
    print(f"  mean share of old ages hit, given >= 1: {frac.mean():.3f}   (a forcing-driven event would be near 1, cohort dynamics near 1/8)")
    # pairwise: P(event at age b | event at age a), same site-year
    print("  P(event at age b | event at age a), same site and year, rows a / cols b:")
    hdr = "        " + " ".join(f"{io.SEED_AGES[j]:>5d}" for j in old)
    print(hdr)
    pair = np.zeros((len(old), len(old)))
    for i, ja in enumerate(old):
        ea = ev[ja]
        for m, jb in enumerate(old):
            pair[i, m] = ev[jb][ea].mean() if ea.any() else np.nan
        print(f"  {io.SEED_AGES[ja]:5d} " + " ".join(f"{pair[i, m]:5.2f}" for m in range(len(old))))
    payload["pair_cond_prob"] = pair.tolist()
    # does the event hit the young ages as well?
    young = np.array([j for j, a in enumerate(io.SEED_AGES) if a <= 100])
    any_old = ev[old].any(axis=0)
    print("  P(event at age a | >= 1 old-age event at that site-year), young/middle ages:")
    for j in young:
        e = ev[j][any_old & veg[j][:, None]]
        print(f"    age {io.SEED_AGES[j]:3d}: {e.mean():.3f}  (rate elsewhere {ev[j][~any_old & veg[j][:, None]].mean():.4f})")
    # a drop that is not an event by the fixed threshold may still be a relative collapse: the same at a relative threshold
    rel_all = d / np.where(agb[:, :, :-1] > 0.5, agb[:, :, :-1], np.nan)
    ev_rel = (rel_all < -mort.REL_THR) & veg[:, :, None]
    k_rel = ev_rel[old].sum(axis=0)
    both_rel = (k_rel > 0) & (nveg_old[:, None] == len(old))
    print(f"  the same coincidence with a relative threshold ({mort.REL_THR:.0%} of last year's agb): "
          f"mean share of old ages hit {(k_rel[both_rel] / len(old)).mean():.3f} over {both_rel.sum()} site-years")

    # ---- 3. what the event does -------------------------------------------
    print("\n== 3. the event's signature in the seven variables, seed age 500 ==")
    j500 = io.SEED_AGES.index(500)
    s, t = np.nonzero(ev[j500])  # event in forcing year t+1: state slot t -> t+1
    names = io.STATE_NAMES
    before = np.asarray(traj[j500, s, t, :], dtype=np.float64)
    at = np.asarray(traj[j500, s, t + 1, :], dtype=np.float64)
    ok_next = t + 2 <= 40
    after = np.full_like(at, np.nan)
    after[ok_next] = np.asarray(traj[j500, s[ok_next], t[ok_next] + 2, :], dtype=np.float64)
    print(f"  events: {s.size};  median relative change (slot t+1 vs t), and the year after (t+2 vs t+1):")
    for v, n in enumerate(names):
        r1 = (at[:, v] - before[:, v]) / np.where(np.abs(before[:, v]) > 1e-6, before[:, v], np.nan)
        r2 = (after[:, v] - at[:, v]) / np.where(np.abs(at[:, v]) > 1e-6, at[:, v], np.nan)
        print(f"    {n:7s}: event year {np.nanmedian(r1):+.3f} [IQR {np.nanpercentile(r1, 25):+.3f}, {np.nanpercentile(r1, 75):+.3f}]   "
              f"year after {np.nanmedian(r2):+.3f}")
    drop = before[:, 1] - at[:, 1]
    print(f"  drop size: median {np.median(drop):.2f} kgC m-2, IQR [{np.percentile(drop, 25):.2f}, {np.percentile(drop, 75):.2f}], max {drop.max():.2f}")
    print(f"  drop as a fraction of agb before: median {np.median(drop / before[:, 1]):.3f}, IQR [{np.percentile(drop / before[:, 1], 25):.3f}, {np.percentile(drop / before[:, 1], 75):.3f}]")
    # dependence on the state before
    print("  drop fraction by agb-before bin:")
    bins = [0.5, 2, 4, 6, 8, 10, 12, 15, 20, 40]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (before[:, 1] >= lo) & (before[:, 1] < hi)
        if m.sum() >= 20:
            print(f"    agb0 in [{lo:>4.1f},{hi:>4.1f}): n {m.sum():6d}  median drop {np.median(drop[m]):5.2f}  fraction {np.median(drop[m] / before[m, 1]):.3f}  "
                  f"height rel {np.median((at[m, 0] - before[m, 0]) / before[m, 0]):+.3f}  lai rel {np.median((at[m, 3] - before[m, 3]) / np.maximum(before[m, 3], 1e-6)):+.3f}")
    # the flux in the event year relative to the site's own 40-year median flux
    gpp_all = np.asarray(traj[j500, :, 1:, 4], dtype=np.float64)  # (N, 40) annual gpp of forcing years 1..40
    npp_all = np.asarray(traj[j500, :, 1:, 5], dtype=np.float64)
    med_gpp = np.median(gpp_all, axis=1)
    med_npp = np.median(npp_all, axis=1)
    rg = gpp_all[s, t] / np.where(med_gpp[s] > 1e-6, med_gpp[s], np.nan)
    rn = npp_all[s, t] / np.where(med_npp[s] > 1e-6, med_npp[s], np.nan)
    rg_prev = gpp_all[s[t > 0], t[t > 0] - 1] / np.where(med_gpp[s[t > 0]] > 1e-6, med_gpp[s[t > 0]], np.nan)
    print(f"  event-year gpp / site median gpp: median {np.nanmedian(rg):.3f} [IQR {np.nanpercentile(rg, 25):.3f}, {np.nanpercentile(rg, 75):.3f}];  "
          f"npp: {np.nanmedian(rn):.3f} [{np.nanpercentile(rn, 25):.3f}, {np.nanpercentile(rn, 75):.3f}];  previous year's gpp ratio {np.nanmedian(rg_prev):.3f}")
    # against non-event years at the same sites
    non = ~ev[j500][s]  # (n_ev, 40)
    rg_non = gpp_all[s] / np.where(med_gpp[s] > 1e-6, med_gpp[s], np.nan)[:, None]
    print(f"  non-event years at the same sites: gpp ratio median {np.nanmedian(rg_non[non]):.3f} [{np.nanpercentile(rg_non[non], 25):.3f}, {np.nanpercentile(rg_non[non], 75):.3f}]")
    payload["age500_signature"] = {
        "n_events": int(s.size),
        "drop_median": float(np.median(drop)),
        "gpp_ratio_median": float(np.nanmedian(rg)),
        "npp_ratio_median": float(np.nanmedian(rn)),
    }

    # ---- 4. survival to year 40 ---------------------------------------------
    print("\n== 4. survival of an event's effect to year 40, seed ages >= 150 ==")
    print("  for events in forcing year T: agb40 - agb(T-1) against the drop at T; 'survival' = (agb(T-1) - agb40) / drop")
    print(f"  {'T':>3} {'n':>6} {'drop med':>9} {'surv med':>9} {'surv q25':>9} {'surv q75':>9} {'P(surv>0.5)':>11} {'net40 med':>10}")
    surv_curve = []
    for T in range(1, 41):
        rows = []
        for j in old:
            sj = np.nonzero(ev[j][:, T - 1])[0]
            if sj.size == 0:
                continue
            b = agb[j, sj, T - 1].astype(np.float64)
            a = agb[j, sj, T].astype(np.float64)
            e = agb[j, sj, 40].astype(np.float64)
            rows.append(np.stack([b - a, (b - e), b, e], axis=1))
        r = np.concatenate(rows) if rows else np.zeros((0, 4))
        if r.shape[0] == 0:
            continue
        sv = r[:, 1] / r[:, 0]
        surv_curve.append((T, int(r.shape[0]), float(np.median(r[:, 0])), float(np.median(sv)), float(np.percentile(sv, 25)), float(np.percentile(sv, 75)), float((sv > 0.5).mean()), float(np.median(r[:, 3] - r[:, 2]))))
        print(f"  {T:3d} {r.shape[0]:6d} {np.median(r[:, 0]):9.2f} {np.median(sv):9.3f} {np.percentile(sv, 25):9.3f} {np.percentile(sv, 75):9.3f} {(sv > 0.5).mean():11.3f} {np.median(r[:, 3] - r[:, 2]):10.2f}")
    payload["survival_curve"] = surv_curve

    # ---- 5. loss accounting ---------------------------------------------------
    print("\n== 5. share of the year-40 agb persistence loss, seed ages >= 150 ==")
    loss = ((agb[:, :, 40] - agb[:, :, 0]) / agb_div) ** 2  # (15, N)
    tot_old = loss[old].sum()
    has_ev = ev.any(axis=2)  # (15, N)
    print(f"  total old-age agb persistence loss (sum over 8 ages x {n_sites} sites): {tot_old:.1f}; mean per row {loss[old].mean():.4f}")
    print(f"  rows with >= 1 event: {has_ev[old].mean():.4f} of old rows, carrying {loss[old][has_ev[old]].sum() / tot_old:.3f} of the loss")
    # by the year of the largest event at that row
    biggest = np.where(ev, -d, 0.0).argmax(axis=2)  # (15, N) index of the largest event drop (0 if none)
    print("  loss share by the forcing year of the row's largest event (rows with an event):")
    shares = []
    for T in range(1, 41):
        m = has_ev[old] & (biggest[old] == T - 1)
        shares.append(float(loss[old][m].sum() / tot_old))
    for lo in range(0, 40, 10):
        print("    " + " ".join(f"T{T + 1:02d}:{shares[T]:.3f}" for T in range(lo, lo + 10)))
    print(f"  events in years 1-3 carry {sum(shares[:3]):.3f}, years 4-30 {sum(shares[3:30]):.3f}, years 31-40 {sum(shares[30:]):.3f} of the old-age loss")
    payload["loss_share_by_event_year"] = shares
    # top-10% loss rows: how many have events, and what share of their change is events
    top = loss[old] > np.quantile(loss[old], 0.9)
    step = np.where(ev[old], d[old], 0.0).sum(axis=2)
    tot = agb[old, :, 40] - agb[old, :, 0]
    share_ev = step[top] / np.where(np.abs(tot[top]) > 1e-6, tot[top], np.nan)
    print(f"  top-10% loss rows (old ages): {top.sum()} rows carry {loss[old][top].sum() / tot_old:.3f} of the loss; "
          f"{has_ev[old][top].mean():.3f} have an event; median (event drops)/(40-yr change) {np.nanmedian(share_ev):.2f}")
    # is the rest a smooth decline?  the non-event change on those rows
    print(f"  on those rows, the mean 40-yr change is {tot[top].mean():+.2f} kgC m-2 of which events {step[top].mean():+.2f}; the rest is drift/recovery")
    # a stricter view: loss of the top-10% rows split into event sites vs non-event sites
    print(f"  top-10% rows without any event: {(~has_ev[old][top]).sum()} rows, loss share {loss[old][top & ~has_ev[old]].sum() / tot_old:.3f}")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "mort_census.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {METRICS / 'mort_census.json'}")


if __name__ == "__main__":
    main()

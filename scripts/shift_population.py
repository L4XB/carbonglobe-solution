#!/usr/bin/env python
"""Where the scored population differs from the training population, measured directly.

The board says the test population carries less second moment, changes less over
forty years, and has a persistence slope for height of 0.960 against 1.084 offline
(``notes/board-anchors.md``). Those are aggregates over 185,130 rows. This script
compares the two populations where they are both observable -- ``test_y0`` against
the training set's own year-0 states, and the test drivers against the training
drivers -- and asks, in order:

A. Are the 6,171 test sites training cells? (static-driver fingerprint; nearest
   neighbour distances against the training set's own neighbour distances)
B. Are they a regional subsample? (latitude-band shares through a climate proxy,
   with the proxy's own leave-one-out validation printed first)
C. Per seed age, how do decoded test states compare with training year-0 states?
D. At a given height, do test stands carry the same agb, soil, lai? (the allometry
   check that separates a transform constant from a thinner population)
E. Does any hard constant pin the agb decode the way the cap pinned height?
F. How different are the two scenarios' initial states?
G. The training population's own persistence slopes per band and column, which
   the board's per-band probes are compared against.
H. A first-moment check: the board's E[T] against persistence means under each
   agb decode, assuming the training mean growth ratios.

Every number is measured on the complete training set and both test files.

    ./.venv/bin/python scripts/shift_population.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, shift  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
SOIL = [129, 130, 131, 132, 133, 134, 135]
WIND = list(range(36, 129, 4))  # the 24 seasonal-only drivers (wind at each hour of day)
AGES = np.asarray(io.SEED_AGES)
HEIGHT_BINS = [(3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)]


def nn(query: np.ndarray, ref: np.ndarray, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean nearest neighbour of every query row among ``ref`` rows, chunked."""
    d = np.empty(len(query))
    idx = np.empty(len(query), dtype=np.int64)
    r2 = (ref**2).sum(1)
    for lo in range(0, len(query), 256):
        q = query[lo : lo + 256]
        dist = (q**2).sum(1)[:, None] + r2[None, :] - 2 * q @ ref.T
        if exclude is not None:
            dist[np.arange(len(q)), exclude[lo : lo + 256]] = np.inf
        j = dist.argmin(1)
        idx[lo : lo + 256] = j
        d[lo : lo + 256] = np.sqrt(np.maximum(dist[np.arange(len(q)), j], 0))
    return d, idx


def haversine_km(la1, lo1, la2, lo2) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def train_latlon() -> tuple[np.ndarray, np.ndarray]:
    """Site coordinates from the 0.5 degree mask, row-major over mask > 0.

    The mask's values 1/2/3 are the paper's train/val/test splits (3035/338/50779
    cells); the row-major order is checked below by the latitude-temperature
    correlation, which must be strongly negative.
    """
    mask = np.load(io.CACHE / "mask2d.npy")
    row, col = np.nonzero(mask > 0)
    return 90.0 - 0.5 * row - 0.25, -180.0 + 0.5 * col + 0.25


def main() -> None:
    xa = np.load(cache.CACHE_DIR / "x_annual.npy", mmap_mode="r")
    xs = np.load(cache.CACHE_DIR / "x_season.npy", mmap_mode="r")
    ta = {s: np.load(cache.CACHE_DIR / f"test_{s}_annual.npy", mmap_mode="r") for s in io.SCENARIOS}
    ts = {s: np.load(cache.CACHE_DIR / f"test_{s}_season.npy", mmap_mode="r") for s in io.SCENARIOS}
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = shift.measured_floors()
    bare = np.isclose(frames["ssp126"][:, :, 0], floors[0]) & np.isclose(frames["ssp126"][:, :, 1], floors[1])
    zero = floors.copy()
    zero[3:] = np.median(frames["ssp126"][bare], axis=0)[3:]  # lai and flux plateaus
    m_all = shift.M_HEIGHT
    y0 = cache.state_block(0)  # (15, 54152, 7) physical
    y40 = cache.state_block(40)

    # ---------------------------------------------------------------- A
    print("== A. are the test sites training cells? ==")
    soil_tr = np.asarray(xa[:, 0, SOIL], dtype=np.float64)
    soil_te = np.asarray(ta["ssp126"][:, 0, SOIL], dtype=np.float64)
    print(f"  static drivers identical across scenarios: {np.array_equal(np.asarray(ta['ssp126'][:, 0, SOIL]), np.asarray(ta['ssp585'][:, 0, SOIL]))}")
    print(f"  distinct soil fingerprints: train {len(np.unique(soil_tr, axis=0)):,} of {len(soil_tr):,}; test {len(np.unique(soil_te, axis=0)):,} of {len(soil_te):,}")
    tr_set = {tuple(r) for r in np.round(soil_tr, 5)}
    print(f"  exact matches (5 dp) test -> train: {sum(tuple(r) in tr_set for r in np.round(soil_te, 5))}")
    mu, sd = soil_tr.mean(0), soil_tr.std(0)
    a, b = (soil_tr - mu) / sd, (soil_te - mu) / sd
    d_te, _ = nn(b, a)
    rng = np.random.default_rng(42)
    qs = np.sort(rng.choice(len(a), 6000, replace=False))
    d_tr, _ = nn(a[qs], a, exclude=qs)
    print(f"  NN distance in soil z-space, quantiles [5 25 50 75 95]:")
    print(f"    test -> train  {np.percentile(d_te, [5, 25, 50, 75, 95])}")
    print(f"    train -> train {np.percentile(d_tr, [5, 25, 50, 75, 95])}   (leave-one-out, 6000 sites)")
    u, c = np.unique(np.asarray(ta["ssp126"][:, 0, 135]), return_counts=True)
    ut, ct = np.unique(np.asarray(xa[:, 0, 135]), return_counts=True)
    print(f"  driver 135: modal value train {ut[np.argmax(ct)]:.4f} on {ct.max():,} sites; test {u[np.argmax(c)]:.4f} on "
          f"{c.max():,} sites -> the test drivers went through a global rescale; a cell sample would keep 1000.0")
    print("  reading: no test site is a training cell, and test sites sit ~2x farther from their nearest training")
    print("  cell than training cells sit from each other. A different grid or a different source, not a subset.")

    # ---------------------------------------------------------------- B
    print("\n== B. are they a regional subsample? (location proxy: nearest training cell in climate+soil space) ==")
    lat, lon = train_latlon()
    tmean = np.asarray(xs[:, 0, :]).mean(1)
    print(f"  corr(|lat|, 40-yr mean T_air) over training sites = {np.corrcoef(np.abs(lat), tmean)[0, 1]:+.3f} (row-major mask order checks out)")

    def feats(season, annual):
        w = np.asarray(season[:, WIND, :], dtype=np.float64).reshape(len(season), -1)
        t = np.asarray(season[:, 0, :], dtype=np.float64)
        p = np.asarray(season[:, 1, :], dtype=np.float64)
        s = np.asarray(annual[:, 0, SOIL], dtype=np.float64)
        return np.concatenate([w, t, p, s], axis=1)

    f_tr, f_te = feats(xs, xa), feats(ts["ssp126"], ta["ssp126"])
    mu, sd = f_tr.mean(0), f_tr.std(0)
    sd[sd == 0] = 1
    a, b = (f_tr - mu) / sd, (f_te - mu) / sd
    d_tr, i_tr = nn(a[qs], a, exclude=qs)
    km = haversine_km(lat[qs], lon[qs], lat[i_tr], lon[i_tr])
    print(f"  proxy validation, training leave-one-out: NN feature distance median {np.median(d_tr):.2f}; "
          f"NN is within 80 km for {(km < 80).mean():.1%} of sites, median {np.median(km):.0f} km")
    d_te, i_te = nn(b, a)
    print(f"  test -> train NN feature distance median {np.median(d_te):.2f} (q5 {np.percentile(d_te, 5):.2f}, q95 {np.percentile(d_te, 95):.2f})")
    print("  -> test sites are farther from any training cell than neighbouring training cells are from each other,")
    print("     so the proxy locates a test site only to its climate zone, not its cell (the scenario forcing is")
    print("     a different climate source). Band shares are still meaningful:")
    lat_te = lat[i_te]
    for lo, hi, name in ((-90, -23.5, "S extratropics"), (-23.5, 23.5, "tropics"), (23.5, 50, "N temperate"), (50, 90, "boreal")):
        print(f"    {name:15s} train {((lat >= lo) & (lat < hi)).mean():.3f}   test {((lat_te >= lo) & (lat_te < hi)).mean():.3f}")

    # ---------------------------------------------------------------- C
    print("\n== C. per seed age: training year-0 state vs decoded test_y0 (m = 1.14114 on all seven), physical units ==")
    for k, name in ((0, "height"), (1, "agb")):
        print(f"\n  {name}: age | train p10 p50 p90 mean bare | ssp126 p10 p50 p90 mean bare | ssp585 mean | mean ratio te/tr")
        for j, age in enumerate(AGES):
            tr = y0[j][:, k]
            te = m_all * (frames["ssp126"][:, j, k] - zero[k])
            te5 = m_all * (frames["ssp585"][:, j, k] - zero[k])
            qt, qe = np.percentile(tr, [10, 50, 90]), np.percentile(te, [10, 50, 90])
            print(f"  {age:4d} | {qt[0]:6.2f} {qt[1]:6.2f} {qt[2]:6.2f} {tr.mean():6.2f} {(tr <= 1e-6).mean():5.3f} | "
                  f"{qe[0]:6.2f} {qe[1]:6.2f} {qe[2]:6.2f} {te.mean():6.2f} {(te <= 1e-4).mean():5.3f} | {te5.mean():6.2f} | {te.mean() / tr.mean():5.3f}")
    print("  reading: ages 10-50 agree to 2%; from age 100 the test stands are 6-11% taller and 4-5% lighter,")
    print("  and the training median height dips at ages 150-200 (5.9-6.8 m) where the test does not (8.8-9.8 m).")

    # ---------------------------------------------------------------- D
    print("\n== D. allometry: implied agb multiplier = train median agb / test median (y0 - floor) inside decoded-height bins ==")
    bands = {"young 10-50": [10, 20, 30, 50], "mid 70-100": [70, 100], "old 150-500": [150, 200, 250, 300, 350, 400, 450, 500]}
    for label, ages in bands.items():
        js = [int(np.flatnonzero(AGES == a)[0]) for a in ages]
        th = y0[js][:, :, 0].ravel()
        eh = m_all * (frames["ssp126"][:, js, 0] - floors[0]).ravel()
        cells = []
        for k, name in ((1, "agb"), (2, "soil"), (3, "lai")):
            tv = y0[js][:, :, k].ravel()
            ev = (frames["ssp126"][:, js, k] - zero[k]).ravel()
            rs, ws = [], []
            for lo, hi in HEIGHT_BINS:
                mt, me = (th >= lo) & (th < hi), (eh >= lo) & (eh < hi)
                if mt.sum() < 100 or me.sum() < 30:
                    continue
                rs.append(np.median(tv[mt]) / np.median(ev[me]))
                ws.append(me.sum())
            cells.append((name, np.average(rs, weights=ws), np.array(rs)))
        print(f"  {label:12s} " + "   ".join(f"{n} {m:.3f} (bins {r.min():.2f}-{r.max():.2f})" for n, m, r in cells))
    js = [int(np.flatnonzero(AGES == a)[0]) for a in bands["young 10-50"]]
    th = y0[js][:, :, 0].ravel()
    eh = m_all * (frames["ssp126"][:, js, 0] - floors[0]).ravel()
    print("  young band, agb bin by bin:")
    for lo, hi in HEIGHT_BINS:
        mt, me = (th >= lo) & (th < hi), (eh >= lo) & (eh < hi)
        r = np.median(y0[js][:, :, 1].ravel()[mt]) / np.median((frames["ssp126"][:, js, 1] - floors[1]).ravel()[me])
        print(f"    {lo:2d}-{hi:2d} m  n_train {mt.sum():6d} n_test {me.sum():5d}  implied m_agb {r:.3f}   (as a population ratio at m=1.141: test/train = {1.14114 / r:.3f})")
    print("  reading: a transform constant is the same in every bin and every band. This is not: it runs 0.98 -> 1.37")
    print("  across the young band's height bins and sits flat near 1.27 only at old ages. Short young test stands are")
    print("  heavier than training at 1.141, tall ones lighter; old test stands are ~11% lighter at every height.")

    # ---------------------------------------------------------------- E
    print("\n== E. does the seedling pin the agb decode? ==")
    h1, a1 = y0[0][:, 0], y0[0][:, 1]
    seed = np.isclose(h1, 1.5, atol=1e-6)
    print(f"  train age-1: {seed.sum()} sites at exactly 1.5000 m; their agb {a1[seed].min():.2e}-{a1[seed].max():.2e}, lai {y0[0][seed, 3].max():.2e}, gpp {y0[0][seed, 4].max():.2e}")
    for s in io.SCENARIOS:
        te = frames[s][:, 0, :]
        row = np.argmin(te[:, 0])
        dec = m_all * (te[row] - zero)
        print(f"  test {s} age-1 height-min row: decoded height {dec[0]:.4f} m, agb {dec[1]:.4f}, lai {dec[3]:.4f}, gpp {dec[4]:+.4f}")
    gp = np.isclose(frames["ssp126"][:, 0, 4], zero[4])
    print(f"  test ssp126 age-1 rows with gpp at its zero: {gp.sum()}, decoded heights {m_all * (frames['ssp126'][gp, 0, 0] - floors[0]).min():.2f}-"
          f"{m_all * (frames['ssp126'][gp, 0, 0] - floors[0]).max():.2f} m; train age-1 sites with gpp = 0: {(y0[0][:, 4] == 0).sum()}, all at 1.5 m")
    print("  reading: the test seedling row has the height of a seedling but 400x its agb and leaves it does not have;")
    print("  it is not the training seedling state, so it anchors nothing. No hard constant pins the agb decode.")

    # ---------------------------------------------------------------- F
    print("\n== F. ssp126 vs ssp585 initial states (decoded, vegetated rows): median relative difference per age ==")
    line = []
    for j, age in enumerate(AGES):
        h6, h5 = frames["ssp126"][:, j, 0] - floors[0], frames["ssp585"][:, j, 0] - floors[0]
        a6, a5 = frames["ssp126"][:, j, 1] - floors[1], frames["ssp585"][:, j, 1] - floors[1]
        veg = (h6 > 0.05) & (h5 > 0.05) & (a6 > 0.01) & (a5 > 0.01)
        line.append(f"{age}: h {np.median(h5[veg] / h6[veg]) - 1:+.3f} a {np.median(a5[veg] / a6[veg]) - 1:+.3f}")
    print("  " + " | ".join(line[:8]))
    print("  " + " | ".join(line[8:]))
    same = [np.isclose(frames["ssp126"][:, j, :2], frames["ssp585"][:, j, :2]).all(axis=1).mean() for j in range(15)]
    print(f"  share of rows with identical height and agb in both scenarios, per age: {np.round(same, 3)}")

    # ---------------------------------------------------------------- G
    print("\n== G. training persistence slopes s* = E[X T]/E[X^2] and mean ratios E[T]/E[X], scaled units ==")
    div = cache.scoring_divisors()
    x_all, t_all = y0[:, :, :2] / div, y40[:, :, :2] / div
    split = holdout.make_split(seed=42)
    print(f"  {'band':6s} {'sites':8s} | {'h s*':>7s} {'h E[T]/E[X]':>12s} {'A_h':>8s} | {'a s*':>7s} {'a E[T]/E[X]':>12s} {'A_a':>8s} | {'pooled s*':>9s}")
    for sites_name, sites in (("all", np.arange(x_all.shape[1])), ("public", split.public)):
        for band, ages in list(shift.BANDS.items()) + [("all", tuple(AGES))]:
            js = [int(np.flatnonzero(AGES == a)[0]) for a in ages]
            x, t = x_all[js][:, sites].reshape(-1, 2), t_all[js][:, sites].reshape(-1, 2)
            a_, b_ = (x**2).mean(0), (x * t).mean(0)
            print(f"  {band:6s} {sites_name:8s} | {b_[0] / a_[0]:7.4f} {t[:, 0].mean() / x[:, 0].mean():12.4f} {a_[0]:8.4f} | "
                  f"{b_[1] / a_[1]:7.4f} {t[:, 1].mean() / x[:, 1].mean():12.4f} {a_[1]:8.4f} | {b_.sum() / a_.sum():9.4f}")
    print("  board (all rows): s*_h 0.960 on the 1.141-decoded height; s*_a 1.065 on the floor-shifted agb, i.e. 0.934 at m = 1.141")

    # ---------------------------------------------------------------- H
    print("\n== H. first moment: board E[T] = 0.9933 against persistence means under each agb decode ==")
    pers = {m: shift.persistence(m_agb=m) for m in (1.0, 1.14114, 1.26)}
    r_h_train = t_all[..., 0].mean() / x_all[..., 0].mean()
    r_a_train = t_all[..., 1].mean() / x_all[..., 1].mean()
    r_h_board = r_h_train * (0.960 / 1.0843)  # if the board's slope deficit carries to the mean
    print(f"  training mean growth ratios: height {r_h_train:.4f}, agb {r_a_train:.4f}; height ratio scaled by the board/offline slope ratio: {r_h_board:.4f}")
    print(f"  {'m_agb':>6s} {'mean X_h':>9s} {'mean X_a':>9s} | {'E[T] if train ratios':>21s} {'E[T] if h ratio scaled':>23s}   (board 0.9933)")
    for m, p in pers.items():
        mh, ma = p[:, 0].mean(), p[:, 1].mean()
        print(f"  {m:6.3f} {mh:9.4f} {ma:9.4f} | {0.5 * (r_h_train * mh + r_a_train * ma):21.4f} {0.5 * (r_h_board * mh + r_a_train * ma):23.4f}")
    print("  reading: with training-like growth in both columns the first moment wants m_agb ~ 1.0; with the height")
    print("  growth deficit the board measured, it wants ~1.14. It cannot decide alone, but 1.26 needs a stronger agb")
    print("  decline than training in a population whose trajectories change less -- the one reading nothing supports.")


if __name__ == "__main__":
    main()

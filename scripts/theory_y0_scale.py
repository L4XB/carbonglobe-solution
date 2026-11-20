"""Pin the per-variable scale of the ``test_y0`` decode without spending a slot.

``test_y0 = a_k * raw_k + floor_k`` (``instrument.md`` 4a; the floor is settled on
the board). Two hard constants of the process model survive any affine map and give
``a_height`` directly:

* the height **cap**: training slot-0 height saturates at exactly 35.0002 m (53 sites
  sit on it at seed age 100); test_y0 height saturates at exactly 29.770555 (31 sites
  at seed age 150) -> ``a_h = (29.770555 - floor_h) / 35.0002``;
* the **seedling** height: training slot-0 age-1 height has a hard floor of 1.5000 m
  (107 sites); the test age-1 minimum is 0.415183 -> ``a_h = (0.415183 - floor_h)/1.5``.

For agb, soil and lai there is no hard constant, so the scale is read off the
allometric relation instead: at old seed ages the state is near equilibrium and
agb given height is set by ED's allometry, not by which sites were sampled. With
heights decoded by ``a_h``, the ratio of the training median raw agb to the test
median ``(y0 - floor)`` inside each height bin is ``a_agb``; a ratio that is flat
across bins is the evidence that it is a scale and not a population difference.

The flux ratio NPP/GPP is a model constant (~0.47, ``theory_growth_curves.py``), so
it pins ``a_npp / a_gpp``. Everything printed is measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
DEC = 11
OLD_AGES = (100, 150, 200, 250, 300, 350, 400, 450, 500)
HEIGHT_BINS = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35.1])


def main() -> None:
    _, y126 = io.open_test("ssp126")
    _, y585 = io.open_test("ssp585")
    floor = y126.min(axis=(0, 1)).astype(np.float64)
    names = io.STATE_NAMES

    # ---- height: two hard anchors --------------------------------------------
    print("== height scale from two process-model constants ==")
    y100 = np.load(io.TRAIN / "glob_Y100.npy", mmap_mode="r")
    cap_train = float(np.asarray(y100[:, 0, DEC, 0]).max())
    n_cap_train = int((np.asarray(y100[:, 0, DEC, 0]) == cap_train).sum())
    h_old = y126[:, [io.SEED_AGES.index(a) for a in OLD_AGES], 0]
    cap_test = float(h_old.max())
    n_cap_test = int((h_old == cap_test).sum())
    a_cap = (cap_test - floor[0]) / cap_train
    y001 = np.load(io.TRAIN / "glob_Y001.npy", mmap_mode="r")
    seed_train = float(np.asarray(y001[:, 0, DEC, 0]).min())
    n_seed_train = int((np.asarray(y001[:, 0, DEC, 0]) == seed_train).sum())
    seed_test = float(y126[:, 0, 0].min())
    a_seed = (seed_test - floor[0]) / seed_train
    print(f"   cap : train slot-0 max {cap_train:.4f} m ({n_cap_train} sites on it, age 100); test max {cap_test:.6f} ({n_cap_test} rows on it, ages >= 100) -> a_h = {a_cap:.5f}")
    print(f"   seed: train age-1 slot-0 min {seed_train:.4f} m ({n_seed_train} sites on it); test age-1 min {seed_test:.6f} -> a_h = {a_seed:.5f}")
    a_h = a_cap
    print(f"   using a_h = {a_h:.5f} (cap anchor: many rows sit on it); decode: raw_h = (y0_h - {floor[0]:.6f}) / {a_h:.5f} = {1 / a_h:.4f} * y0_h + {-floor[0] / a_h:.4f}")

    # ---- agb / soil / lai: allometry conditional on decoded height --------------
    print("\n== agb, soil, lai scale from the relation to height at old seed ages (train slot 0 vs test_y0) ==")
    tr_h, tr_v = [], []
    for age in OLD_AGES:
        y = np.load(io.TRAIN / f"glob_Y{age:03d}.npy", mmap_mode="r")
        s0 = np.asarray(y[:, 0, DEC, :], dtype=np.float64)
        tr_h.append(s0[:, 0])
        tr_v.append(s0[:, 1:4])
    tr_h = np.concatenate(tr_h)
    tr_v = np.concatenate(tr_v)
    j_old = [io.SEED_AGES.index(a) for a in OLD_AGES]
    te = y126[:, j_old, :].reshape(-1, 7).astype(np.float64)
    te_h = (te[:, 0] - floor[0]) / a_h
    te_v = te[:, 1:4] - floor[1:4]
    veg_tr = tr_h > 2.0
    veg_te = te_h > 2.0
    print(f"   {'height bin':>12} {'n_train':>8} {'n_test':>7}   " + "   ".join(f"{n:>22}" for n in names[1:4]))
    print(f"   {'':>12} {'':>8} {'':>7}   " + "   ".join(f"{'train med / test med':>22}" for _ in range(3)))
    ratios = {k: [] for k in range(3)}
    weights = []
    for lo, hi in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:]):
        mt = veg_tr & (tr_h >= lo) & (tr_h < hi)
        me = veg_te & (te_h >= lo) & (te_h < hi)
        if mt.sum() < 200 or me.sum() < 50:
            continue
        cells = []
        for k in range(3):
            r = np.median(tr_v[mt, k]) / max(np.median(te_v[me, k]), 1e-9)
            ratios[k].append(r)
            cells.append(f"{np.median(tr_v[mt, k]):8.3f} / {np.median(te_v[me, k]):8.3f} = {r:.3f}")
        weights.append(me.sum())
        print(f"   {lo:5.0f}-{hi:5.0f} m {mt.sum():>8} {me.sum():>7}   " + "   ".join(cells))
    w = np.array(weights, dtype=np.float64)
    for k, n in enumerate(names[1:4]):
        r = np.array(ratios[k])
        print(f"   {n:>6}: decode multiplier m_{n} = raw/(y0-floor) = {np.average(r, weights=w):.3f} (test-row-weighted mean over bins), bin range [{r.min():.3f}, {r.max():.3f}], sd {r.std():.3f}")
    print("   (for height the same quantity is m_h = 1/a_h = %.4f; bins above 30 m are dominated by the cap artifact, see theory.md)" % (1 / a_h))

    # ---- the cap sites as a second agb anchor ------------------------------------
    print("\n== agb at the height cap (train 35 m sites vs test 29.77 rows), old ages pooled ==")
    at_cap_tr = tr_h >= cap_train - 1e-3
    at_cap_te = te[:, 0] >= cap_test - 1e-5
    for k, n in enumerate(names[1:4]):
        print(f"   {n:>6}: train median {np.median(tr_v[at_cap_tr, k]):.3f} (n={at_cap_tr.sum()}), test median (y0-floor) {np.median(te_v[at_cap_te, k]):.3f} (n={at_cap_te.sum()}) -> ratio {np.median(tr_v[at_cap_tr, k]) / np.median(te_v[at_cap_te, k]):.3f}")

    # ---- fluxes: NPP/GPP ---------------------------------------------------------
    # The transformed zero of gpp/npp is the value every bare row shares (the plateau),
    # not the column minimum: a few thousand rows sit below the plateau.
    print("\n== flux scale ratio from NPP/GPP (floor = bare-row plateau) ==")
    bare = np.isclose(te[:, 0], floor[0]) & np.isclose(te[:, 1], floor[1])
    plateau = np.median(te[bare], axis=0)
    g = te[:, 4] - plateau[4]
    n_ = te[:, 5] - plateau[5]
    ok = veg_te & (g > 0.05)
    cue_test = np.median(n_[ok] / g[ok])
    y500 = np.load(io.TRAIN / "glob_Y500.npy", mmap_mode="r")
    s0 = np.asarray(y500[:, 0, DEC, :], dtype=np.float64)
    okt = (s0[:, 0] > 2.0) & (s0[:, 4] > 0.05)
    cue_train = np.median(s0[okt, 5] / s0[okt, 4])
    print(f"   flux plateaus gpp {plateau[4]:.6f} npp {plateau[5]:.6f} rh {plateau[6]:.6f} (column minima {floor[4]:.4f}, {floor[5]:.4f}, {floor[6]:.4f})")
    print(f"   train slot-0 median NPP/GPP {cue_train:.4f}; test median (y0_npp - p)/(y0_gpp - p) {cue_test:.4f} -> m_npp / m_gpp = {cue_train / cue_test:.3f}")

    # ---- scenario check: the decode must give identical bare rows -----------------
    print("\n== consistency ==")
    print(f"   floors identical in ssp585: {np.allclose(y585.min(axis=(0, 1)), floor)}; ssp585 height cap {y585[:, j_old, 0].max():.6f}")
    print(f"   decoded test height at old ages: median {np.median(te_h[veg_te]):.2f} m, 99% {np.percentile(te_h[veg_te], 99):.2f}; train slot 0 old ages: median {np.median(tr_h[veg_tr]):.2f}, 99% {np.percentile(tr_h[veg_tr], 99):.2f}")


if __name__ == "__main__":
    main()

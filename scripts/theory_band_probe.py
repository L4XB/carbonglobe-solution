"""Persistence probes that rescale only the rows of one seed-age band.

The lead's parabola (``notes/board-anchors.md``): for a fixed prediction X, the score
of ``s * X`` is ``s^2 A - 2 s B + C`` with ``A = E[X^2]`` known. That identity holds
row-subset by row-subset, so scaling only the rows of one band by ``s`` and leaving
the rest at ``s = 1`` changes the score by ``n_band/N * ((s^2 - 1) A_band - 2 (s - 1) B_band)``,
and one slot per band yields ``B_band`` and hence ``s*_band = B_band / A_band``: how
much the test trajectories of that band change over 40 years, in the direction of
their initial state. The same quantity on the training holdout (height 1.204 for
seed ages < 150 and 1.028 for >= 150; agb 1.136 and 0.878) is the comparison, and
the difference is the per-band transfer the offline holdout cannot see
(``theory.md`` Sec. 5 Q2).

Base decode: height x1.141 (pinned), agb x1.14 (the one-multiplier reading), i.e.
``results/persistence_mh1141_agb114.csv`` is the ``s = 1`` reference for all three.
Nothing here submits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("ssp126", "ssp585")
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
M_HEIGHT, M_AGB = 1.14114, 1.14
S_BAND = 0.5
BANDS = {"young": [1, 10, 20, 30], "mid": [50, 70, 100], "old": [150, 200, 250, 300, 350, 400, 450, 500]}


def main() -> None:
    stats = np.load(ROOT / "cache" / "data_stats.npz")
    means = stats["y_mean"][:2].astype(np.float64)
    frames = {}
    for s in SCENARIOS:
        with np.load(ROOT / "data" / f"test_{s}.npz") as z:
            frames[s] = np.asarray(z["test_y0"], dtype=np.float64)
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    template = np.loadtxt(ROOT / "cache" / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, base_h, base_a, age_of_row = [], [], [], []
    for s in SCENARIOS:
        y0 = frames[s]
        h = M_HEIGHT * (y0[:, :, 0] - floors[0]) / means[0]
        a = M_AGB * (y0[:, :, 1] - floors[1]) / means[1]
        for site in range(y0.shape[0]):
            for j, age in enumerate(AGES):
                ids.append(f"{s}_{site}_{age}")
                age_of_row.append(age)
        base_h.append(h.reshape(-1))
        base_a.append(a.reshape(-1))
    ids = np.array(ids)
    assert np.array_equal(ids, template)
    base_h, base_a, age_of_row = np.concatenate(base_h), np.concatenate(base_a), np.array(age_of_row)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for name, ages in BANDS.items():
        m = np.isin(age_of_row, ages)
        h, a = base_h.copy(), base_a.copy()
        h[m] *= S_BAND
        a[m] *= S_BAND
        a_band = 0.5 * (float((base_h[m] ** 2).mean()) + float((base_a[m] ** 2).mean()))
        path = out / f"persistence_mh1141_agb114_band_{name}_s05.csv"
        with open(path, "w") as fh:
            fh.write("id,height,agb\n")
            for i in range(len(ids)):
                fh.write(f"{ids[i]},{h[i]:.6f},{a[i]:.6f}\n")
        print(f"{name:6s}: {m.sum():,} rows scaled by {S_BAND} ({m.mean():.3f} of rows); A_band = E[X^2] over the band (both columns pooled) = {a_band:.6f}; "
              f"A_band height {float((base_h[m] ** 2).mean()):.6f} agb {float((base_a[m] ** 2).mean()):.6f} -> {path.name}")
    print("reference (s = 1 everywhere): results/persistence_mh1141_agb114.csv. Nothing submitted.")


if __name__ == "__main__":
    main()

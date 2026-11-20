"""Triangulate the agb decode multiplier: correct the QQ slope by height's known answer.

Quantile-quantile matching of ``test_y0`` against training slot 0 gives, per
variable, a slope ``s`` in ``test = s * raw + b``. For height the true ``s`` is known
independently, ``1/1.14114 = 0.8764``, from two process constants (the 35 m cap and
the 1.5 m seedling, ``theory_y0_scale.py``). Whatever height's QQ slope deviates from
0.8764 is population contamination of the QQ method (the test sites are a
different, less extreme sample). Applying the same multiplicative correction to
agb's QQ slope removes that contamination to first order and leaves the decode.

Bare rows are excluded on both sides (their share differs between the populations
and would tilt the low quantiles), and several quantile ranges are shown so the
answer's stability is visible rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
S_HEIGHT_TRUE = 1.0 / 1.14114
AGES = (30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)
RANGES = ((10, 90), (20, 95), (30, 97))


def qq_slope(test: np.ndarray, train: np.ndarray, lo: float, hi: float) -> float:
    qs = np.linspace(lo, hi, 25)
    return float(np.polyfit(np.percentile(train, qs), np.percentile(test, qs), 1)[0])


def main() -> None:
    traj = np.load(io.ROOT / "work" / "cache" / "y_traj.npy", mmap_mode="r")
    _, y0 = io.open_test("ssp126")
    y0 = y0.astype(np.float64)
    floor = y0.min(axis=(0, 1))
    print(f"{'age':>4} | {'range':>7} | {'s_h (QQ)':>9} {'contam':>7} | {'s_agb (QQ)':>10} {'m_agb raw':>9} {'m_agb corr':>10} | {'m_soil corr':>11} {'m_lai corr':>10}")
    corr_all, raw_all = [], []
    for age in AGES:
        j = io.SEED_AGES.index(age)
        tr = np.asarray(traj[j, :, 0, :], dtype=np.float64)
        te = y0[:, j, :]
        veg_tr = tr[:, 0] > 0.0
        veg_te = te[:, 0] > floor[0] + 1e-6
        for lo, hi in RANGES:
            s_h = qq_slope(te[veg_te, 0], tr[veg_tr, 0], lo, hi)
            contam = s_h / S_HEIGHT_TRUE  # >1: QQ overstates the slope for a known variable
            s_a = qq_slope(te[veg_te, 1], tr[veg_tr, 1], lo, hi)
            s_s = qq_slope(te[veg_te, 2], tr[veg_tr, 2], lo, hi)
            s_l = qq_slope(te[veg_te, 3], tr[veg_tr, 3], lo, hi)
            m_raw, m_corr = 1.0 / s_a, contam / s_a
            corr_all.append(m_corr)
            raw_all.append(m_raw)
            print(f"{age:>4} | {lo:>3}-{hi:<3} | {s_h:9.4f} {contam:7.4f} | {s_a:10.4f} {m_raw:9.4f} {m_corr:10.4f} | {contam / s_s:11.4f} {contam / s_l:10.4f}")
    print(f"\nheight QQ slope answer is known: {S_HEIGHT_TRUE:.4f}. agb multiplier: raw QQ median {np.median(raw_all):.4f} "
          f"[{np.min(raw_all):.3f}, {np.max(raw_all):.3f}]; corrected by height's contamination median {np.median(corr_all):.4f} "
          f"[{np.min(corr_all):.3f}, {np.max(corr_all):.3f}]")
    # the grounding workflow's two 'impossible' extremes decode onto the process constants
    print(f"\ngrounding check: age-1 height min {y0[:, 0, 0].min():.4f} -> {(y0[:, 0, 0].min() - floor[0]) * 1.14114:.4f} m (seedling 1.5); "
          f"age-1 agb min {y0[:, 0, 1].min():.4f} -> {(y0[:, 0, 1].min() - floor[1]) * 1.14114:.5f} kgC/m^2 (floor 0)")


if __name__ == "__main__":
    main()

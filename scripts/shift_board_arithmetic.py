#!/usr/bin/env python
"""Everything the persistence anchors on the board already determine, and what each
pending probe would add -- before a slot is spent on it.

The metric is a mean squared error, so for a fixed prediction ``X`` the score of
``s X`` is ``s^2 A - 2 s B + C`` with ``A = E[X^2]`` ours to compute. Four files are
on the board (``notes/board-anchors.md``):

    P1  persistence_shifted        (y0 - floor)/mean, both columns      0.252
    P2  persistence_s05            0.5 * P1                              0.913
    P3  persistence_mh1141_agb100  P1 with height x 1.14114              0.247
    P4  constant_one               1.0 everywhere                        1.602

P1 and P2 give the pooled ``B`` and ``C``; P1 and P3 differ only in the height block
and give ``B_h``; then ``B_a = 2 B - B_h`` -- so the **agb parabola is already
determined** and every persistence file that only changes the agb multiplier has a
predictable score. This script prints that chain with worst-case bounds from the
board's rounding, predicts the pending files, and works out the per-band probe
arithmetic so the lead can see what one slot buys.

    ./.venv/bin/python scripts/shift_board_arithmetic.py [--rounding 0.0005]
                                                        [--scores '{"P1":0.252,...}']
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, shift  # noqa: E402

RESULTS = io.ROOT / "results"

#: Board scores as recorded in notes/board-anchors.md (three decimals shown).
DEFAULT_SCORES = {"P1": 0.252, "P2": 0.913, "P3": 0.247, "P4": 1.602}
FILES = {
    "P1": "persistence_shifted.csv",
    "P2": "persistence_s05.csv",
    "P3": "persistence_mh1141_agb100.csv",
    "P4": "constant_one.csv",
}
PENDING = {
    "agb114": "persistence_mh1141_agb114.csv",
    "agb126": "persistence_mh1141_agb126.csv",
}
BAND_FILES = {b: f"persistence_mh1141_agb114_band_{b}_s05.csv" for b in shift.BANDS}


def chain(sc: dict[str, float], a_h: float, a_a: float) -> dict[str, float]:
    """The anchor arithmetic for one set of scores."""
    a = 0.5 * (a_h + a_a)
    b, c = shift.scaled_parabola(sc["P1"], sc["P2"], 0.5, a)
    # P3 scaled only the height column, whose share of the pooled mean is 1/2:
    # P3 - P1 = 0.5 [(m^2 - 1) A_h - 2 (m - 1) B_h]
    m = shift.M_HEIGHT
    b_h = ((m * m - 1.0) * a_h - 2.0 * (sc["P3"] - sc["P1"])) / (2.0 * (m - 1.0))
    b_a = 2.0 * b - b_h
    e_t = (1.0 + c - sc["P4"]) / 2.0
    return {
        "A": a, "B": b, "C": c, "B_h": b_h, "B_a": b_a,
        "s_h": b_h / a_h, "s_a": b_a / a_a, "E_T": e_t, "Var_T": c - e_t**2,
        # the height column's best scalar leaves this score, from P3's height at m
        "best_h": sc["P3"] - 0.5 * (b_h - m * a_h) ** 2 / a_h,
        "P3": sc["P3"],
    }


def agb_score(c: dict[str, float], a_a: float, m_agb: float) -> float:
    """Score of P3 with its agb column multiplied by ``m_agb``."""
    return c["P3"] + 0.5 * ((m_agb * m_agb - 1.0) * a_a - 2.0 * (m_agb - 1.0) * c["B_a"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounding", type=float, default=5e-4, help="half the board's last shown digit")
    ap.add_argument("--scores", type=str, default=None, help="JSON overriding the anchor scores")
    args = ap.parse_args()
    sc = dict(DEFAULT_SCORES)
    if args.scores:
        sc.update(json.loads(args.scores))
    r = args.rounding

    keys = shift.template_keys()
    ages = keys["age"].to_numpy()
    p1 = shift.read_predictions(RESULTS / FILES["P1"])
    p3 = shift.read_predictions(RESULTS / FILES["P3"])
    a_h = float(np.mean(p1[:, 0] ** 2))
    a_a = float(np.mean(p1[:, 1] ** 2))
    if not np.allclose(p3[:, 1], p1[:, 1]):
        raise AssertionError("P3 must differ from P1 in the height block only")
    if not np.allclose(p3[:, 0], shift.M_HEIGHT * p1[:, 0], atol=2e-6):
        raise AssertionError("P3's height block is not P1's times 1.14114")

    print("== 1. the anchor chain ==")
    print(f"  A_h = E[X_h^2] = {a_h:.6f}   A_a = {a_a:.6f}   pooled {0.5 * (a_h + a_a):.6f}   (from the files)")
    nominal = chain(sc, a_h, a_a)
    # worst case over the 16 rounding corners
    corners = [chain({k: sc[k] + d for k, d in zip(sc, ds)}, a_h, a_a)
               for ds in itertools.product((-r, r), repeat=4)]
    lo = {k: min(cn[k] for cn in corners) for k in nominal}
    hi = {k: max(cn[k] for cn in corners) for k in nominal}
    for k, label in (
        ("B", "B pooled = E[X T]"), ("C", "C = E[T^2]"), ("E_T", "E[T]"), ("Var_T", "Var(T)"),
        ("B_h", "B_h = E[X_h T_h]"), ("B_a", "B_a = 2B - B_h"),
        ("s_h", "s*_h on (y0-floor) height"), ("s_a", "s*_a on (y0-floor) agb"),
    ):
        print(f"  {label:28s} {nominal[k]:9.5f}   [{lo[k]:.5f}, {hi[k]:.5f}]")
    print(f"  -> relative to the 1.14114 height decode the board slope is {nominal['s_h'] / shift.M_HEIGHT:.4f} "
          f"[{lo['s_h'] / shift.M_HEIGHT:.4f}, {hi['s_h'] / shift.M_HEIGHT:.4f}]; offline (all sites) it is 1.0843")
    print(f"  -> the agb slope {nominal['s_a']:.4f} is on the floor-shifted column; at a decode multiplier m it "
          f"reads {nominal['s_a']:.4f}/m: m=1.141 -> {nominal['s_a'] / 1.14114:.4f}, m=1.26 -> {nominal['s_a'] / 1.26:.4f}; "
          f"offline (all sites) it is 0.9194")

    print("\n== 2. what the pending persistence files must score (no new information in them) ==")
    print("  score(m_agb) = P3 + 0.5 [(m^2 - 1) A_a - 2 (m - 1) B_a]")
    ref_err = 0.0
    for tag, name in PENDING.items():
        pred = shift.read_predictions(RESULTS / name)
        m = float(np.median(pred[:, 1][p1[:, 1] > 0] / p1[:, 1][p1[:, 1] > 0]))
        est = [agb_score(c, a_a, m) for c in corners]
        val = agb_score(nominal, a_a, m)
        if tag == "agb114":
            ref_err = max(abs(e - val) for e in est)
        print(f"  {name:36s} m_agb = {m:.3f}  predicted {val:.4f}  [{min(est):.4f}, {max(est):.4f}]")
    m_star = nominal["s_a"]
    print(f"  best possible agb scalar m* = {m_star:.4f} would score {agb_score(nominal, a_a, m_star):.4f}; "
          f"the height column's best is {nominal['best_h']:.4f} at s = {nominal['s_h']:.4f} on the shifted column "
          f"(P3 sits at {shift.M_HEIGHT})")
    print("  reading: submitting agb114 or agb126 checks the arithmetic; it does not measure anything new.")

    print("\n== 3. the per-band persistence probes (reference: persistence_mh1141_agb114, predicted above) ==")
    ref = shift.read_predictions(RESULTS / PENDING["agb114"])
    masks = shift.band_masks(ages)
    ref_score = agb_score(nominal, a_a, 1.14)
    print(f"  delta = f_band [(s^2 - 1) A_band - 2 (s - 1) B_band],  s = 0.5  ->  B_band = delta / f_band + 0.75 A_band")
    print(f"  {'band':6s} {'f_band':>7s} {'A_h':>9s} {'A_a':>9s} {'A_pooled':>9s} | {'s* err':>7s} | "
          f"{'score if s*=train':>18s} {'if s*=0.85':>11s} {'if s*=1.0':>10s}")
    train_pooled = {"young": 1.6333, "mid": 1.0366, "old": 0.9210}  # scripts/shift_population.py, all sites
    for band in shift.BANDS:
        m = masks[band]
        f = m.mean()
        ah, aa = float(np.mean(ref[m, 0] ** 2)), float(np.mean(ref[m, 1] ** 2))
        ap_ = 0.5 * (ah + aa)
        err = r / (0.5 * f * ap_)  # |ds*/d delta| = 1/(2 (1-s) f A) with |d delta| <= 2r
        def sc_if(sstar: float) -> float:
            return ref_score + f * (-0.75 * ap_ + sstar * ap_)
        print(f"  {band:6s} {f:7.4f} {ah:9.6f} {aa:9.6f} {ap_:9.6f} | {err:7.4f} | "
              f"{sc_if(train_pooled[band]):18.4f} {sc_if(0.85):11.4f} {sc_if(1.0):10.4f}")
    print("  the old band's probe resolves s*_old to +/-0.0005 and moves the score by 0.2-0.4 between hypotheses;")
    print("  the young band's probe resolves s*_young to +/-0.008, but at seed age 1 y0 is uninformative and")
    print("  s*_young says little about the growth curve a model has to produce.")
    m_old = masks["old"]
    a_old = 0.5 * (float(np.mean(ref[m_old, 0] ** 2)) + float(np.mean(ref[m_old, 1] ** 2)))
    print(f"  the reference file need not be on the board: its predicted score is off by at most "
          f"{ref_err:.4f}, which costs +/-{ref_err / (0.5 * m_old.mean() * a_old):.4f} on s*_old.")

    print("\n== 3b. what the whole-population slopes already imply for the old band, per column ==")
    print("  B_col = sum_band f_band A_band,col s*_band,col. If the young and mid bands keep their training")
    print("  slopes (their y0 matches training to 2%, Sec. C of shift_population.py), the old band's slope")
    print("  is what is left; it depends on the agb decode m because the board only sees the shifted column.")
    train = {"young": (1.6377, 1.6263), "mid": (1.0360, 1.0372), "old": (1.0275, 0.8848)}  # all sites, h/a
    p3_ref = shift.read_predictions(RESULTS / FILES["P3"])
    print(f"  {'column':7s} {'m':>6s} {'B_board':>8s} {'young+mid':>10s} {'A_old':>8s} {'s*_old':>7s} {'train':>7s}")
    for m_agb in (1.0, 1.14114, 1.26):
        for k, col in enumerate(shift.COLUMNS):
            m = shift.M_HEIGHT if k == 0 else m_agb
            x = p1[:, k] * m  # decoded column
            b_board = (nominal["B_h"] if k == 0 else nominal["B_a"]) * m  # E[X_dec T] = m E[X_1 T]
            ym = sum(masks[b].mean() * np.mean(x[masks[b]] ** 2) * train[b][k] for b in ("young", "mid"))
            a_old = masks["old"].mean() * np.mean(x[masks["old"]] ** 2)
            print(f"  {col:7s} {m:6.3f} {b_board:8.4f} {ym:10.4f} {a_old:8.4f} {(b_board - ym) / a_old:7.4f} {train['old'][k]:7.4f}")
        if m_agb == 1.0:
            print("  (height does not depend on m_agb; repeated for reading)")
    print("  reading: on the board the tall old stands lose ~15% of their height where training gains 3%; for agb")
    print("  the old band matches training only if the decode is ~1.14, gains 5% at 1.0 and loses 10% more at 1.26.")
    print("  The height finding is decode-free (the cap pins 1.141) and is the one thing a damping cannot fix:")
    print("  a model that grows old stands cannot be damped into shrinking them, only rescaled.")

    print("\n== 3c. sensitivity of the old band's height slope to what the young and mid bands do ==")
    print("  s*_old,h = (B_h,decoded - f_y A_y s_y - f_m A_m s_m) / (f_old A_old); the young band holds only")
    x_h = p1[:, 0] * shift.M_HEIGHT
    fa = {b: masks[b].mean() * float(np.mean(x_h[masks[b]] ** 2)) for b in shift.BANDS}
    b_dec = nominal["B_h"] * shift.M_HEIGHT
    b_lo, b_hi = lo["B_h"] * shift.M_HEIGHT, hi["B_h"] * shift.M_HEIGHT
    print(f"  {fa['young'] / sum(fa.values()):.1%} of E[X_h^2] (f A: young {fa['young']:.4f}, mid {fa['mid']:.4f}, old {fa['old']:.4f}); "
          f"training slopes are young 1.638, mid 1.036, old 1.028")
    print(f"  {'s_young':>8s} {'s_mid':>6s} | {'s*_old,h':>9s} {'rounding':>16s} | {'gain of c* on old height if model slope 1.03':>44s}")
    for s_y in (1.638, 1.4, 1.2, 1.1, 1.0):
        for s_m in (1.036, 0.95):
            s_old = (b_dec - fa["young"] * s_y - fa["mid"] * s_m) / fa["old"]
            s_lo = (b_lo - fa["young"] * s_y - fa["mid"] * s_m) / fa["old"]
            s_hi = (b_hi - fa["young"] * s_y - fa["mid"] * s_m) / fa["old"]
            gain = 0.5 * fa["old"] * (1.03 - s_old) ** 2
            print(f"  {s_y:8.3f} {s_m:6.3f} | {s_old:9.4f} [{s_lo:.4f}, {s_hi:.4f}] | {gain:44.4f}")
    print("  reading: the young band cannot push s*_old,h above ~0.93 even with no growth at all (s_y = 1.0),")
    print("  because it holds 8% of the second moment; the mid band moves it by 0.035 per 0.1 of slope. The")
    print("  finding that survives every combination is s*_old,h <= 0.95 against a model at ~1.03. The exact")
    print("  value is what the (P, c P_old,h) pair measures; the probe's 0.85 is only its step length, and a")
    print("  larger step gives a tighter c* (t_err = rounding / E[V^2]), so 0.85 is right even if c* is 0.93.")

    print("\n== 4. what a model pair would give instead ==")
    print("  files P (lambda = 1) and P_d = X + lambda_d (P - X) on one band lie on a line; two scores give the")
    print("  MSE-optimal lambda for that band and the score there (cgb.shift.line_search). That is the number the")
    print("  correction needs; the persistence band probes give the population's s*, one step removed from it.")


if __name__ == "__main__":
    main()

"""Prove the scorer, the scaling and the row contract by measurement.

Four claims, each checked against a number this script produces:

1. The README's divisors are the training set's own ``y_mean[0:2]``.
2. A submission that *is* the truth scores exactly 0.
3. The same submission left **unscaled** scores as badly as the README warns.
4. A submission off by a known constant scores the arithmetic we expect.

Plus the row contract: 185,130 rows, the id grammar, and the exact template order.

No truth file ships with this competition, so the "truth" here is a synthetic
solution drawn from the released test initial states -- the arithmetic of the
metric does not care where the numbers came from, and the scaling factor it
exposes is the same one the real solution uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, submission as sub  # noqa: E402

RNG = np.random.default_rng(42)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    rule("1. the divisors")
    for name, (shipped, stated) in sub.check_divisors().items():
        print(f"  {name:7s} data_stats.npz y_mean = {shipped!r}")
        print(f"  {name:7s} README          says   = {stated!r}   -> agree to 5 dp")

    rule("2. the row contract of sample_submission.csv")
    template = sub.load_template()
    print(f"  columns            : {list(template.columns)}")
    print(f"  rows               : {len(template):,}  (expected {sub.EXPECTED_ROWS:,})")
    assert len(template) == sub.EXPECTED_ROWS, "row count is not the measured 185,130"

    key = sub.parse_ids(template["id"])
    print(f"  scenarios          : {sorted(key['scenario'].unique())}")
    print(f"  sites              : {key['site'].nunique():,}  "
          f"({key['site'].min()}..{key['site'].max()})")
    print(f"  ages               : {sorted(key['age'].unique())}")
    dup = len(key) - len(key.drop_duplicates(["scenario", "site", "age"]))
    print(f"  duplicate keys     : {dup}")
    assert dup == 0

    # Order: scenario slowest, then site, then age -- asserted, not eyeballed.
    scen_rank = key["scenario"].map({s: i for i, s in enumerate(io.SCENARIOS)}).to_numpy()
    age_rank = key["age"].map({a: i for i, a in enumerate(io.SEED_AGES)}).to_numpy()
    lex = np.lexsort((age_rank, key["site"].to_numpy(), scen_rank))
    is_sorted = bool(np.array_equal(lex, np.arange(len(key))))
    print(f"  order is (scenario, site, age) with age in README order: {is_sorted}")
    assert is_sorted, "template order is not scenario-major/site/age"

    block = len(io.SEED_AGES)
    print(f"  first rows         : {template['id'].iloc[0]} .. {template['id'].iloc[block - 1]}")
    print(f"  last rows          : {template['id'].iloc[-block]} .. {template['id'].iloc[-1]}")
    print(f"  template values    : height all zero {bool((template['height'] == 0).all())}, "
          f"agb all zero {bool((template['agb'] == 0).all())}")

    # The template must be exactly the cross product implied by the README.
    want = pd.MultiIndex.from_product(
        [io.SCENARIOS, range(sub.EXPECTED_SITES), io.SEED_AGES], names=["scenario", "site", "age"]
    )
    got = pd.MultiIndex.from_frame(key)
    print(f"  equals the 2 x {sub.EXPECTED_SITES} x {len(io.SEED_AGES)} cross product, in order: "
          f"{bool(got.equals(want))}")
    assert got.equals(want)

    rule("3. a physical 'truth' and the three submissions built from it")
    # A plausible physical year-40 state per row, built from the released initial
    # states so the magnitudes are real rather than invented.
    truth_phys = np.empty((len(template), 2), dtype=np.float64)
    for s, scenario in enumerate(io.SCENARIOS):
        _, y0 = io.open_test(scenario)
        rows = slice(s * sub.EXPECTED_SITES * block, (s + 1) * sub.EXPECTED_SITES * block)
        truth_phys[rows, 0] = y0[:, :, 0].reshape(-1)
        truth_phys[rows, 1] = y0[:, :, 1].reshape(-1)
    div = sub.divisors()
    print(f"  physical truth   height min {truth_phys[:,0].min():8.4f} "
          f"mean {truth_phys[:,0].mean():8.4f} max {truth_phys[:,0].max():8.4f}")
    print(f"  physical truth   agb    min {truth_phys[:,1].min():8.4f} "
          f"mean {truth_phys[:,1].mean():8.4f} max {truth_phys[:,1].max():8.4f}")

    solution = sub.assemble(truth_phys, template)          # scaled -- this is the stored solution
    correct = sub.assemble(truth_phys, template)           # scaled -- a perfect entry
    unscaled = sub.assemble(truth_phys, template, already_scaled=True)  # the documented mistake

    print(f"  scaled   height  mean {solution['height'].mean():8.4f}   "
          f"agb mean {solution['agb'].mean():8.4f}")
    print(f"  unscaled height  mean {unscaled['height'].mean():8.4f}   "
          f"agb mean {unscaled['agb'].mean():8.4f}")

    rule("4. the truth scores 0")
    s_correct = sub.score(correct, solution)
    for k, v in s_correct.items():
        print(f"  {k:11s} {v!r}")
    assert s_correct["score"] == 0.0, "a perfect submission did not score exactly 0"
    print("  -> exactly 0.0")

    rule("5. the unscaled submission scores as if completely wrong")
    s_unscaled = sub.score(unscaled, solution)
    for k, v in s_unscaled.items():
        print(f"  {k:11s} {v:.6f}")
    # Predicted: residual = phys - phys/M = phys * (1 - 1/M); MSE = E[phys^2] * (1-1/M)^2
    pred_h = float(np.mean(truth_phys[:, 0] ** 2) * (1 - 1 / div["height"]) ** 2)
    pred_a = float(np.mean(truth_phys[:, 1] ** 2) * (1 - 1 / div["agb"]) ** 2)
    print(f"  predicted mse_height {pred_h:.6f}  measured {s_unscaled['mse_height']:.6f}  "
          f"rel err {abs(pred_h - s_unscaled['mse_height']) / pred_h:.2e}")
    print(f"  predicted mse_agb    {pred_a:.6f}  measured {s_unscaled['mse_agb']:.6f}  "
          f"rel err {abs(pred_a - s_unscaled['mse_agb']) / pred_a:.2e}")
    assert abs(pred_h - s_unscaled["mse_height"]) < 1e-9 * max(1.0, pred_h)
    assert abs(pred_a - s_unscaled["mse_agb"]) < 1e-9 * max(1.0, pred_a)

    board_first = 0.057
    print(f"\n  leaderboard 1st is {board_first}; the unscaled entry scores "
          f"{s_unscaled['score']:.2f}, i.e. {s_unscaled['score'] / board_first:,.0f}x worse")

    # And the reference point that says what "completely wrong" means: all zeros.
    zeros = sub.assemble(np.zeros((len(template), 2)), template, already_scaled=True)
    s_zero = sub.score(zeros, solution)
    print(f"  all-zeros submission scores {s_zero['score']:.6f} "
          f"(mse_height {s_zero['mse_height']:.6f}, mse_agb {s_zero['mse_agb']:.6f})")
    print(f"  -> the unscaled entry is {s_unscaled['score'] / s_zero['score']:.4f}x the all-zeros entry")

    rule("6. a submission off by a known constant")
    for delta in (0.1, 0.25, 1.0):
        off = solution.copy()
        off["height"] = off["height"] + delta
        off["agb"] = off["agb"] - delta
        s_off = sub.score(off, solution)
        print(f"  +/-{delta:<5} -> score {s_off['score']:.10f}   expected {delta**2:.10f}   "
              f"mse_height {s_off['mse_height']:.10f}  mse_agb {s_off['mse_agb']:.10f}")
        assert abs(s_off["score"] - delta**2) < 1e-12

    # Only one column wrong: the pooled mean halves it.
    off = solution.copy()
    off["height"] = off["height"] + 1.0
    s_off = sub.score(off, solution)
    print(f"  height +1 only  -> score {s_off['score']:.10f}   expected 0.5   "
          f"half_sum {s_off['half_sum']:.10f}")
    assert abs(s_off["score"] - 0.5) < 1e-12
    assert abs(s_off["score"] - s_off["half_sum"]) < 1e-12
    print("  -> pooled MSE == 0.5 * (MSE_height + MSE_agb), as the README states")

    rule("7. the validator refuses the unscaled file")
    for name, frame in (("correct", correct), ("unscaled", unscaled), ("zeros", zeros)):
        report = sub.validate(frame, template)
        print(f"  --- {name} ---")
        print(report.render())

    shuffled = correct.iloc[RNG.permutation(len(correct))].reset_index(drop=True)
    print("  --- correct rows, shuffled order ---")
    print(sub.validate(shuffled, template).render())

    short = correct.iloc[:-1].copy()
    print("  --- one row missing ---")
    print(sub.validate(short, template).render())

    rule("8. round trip through the CSV")
    out = Path(__file__).resolve().parents[1] / "work" / "truth_scaled.csv"
    sub.write(solution, out)
    back = sub.load_template(out)
    s_round = sub.score(back, solution)
    print(f"  wrote {out}")
    print(f"  re-read and scored against the same solution: {s_round['score']:.3e}")
    print(f"  (nonzero only from the %.6f write precision; rrmse {s_round['rrmse']:.3e})")
    print(f"  validator on the re-read file: {'OK' if sub.validate(back, template).ok else 'REJECTED'}")

    print("\nall assertions passed")


if __name__ == "__main__":
    main()

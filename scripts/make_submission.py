"""Build, validate and write a submission from a named baseline.

This exists to prove the whole path end to end -- predictions in physical units,
scaled by the competition's divisors, laid down in the template's exact row order,
and refused by the validator if any of that is wrong -- before any model is trained.

``persistence`` predicts year-40 = year-0, reading ``test_y0`` straight out of each
scenario file. It is the honest null, and on the site-held-out offline holdout it
scores 0.3115 (public) / 0.3029 (private) against a leaderboard leader of 0.057.
It is not competitive; it is a file whose plumbing is known to be correct.

Nothing here submits. It writes a file and prints its path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, submission as sub  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def persistence() -> np.ndarray:
    """Year-40 = year-0, in physical units, in template row order.

    The template is scenario-major, then site ascending, then seed age in README
    order -- asserted in ``scripts/prove_metric.py`` -- which is exactly the
    ``(site, age)`` C-order flattening of ``test_y0[:, :, :2]`` per scenario.
    """
    blocks = []
    for scenario in io.SCENARIOS:
        _, y0 = io.open_test(scenario)
        blocks.append(np.asarray(y0[:, :, :2], dtype=np.float64).reshape(-1, 2))
    return np.concatenate(blocks, axis=0)


def age_climatology() -> np.ndarray:
    """The fit sites' mean year-40 state per seed age, in physical units.

    Uses no site information, so every site of a given seed age gets the same
    answer. Scored 1.64 / 1.59 offline -- five times worse than persistence,
    which is the measurement saying site identity carries most of the signal.
    """
    means = np.empty((len(io.SEED_AGES), 2), dtype=np.float64)
    for j, age in enumerate(io.SEED_AGES):
        y = io.open_train_y(age)
        means[j] = np.asarray(y[:, 40, 0, :2], dtype=np.float64).mean(axis=0)
    n_sites = sub.EXPECTED_SITES
    one = np.broadcast_to(means, (n_sites, len(io.SEED_AGES), 2)).reshape(-1, 2)
    return np.concatenate([one, one], axis=0)


BASELINES = {"persistence": persistence, "age_climatology": age_climatology}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", choices=sorted(BASELINES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    values = BASELINES[args.baseline]()
    print(f"{args.baseline}: predictions {values.shape} in PHYSICAL units")
    print(f"  height  min {values[:,0].min():8.4f}  mean {values[:,0].mean():8.4f}  max {values[:,0].max():8.4f}")
    print(f"  agb     min {values[:,1].min():8.4f}  mean {values[:,1].mean():8.4f}  max {values[:,1].max():8.4f}")

    template = sub.load_template()
    frame = sub.assemble(values, template)  # divides by the divisors
    div = sub.divisors()
    print(f"\nscaled by height/{div['height']:.5f}, agb/{div['agb']:.5f}")

    report = sub.validate(frame, template)
    print("\nvalidator:")
    print(report.render())
    if not report.ok:
        raise SystemExit("submission REJECTED -- not written")

    out = Path(args.out) if args.out else ROOT / "work" / f"submission_{args.baseline}.csv"
    path = sub.write(frame, out)
    print(f"\nwrote {path}  ({path.stat().st_size:,} bytes)")

    # Re-read from disk and re-validate: the file, not the frame, is what is scored.
    back = sub.load_template(path)
    again = sub.validate(back, template)
    print(f"re-read from disk and re-validated: {'OK' if again.ok else 'REJECTED'}")
    print(f"\nNOT SUBMITTED. Path: {path}")


if __name__ == "__main__":
    main()

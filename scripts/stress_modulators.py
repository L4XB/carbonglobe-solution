#!/usr/bin/env python
"""Site x scenario summaries of window drought/heat stress, as modulators of the stack's old-agb collapse.

Uses the monthly forcing cache and the indices of `mort_trigger.py` (each relative to the site's own 40-year monthly
climatology, larger = more stress) and reduces them over the 40 window years: the worst value, the worst value in the
last ten years (a late event leaves a fresh loss at year 40), and the late-minus-early mean. Writes
work/stress_modulators.npz with arrays (2, 6171) per summary.

    ./.venv/bin/python scripts/stress_modulators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cgb import io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices  # noqa: E402


def main() -> None:
    out: dict[str, list[np.ndarray]] = {}
    for s in io.SCENARIOS:
        m = np.asarray(mort.open_monthly(s), dtype=np.float32)
        ix = build_indices(m)
        for k in ("wb12", "wb24", "spi12", "vpd3", "vpd12", "tmax3", "bucket"):
            v = np.nan_to_num(ix[k], nan=0.0)
            out.setdefault(f"{k}_max", []).append(v.max(axis=1))
            out.setdefault(f"{k}_late", []).append(v[:, 30:].max(axis=1))
            out.setdefault(f"{k}_trend", []).append(v[:, 30:].mean(axis=1) - v[:, :10].mean(axis=1))
        out.setdefault("tanom_trend", []).append(ix["tanom"][:, 30:].mean(axis=1) - ix["tanom"][:, :10].mean(axis=1))
        print(f"{s} done", flush=True)
    np.savez(ROOT / "work" / "stress_modulators.npz", **{k: np.stack(v) for k, v in out.items()})
    print("wrote work/stress_modulators.npz:", ", ".join(sorted(out)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""The leaderboard as a linear instrument: every scored file measures E[F.T].

On an MSE board, ``score(F) = E[F^2] - 2 E[F.T] + E[T^2]`` pooled over the 2N scored
cells. ``E[F^2]`` is our own file and ``E[T^2] = 2.588558`` was pinned by the
persistence pair (``notes/board-anchors.md``), so a board score hands back
``E[F.T]`` to within the board's rounding (scores carry three decimals, so
``|delta E[F.T]| <= 2.5e-4``). With K scored files the optimal linear blend
``sum_k w_k F_k`` (optionally with an intercept, i.e. a constant column) follows
from the Gram matrix ``G_kl = E[F_k.F_l]`` and the vector ``b_k = E[F_k.T]``:
``w* = G^-1 b``, score ``E[T^2] - b^T G^-1 b``. Nothing about the truth beyond those
K numbers is used, and the rounding propagates into an interval on the blend's score.

The same algebra restricted to a row subset (a seed-age band, a scenario, a column)
needs files that differ only there; ``scripts/band_probes.py`` writes those.

    ./.venv/bin/python scripts/board_solve.py \
        --file results/latent_v1x2_m114.csv=0.128 \
        --file results/colprobe_latent_v1x2_lama050.csv=0.123 ... [--intercept]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import shift  # noqa: E402

E_T2 = 2.588558  # E[T^2] over the 2N scored cells, from persistence x1.0 and x0.5
ROUNDING = 5e-4  # a three-decimal score is within +-5e-4 of the exact value


def load(spec: str) -> tuple[str, np.ndarray, float]:
    path, _, score = spec.rpartition("=")
    return Path(path).name, shift.read_predictions(path).astype(np.float64), float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", action="append", required=True, help="path=board_score")
    ap.add_argument("--intercept", action="store_true", help="add a constant-1 column to the blend")
    ap.add_argument("--nonneg", action="store_true", help="report also the best blend with non-negative weights (coordinate descent)")
    args = ap.parse_args()

    names, preds, scores = zip(*[load(s) for s in args.file])
    F = np.stack([p.reshape(-1) for p in preds], axis=1)  # (2N, K) pooled cells
    if args.intercept:
        F = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
        names = list(names) + ["const_1"]
        scores = list(scores) + [1.602]  # the constant-1 probe of 2026-09-18
    K = F.shape[1]
    G = F.T @ F / F.shape[0]
    e_f2 = np.diag(G)
    b = (e_f2 + E_T2 - np.asarray(scores)) / 2.0
    print(f"{'file':44s} {'score':>7s} {'E[F^2]':>8s} {'E[F.T]':>8s} {'corr(F,T)':>9s}")
    for n, s, a, bb in zip(names, scores, e_f2, b):
        print(f"{n:44s} {s:7.4f} {a:8.4f} {bb:8.4f} {bb / np.sqrt(a * E_T2):9.4f}")
    # collinear probe files (a scaled copy, a band replacement) make G singular; the pseudo-inverse
    # returns the minimum-norm optimum, and the greedy subset search below is what to read
    w = np.linalg.pinv(G, rcond=1e-8) @ b
    best = E_T2 - b @ w
    # rounding: b_k within +-ROUNDING/2 each; worst-case linearised bound on the blend score
    db = ROUNDING / 2.0
    grad = -2.0 * w  # d(best)/d(b_k) at the optimum is -2 w_k (envelope theorem)
    bound = float(np.abs(grad).sum() * db)
    print("\noptimal linear blend (unconstrained):")
    for n, wk in zip(names, w):
        print(f"  {wk:+8.4f}  {n}")
    print(f"  predicted score {best:.4f}  (rounding bound +-{bound:.4f}); sum of weights {w.sum():.4f}")
    # pairwise: best two-file blends
    if K >= 2:
        print("\nbest two-file blends:")
        rows = []
        for i in range(K):
            for j in range(i + 1, K):
                Gi = G[np.ix_([i, j], [i, j])]
                wi = np.linalg.solve(Gi, b[[i, j]])
                rows.append((E_T2 - b[[i, j]] @ wi, names[i], names[j], wi))
        for s, ni, nj, wi in sorted(rows)[:6]:
            print(f"  {s:.4f}  {wi[0]:+.3f} {ni} {wi[1]:+.3f} {nj}")
    # greedy forward selection: add the file that improves the exact blend score most, stop when the
    # gain is below the rounding noise of one more functional
    chosen: list[int] = []
    print("\ngreedy forward selection (exact blend score at each step):")
    while len(chosen) < K:
        cands = []
        for j in range(K):
            if j in chosen:
                continue
            idx = chosen + [j]
            Gi = G[np.ix_(idx, idx)]
            wi = np.linalg.pinv(Gi, rcond=1e-8) @ b[idx]
            cands.append((E_T2 - b[idx] @ wi, j, wi))
        s_new, j, wi = min(cands, key=lambda c: c[0])
        s_old = E_T2 - b[chosen] @ (np.linalg.pinv(G[np.ix_(chosen, chosen)], rcond=1e-8) @ b[chosen]) if chosen else E_T2
        if chosen and s_old - s_new < 2 * ROUNDING:
            break
        chosen.append(j)
        print(f"  + {names[j]:44s} -> {s_new:.4f}   weights {np.round(wi, 3)}")
    if args.nonneg:
        w2 = np.clip(w, 0, None)
        for _ in range(2000):
            for k in range(K):
                r = b[k] - G[k] @ w2 + G[k, k] * w2[k]
                w2[k] = max(0.0, r / G[k, k])
        print("\nbest non-negative blend:")
        for n, wk in zip(names, w2):
            if wk > 1e-4:
                print(f"  {wk:+8.4f}  {n}")
        print(f"  predicted score {E_T2 - 2 * b @ w2 + w2 @ G @ w2:.4f}")


if __name__ == "__main__":
    main()

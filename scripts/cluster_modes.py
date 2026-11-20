#!/usr/bin/env python
"""Local Laplacian modes of the main collapse cluster: the sub-regional structure of the blend's old-agb collapses.

The 5% of sites with the strongest predicted old-agb loss carry half of the blend's old-agb deviation energy, and most
of them form one connected region of the weather-neighbour graph. Global graph modes are nearly flat across it, so the
spatial probes so far put only 2-4% of their energy there. Writes the eigenvectors of the normalised Laplacian of the
cluster's own subgraph (mode 0 = uniform) as a (6171, k) array, zero outside the cluster.

    ./.venv/bin/python scripts/cluster_modes.py --cluster 0 --k 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--labels", default="work/collapse_cluster_labels.npy")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lab = np.load(ROOT / args.labels)
    nb = np.load(ROOT / "work" / "weather_neighbours.npy")
    sites = np.where(lab == args.cluster)[0]
    pos = {s: i for i, s in enumerate(sites)}
    W = np.zeros((len(sites), len(sites)))
    for s in sites:
        for t in nb[s]:
            if t in pos:
                W[pos[s], pos[t]] = W[pos[t], pos[s]] = 1.0
    deg = W.sum(1)
    if (deg == 0).any():
        raise SystemExit(f"{int((deg == 0).sum())} isolated sites in the cluster")
    Lsym = np.eye(len(sites)) - W / np.sqrt(np.outer(deg, deg))
    vals, vecs = np.linalg.eigh(Lsym)
    vecs = vecs / np.sqrt(deg)[:, None]                 # random-walk eigenvectors (mode 0 is constant)
    vecs = vecs / vecs.std(axis=0, keepdims=True).clip(1e-12)
    vecs[:, 0] = 1.0
    out = np.zeros((len(lab), args.k))
    out[sites] = vecs[:, : args.k]
    np.save(ROOT / (args.out or f"work/cluster{args.cluster}_modes.npy"), out)
    print(f"cluster {args.cluster}: {len(sites)} sites, {int(W.sum() / 2)} edges, eigenvalues {np.round(vals[: args.k], 3)}")


if __name__ == "__main__":
    main()

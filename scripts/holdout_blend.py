#!/usr/bin/env python
"""Blend the latent checkpoints and the direct GBDT on the holdout: weights fitted on the public
fold, scored on the private fold (and the reverse), so the offline complementarity is measured
before any board slot is spent on it.

    <torch+lgb venv>/bin/python scripts/holdout_blend.py --latent noco2 --gbdt gbdt1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cgb import cache, holdout, io, latent  # noqa: E402
from direct_gbdt import rows, site_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent", default="noco2", help="comma-separated latent tags (holdout preds averaged)")
    ap.add_argument("--gbdt", default="gbdt1", help="comma-separated direct tags")
    args = ap.parse_args()
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    traj = cache.open_traj()
    ann = cache.open_x_annual()
    sea = cache.open_x_season()
    preds = {}
    for fold, sites in (("public", split.public), ("private", split.private)):
        true40 = np.transpose(np.asarray(traj[:, sites, 40, :2], dtype=np.float64) / y_mean[:2], (1, 0, 2))
        u0 = np.transpose(np.asarray(traj[:, sites, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))
        pers = u0[:, :, :2]
        bare = u0[:, :, 0] * y_mean[0] < 1.0
        members = {}
        for tag in args.latent.split(","):
            p = np.load(io.ROOT / "work" / "models" / f"latent_{tag}" / f"pred_{fold}.npy")[:, :, -1, :2]  # (15, n, 2) year 40
            p = np.transpose(p, (1, 0, 2)).astype(np.float64)
            p[bare] = 0.0
            members[f"latent_{tag}"] = p
        feat = site_features(np.asarray(ann[sites]), np.asarray(sea[sites]))
        X = rows(feat, u0, agef)
        for tag in args.gbdt.split(","):
            d = io.ROOT / "work" / "models" / f"direct_{tag}"
            ch = np.stack([lgb.Booster(model_file=str(d / f"{k}.txt")).predict(X) for k in ("height", "agb")], axis=1)
            p = np.clip(X[:, -19:-15][:, :2] + ch, 0, None).reshape(15, len(sites), 2).transpose(1, 0, 2)
            p[bare] = 0.0
            members[f"direct_{tag}"] = p
        members["persistence"] = pers
        preds[fold] = (members, true40)
        for k, p in members.items():
            print(f"  {fold}: {k:16s} {np.mean((p - true40) ** 2):.5f}")
    names = [k for k in preds["public"][0] if k != "persistence"]
    # least-squares blend weights (per column) fitted on one fold, applied to the other
    for fit_fold, eval_fold in (("public", "private"), ("private", "public")):
        m_fit, t_fit = preds[fit_fold]
        m_ev, t_ev = preds[eval_fold]
        out = np.zeros_like(t_ev)
        ws = {}
        for k in range(2):
            A = np.stack([m_fit[n][..., k].reshape(-1) for n in names], axis=1)
            w, *_ = np.linalg.lstsq(A, t_fit[..., k].reshape(-1), rcond=None)
            ws[k] = w
            out[..., k] = sum(wi * m_ev[n][..., k] for wi, n in zip(w, names))
        eq = sum(m_ev[n] for n in names) / len(names)
        print(f"fit on {fit_fold}, eval {eval_fold}: weights h {np.round(ws[0], 3)} a {np.round(ws[1], 3)} -> "
              f"blend {np.mean((out - t_ev) ** 2):.5f} | equal-weight mean {np.mean((eq - t_ev) ** 2):.5f}")


if __name__ == "__main__":
    main()

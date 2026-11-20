#!/usr/bin/env python
"""Stage-10 check for a direct GBDT: start at year slot 10 with forcing summaries over years 10-40 (the 40-year
summaries the model was trained on are replaced by the window's), score against slot 40 on the public fold."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cgb import cache, holdout, io, latent  # noqa: E402
import direct_gbdt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="gbdt2")
    ap.add_argument("--s0", type=int, default=10)
    args = ap.parse_args()
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42); sites = split.public
    ann = np.asarray(cache.open_x_annual()[sites]); sea = np.asarray(cache.open_x_season()[sites])
    traj = cache.open_traj(); ages = np.array(io.SEED_AGES); old = ages >= 150
    u_all = np.transpose(np.asarray(traj[:, sites, :, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2, 3))
    true40 = u_all[:, :, 40, :2]
    for tag in args.tags.split(","):
        d = io.ROOT / "work" / "models" / f"direct_{tag}"
        boosters = [lgb.Booster(model_file=str(d / f"{k}.txt")) for k in ("height", "agb")]
        for s0 in (0, args.s0):
            a_win = ann[:, s0:, :]
            feat = direct_gbdt.site_features(a_win, sea)  # summaries over the window only
            X = direct_gbdt.rows(feat, u_all[:, :, s0, :], agef)
            ch = np.stack([b.predict(X) for b in boosters], axis=1)
            p40 = np.clip(X[:, -19:-15][:, :2] + ch, 0, None).reshape(15, len(sites), 2).transpose(1, 0, 2)
            pers = u_all[:, :, s0, :2]
            p40[pers[:, :, 0] * y_mean[0] < 1.0] = 0.0
            se = (p40 - true40) ** 2; sp = (pers - true40) ** 2
            print(f"{tag:8s} from slot {s0:2d}: score {se.mean():.4f} (old {se[:, old].mean():.4f}) | persistence {sp.mean():.4f} (old {sp[:, old].mean():.4f}) | "
                  f"old-band explained {1 - se[:, old].mean() / sp[:, old].mean():.3f} | mean pred old agb change {np.mean(p40[:, old, 1] - pers[:, old, 1]):+.4f} true {np.mean(true40[:, old, 1] - pers[:, old, 1]):+.4f}", flush=True)


if __name__ == "__main__":
    main()

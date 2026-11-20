#!/usr/bin/env python
"""Score a latent checkpoint from a later start slot on the public fold: the training-side twin of
the competition's partially relaxed old stands. ``<torch venv>/bin/python scripts/eval_stage.py --tags noco2,clim --s0 10``"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import holdout, io, latent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="noco2,clim")
    ap.add_argument("--s0", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cpu")
    split = holdout.make_split(seed=42)
    arr = latent.load_arrays(split.fit)
    sites = split.public
    ages = np.array(io.SEED_AGES)
    for tag in args.tags.split(","):
        ckpt = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = latent.LatentRollout(n_drv=136 + (latent.N_MONTHLY_INPUTS if a.get("monthly") else 0), d_enc=a.get("d_enc", 128), d_h=a.get("d_h", 256), n_layers=a.get("layers", 2), use_t=not a.get("no_t", False))
        model.load_state_dict(ckpt["model"])
        input_set = a.get("input_set", "noco2" if a.get("drop_co2") else "full")
        x_year, season = arr.x_year.copy(), arr.season.copy()
        latent.apply_input_set(x_year, season, input_set)
        arr2 = latent.Arrays(x_year, season, arr.static, arr.u, arr.agef, arr.y_mean, arr.lat, arr.season_mu, arr.season_sd)
        true40 = np.transpose(arr.u[:, sites, 40, :2], (1, 0, 2))
        for s0 in (0, args.s0):
            pred = latent.rollout_sites(model, arr2, sites, device, s0=s0)
            p40 = np.transpose(pred[:, :, -1, :2], (1, 0, 2))
            pers = np.transpose(arr.u[:, sites, s0, :2], (1, 0, 2))
            se = (p40 - true40) ** 2; sp = (pers - true40) ** 2
            old = ages >= 150
            print(f"{tag:10s} from slot {s0:2d}: score {se.mean():.4f} (old band {se[:, old].mean():.4f}, young {se[:, ages <= 30].mean():.4f}) | "
                  f"persistence {sp.mean():.4f} (old {sp[:, old].mean():.4f}) | old-band explained {1 - se[:, old].mean() / sp[:, old].mean():.3f} | "
                  f"mean pred change old agb {np.mean(p40[:, old, 1] - pers[:, old, 1]):+.4f} true {np.mean(true40[:, old, 1] - pers[:, old, 1]):+.4f}", flush=True)


if __name__ == "__main__":
    main()

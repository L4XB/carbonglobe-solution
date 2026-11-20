"""Score an average of latent checkpoints on the site-held-out split, both folds.

Same breakdowns as the trainers (``latent.breakdown``), so the ensemble row sits
next to the single-seed rows in ``notes/latent-vs-control.md``. Also reports the
per-seed rows and the pairwise agreement, since a deterministic target with a
seed-dependent answer would mean the model is fitting something other than the map.

    <torch venv>/bin/python scripts/ensemble_holdout.py --tags v1,v1s43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import holdout, io, latent  # noqa: E402


def load_model(tag: str, device: torch.device) -> latent.LatentRollout:
    ck = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = latent.LatentRollout(d_enc=a.get("d_enc", 128), d_h=a.get("d_h", 256), n_layers=a.get("layers", 2)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="v1,v1s43")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    tags = args.tags.split(",")
    split = holdout.make_split(seed=42)
    arr = latent.load_arrays(split.fit)
    models = [load_model(t, device) for t in tags]
    for fold, sites in (("public", split.public), ("private", split.private)):
        true = arr.u[:, sites]
        true40 = np.transpose(true[:, :, 40, :2], (1, 0, 2))
        pers40 = np.transpose(true[:, :, 0, :2], (1, 0, 2))
        persist_loss = ((pers40 - true40) ** 2).reshape(len(sites), -1).mean(1)
        preds = [np.transpose(latent.rollout_sites(m, arr, sites, device)[:, :, -1, :2], (1, 0, 2)) for m in models]
        print(f"--- {fold} fold, {len(sites):,} sites ---")
        rows = [latent.breakdown(f"latent_{t}", pr, true40, sites, arr, persist_loss) for t, pr in zip(tags, preds)]
        mean = np.mean(preds, axis=0)
        rows.append(latent.breakdown(f"ensemble_{len(tags)}", mean, true40, sites, arr, persist_loss))
        for r in rows:
            latent.print_row(r)
        if len(preds) > 1:
            diffs = [float(np.mean((preds[i] - preds[j]) ** 2)) for i in range(len(preds)) for j in range(i + 1, len(preds))]
            print(f"  seed disagreement, mean squared difference between predictions: {np.mean(diffs):.5f} (vs error {rows[-1]['score']:.4f})")


if __name__ == "__main__":
    main()

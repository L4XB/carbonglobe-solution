#!/usr/bin/env python
"""The collapse detector on quantile-mapped drought indices.

The competition forcing has twice the training's water-balance extremes (wb12 beyond the holdout q99 on 1.9-2.2%
of site-years against 1.0%) and a tenth of its precipitation droughts; the detector counts exceedances of
training thresholds, so on the scenarios it fires where the GCM's trend-driven heat and demand pile up. If ED
collapses a similar share of stands as in training, the events follow the *relative* stress. Each drought
index of the scenarios is mapped quantile-by-quantile (pooled over sites and years) onto the training
distribution before the detector's summaries are built; the rest of the pipeline is unchanged.

    <lightgbm venv>/bin/python scripts/detector_qmap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, shift  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_detector import M_ALL, SEQ_KEYS, climatology, seq_features  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

OUT = ROOT / "work" / "models" / "mort_detector"


def qmap(x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    qs = np.linspace(0, 1, 2001)
    s = np.nanquantile(src, qs); d = np.nanquantile(dst, qs)
    s, idx = np.unique(s, return_index=True)
    return np.interp(x, s, d[idx])


def main() -> None:
    spec = json.load(open(OUT / "gbdt_position_free_features.json"))
    names, cols, thr = spec["names"], spec["cols"], {k: tuple(v) for k, v in spec["thr"].items()}
    booster = lgb.Booster(model_file=str(OUT / "gbdt_position_free.txt"))
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    ix_tr = build_indices(np.asarray(mort.open_monthly("train")[split.fit[::4]]))
    old = [io.SEED_AGES.index(a) for a in (150, 200, 250, 300, 350, 400, 450, 500)]
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([fr.reshape(-1, 7) for fr in frames.values()]).min(axis=0)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys()
    sc_k, site_k, age_k = keys["scenario"].to_numpy(), keys["site"].to_numpy(), keys["age"].to_numpy()
    F = R.copy(); Fd = R.copy()
    for sc in io.SCENARIOS:
        mt = np.asarray(mort.open_monthly(sc))
        xt = np.load(cache.CACHE_DIR / f"test_{sc}_annual.npy", mmap_mode="r")
        static = np.asarray(xt[:, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        ix = build_indices(mt); ix.pop("_pet_over_p", None)
        ixq = dict(ix)
        for k in SEQ_KEYS:
            ixq[k] = qmap(ix[k], ix[k].ravel(), ix_tr[k].ravel())
            print(f"{sc} {k}: share above the training q99 raw {np.nanmean(ix[k] > thr[k][1]):.4f} -> mapped {np.nanmean(ixq[k] > thr[k][1]):.4f}")
        clim = climatology(mt)
        raw0 = M_ALL * (frames[sc] - floors)
        n = raw0.shape[0]
        state0 = (np.transpose(raw0[:, old, :4], (1, 0, 2)) / y_mean[:4]).reshape(8 * n, 4)
        veg = (np.transpose(raw0[:, old, 1]) > mort.BARE_AGB).reshape(-1)
        preds = {}
        for lab, ixx in (("raw", ix), ("qmap", ixq)):
            seq, seq_names = seq_features(ixx, thr)
            sx = np.concatenate([seq, clim, static], axis=1)
            x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
            p = np.zeros(len(veg)); p[veg] = booster.predict(x[veg][:, cols])
            preds[lab] = p.reshape(8, n)
        d = preds["qmap"] - preds["raw"]
        print(f"{sc}: mean predicted old agb change raw {preds['raw'].mean():+.4f} qmap {preds['qmap'].mean():+.4f}; corr {np.corrcoef(preds['raw'].ravel(), preds['qmap'].ravel())[0, 1]:.3f}; sd of the difference {d.std():.4f}")
        m = sc_k == sc
        for i, a in enumerate((150, 200, 250, 300, 350, 400, 450, 500)):
            r = np.where(m & (age_k == a))[0]; s = site_k[r]
            F[r, 1] = R[r, 1] + preds["qmap"][i, s]
            Fd[r, 1] = R[r, 1] + d[i, s]
    shift.write_predictions(F, ROOT / "results" / "detprobe_old_agb_qmap.csv")
    shift.write_predictions(Fd, ROOT / "results" / "detprobe_old_agb_qmapleg.csv")
    print("wrote results/detprobe_old_agb_qmap.csv and _qmapleg.csv; nothing submitted")


if __name__ == "__main__":
    main()

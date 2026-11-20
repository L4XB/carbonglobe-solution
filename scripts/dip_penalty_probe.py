#!/usr/bin/env python
"""The stack's spurious spin-up-dip penalty: a counterfactual without it.

In training a site whose spin-up agb profile has a >10% drop between consecutive seed ages collapses in the window
4x as often (old stands 48% vs 11%), because the training spin-up shares the window's climate regime; the models
learned "jagged profile -> poor outcome". The test spin-up is harsher than its window (3x the dips), so on the test
the penalty is spurious. Counterfactual: a LightGBM of the stack's ratio B / (0.876 R) on hour-invariant climate
normals, soil, the row's decoded state, seed age and scenario, fitted on rows of NON-dip sites outside the cluster,
predicts what the stack would say at the dip sites without the dip cue. D = (counterfactual - actual) x 0.876 R on
dip-site rows (outside the cluster), per block.

    python scripts/dip_penalty_probe.py --blocks oa,ma,ya
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

BL = {"ya": ("young", 1), "yh": ("young", 0), "ma": ("mid", 1), "mh": ("mid", 0), "oa": ("old", 1), "oh": ("old", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", default="oa,ma,ya")
    ap.add_argument("--tag", default="dip_penalty")
    ap.add_argument("--stress", action="store_true", help="add window stress indices (SPI, water balance, VPD) to the counterfactual")
    args = ap.parse_args()
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    B = rd(ROOT / "results/stack_20261010.csv"); R = rd(ROOT / "results/latent_ref_y0_m114114.csv")
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    s5 = (keys["scenario"] == "ssp585").to_numpy().astype(int)
    band = shift.band_of(age); cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    ages = np.asarray(io.SEED_AGES, float); ai = np.searchsorted(ages, age)
    prof = np.zeros((2, 6171, 15)); prof[s5, site, ai] = R[:, 1]
    rel = prof[:, :, 1:] / np.maximum(prof[:, :, :-1], 1e-6)
    dips = ((rel < 0.9) & (prof[:, :, :-1] > 0.05)).sum(2)[s5, site]
    # hour-invariant normals + soil per scenario/site
    feats = []
    for s in io.SCENARIOS:
        a = np.load(ROOT / f"work/cache/test_{s}_annual.npy").astype(np.float64)
        cols = [a[:, :, 0:8].mean(1), a[:, :, 0:8].std(1)] + [a[:, :, st + 4 * np.arange(24)].mean((1, 2))[:, None] for st in (33, 34, 35, 36)]
        cols.append(a[:, 0, 129:136])
        feats.append(np.concatenate(cols, 1))
    if args.stress:
        # window stress indices, so the counterfactual removes only the profile cue, not real window stress
        sys.path.insert(0, str(ROOT / "scripts"))
        from cgb import mortality as mort
        from mort_trigger import build_indices
        for si, s in enumerate(io.SCENARIOS):
            ix = build_indices(np.asarray(mort.open_monthly(s)))
            extra = [np.nanmax(np.asarray(ix[k]), 1) for k in ("spi12", "spi6", "spi3", "wb12", "vpd3", "vpd1", "dryrun") if k in ix]
            extra += [np.nanmean(np.asarray(ix[k]), 1) for k in ("spi12", "spi6", "vpd3") if k in ix]
            feats[si] = np.concatenate([feats[si], np.nan_to_num(np.stack(extra, 1))], 1)
    Fs = np.stack(feats)                                  # (2, 6171, F)
    st = np.zeros((2, 6171, 15, 4))
    for si, s in enumerate(io.SCENARIOS):
        y0 = np.asarray(io.open_test(s)[1], dtype=np.float64)
        st[si] = 1.14114 * np.clip(y0[:, :, :4] - shift.FLOORS[:4], 0, None)
    X = np.concatenate([Fs[s5, site], st[s5, site, ai], np.log(ages[ai])[:, None], s5[:, None]], 1)
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=10, seed=42)
    D = np.zeros_like(B)
    for blk in args.blocks.split(","):
        b, c = BL[blk]
        m = (band == b) & ~cl & (R[:, c] > 0.05)
        ratio = B[:, c] / (0.876 * np.maximum(R[:, c], 1e-6))
        tr = m & (dips == 0); te = m & (dips > 0)
        # out-of-fold check on non-dip sites (site split) so the counterfactual's own error is known
        fold = site % 4
        oof = np.zeros(len(B))
        for k in range(4):
            mk = lgb.train(prm, lgb.Dataset(X[tr & (fold != k)], ratio[tr & (fold != k)]), 400)
            oof[tr & (fold == k)] = mk.predict(X[tr & (fold == k)])
        r2 = 1 - np.mean((oof[tr] - ratio[tr]) ** 2) / np.var(ratio[tr])
        md = lgb.train(prm, lgb.Dataset(X[tr], ratio[tr]), 400)
        cf = md.predict(X[te])
        D[te, c] = (cf - ratio[te]) * 0.876 * R[te, c]
        print(f"{blk}: non-dip rows {tr.sum()}, dip rows {te.sum()}; counterfactual site-split R2 {r2:.3f}; "
              f"dip rows: stack ratio {ratio[te].mean():.3f} -> counterfactual {cf.mean():.3f} (non-dip mean {ratio[tr].mean():.3f}); "
              f"raw E {np.sum(D[te, c] ** 2) / B.size:.5f}")
    np.save(ROOT / f"work/dir_{args.tag}.npy", D)
    print(f"wrote work/dir_{args.tag}.npy, raw E {np.mean(D ** 2):.5f}")


if __name__ == "__main__":
    main()

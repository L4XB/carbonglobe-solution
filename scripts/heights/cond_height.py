"""Height of window year 40 GIVEN the agb of year 40, learned on the training truth (54152 sites x 15 seed ages, December of year 0 -> December of year 40).

The stack predicted both columns from the drivers; since 11-10 the agb column is known far better (ED / ORNL composite), and the height channel is a
structural function of the stand (area-weighted height of the tallest cohort per patch): forest rows that lose agb through fire GAIN height, burned-state
rows (tall survivors, little biomass) LOSE it. This model transfers that coupling: features = the seven y0 channels, the seed age and the year-40 agb.

    python scripts/heights/cond_height.py --fit                      # fit + validation report, model -> work/heights/cond_height.txt
    python scripts/heights/cond_height.py --agb results/cand48_A.csv --tag A   # test heights given that file's agb -> work/heights/hcond_A.npy (solution units)
"""
from __future__ import annotations
import argparse, sys, json, random
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from cgb import io, shift  # noqa: E402

SEED = 42
DECODE_MULT = np.array([1.14114, 1.14114 * 1.055, 1.14114, 1.0, 1.14114, 1.14114, 1.14114])   # released test channel -> physical (scripts/ed/make_site_inputs.py; agb x 1.055 = 1 / 1.204 coding)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "heights"
FL = shift.FLOORS
DIV_H, DIV_A = 10.90645, 3.39567
FEATS = ["h0", "a0", "soil0", "lai0", "gpp0", "npp0", "rh0", "log_age", "hc", "ac", "hc_minus_h0", "h0_per_a0", "a40", "da", "dlog", "dac", "dlogc"]
AGB_FEATS = ("a40", "da", "dlog", "dac", "dlogc")
AGES = np.asarray(io.SEED_AGES, dtype=np.float64)


def chrono(prof: np.ndarray) -> np.ndarray:
    """prof: (sites, 15) values of one channel over the seed ages -> the same site's lookup value at seed age + 40 (linear in log age, flat beyond 500)."""
    la = np.log(AGES); x = np.log(np.minimum(AGES + 40.0, 500.0))
    j = np.clip(np.searchsorted(la, x, side="right") - 1, 0, len(la) - 2); w = (x - la[j]) / (la[j + 1] - la[j])
    return prof[:, j] * (1.0 - w) + prof[:, j + 1] * w


def features(y0: np.ndarray, age: np.ndarray, a40: np.ndarray, hc: np.ndarray, ac: np.ndarray) -> np.ndarray:
    """PHYSICAL units. y0: (n, 7), age: (n,), a40: agb of year 40 (kgC), hc / ac: the site's y0 height / agb at age + 40 -> (n, len(FEATS))."""
    a0 = y0[:, 1]
    dlog = np.log((np.maximum(a40, 0) + 0.05) / (np.maximum(a0, 0) + 0.05)); dlogc = np.log((np.maximum(a40, 0) + 0.05) / (np.maximum(ac, 0) + 0.05))
    return np.column_stack([y0, np.log(age), hc, ac, hc - y0[:, 0], y0[:, 0] / (np.maximum(a0, 0) + 0.3), a40, a40 - a0, dlog, a40 - ac, dlogc]).astype(np.float32)


def training_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All training rows: y0 (n, 7), y40 (n, 7), age (n,), site (n,), hc (n,), ac (n,); site-major order."""
    y0 = np.empty((54152, 15, 7), dtype=np.float32); y40 = np.empty((54152, 15, 7), dtype=np.float32)
    for j, a in enumerate(io.SEED_AGES):
        y = io.open_train_y(a)                                        # (54152, 41, 12, 7)
        y0[:, j] = y[:, 0, 11, :]; y40[:, j] = y[:, 40, 11, :]
    hc = chrono(y0[:, :, 0].astype(np.float64)); ac = chrono(y0[:, :, 1].astype(np.float64))
    age = np.broadcast_to(AGES.astype(np.float32), (54152, 15)); site = np.broadcast_to(np.arange(54152)[:, None], (54152, 15))
    return y0.reshape(-1, 7), y40.reshape(-1, 7), age.reshape(-1).copy(), site.reshape(-1).copy(), hc.reshape(-1), ac.reshape(-1)


def report(name: str, pred: np.ndarray, truth: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    print(f"   {name:34s} " + " | ".join(f"{k} {np.sqrt(np.mean(((pred - truth)[m] / (DIV_H * shift.M_HEIGHT)) ** 2)):.4f}" for k, m in masks.items()))


def fit() -> None:
    import lightgbm as lgb
    random.seed(SEED); np.random.seed(SEED)
    y0, y40, age, site, hc, ac = training_rows()
    ok = np.isfinite(y0).all(1) & np.isfinite(y40).all(1) & np.isfinite(hc) & np.isfinite(ac)
    y0, y40, age, site, hc, ac = y0[ok], y40[ok], age[ok], site[ok], hc[ok], ac[ok]
    rng = np.random.default_rng(SEED); val_sites = rng.random(54152) < 0.2; va = val_sites[site]; tr = ~va
    X = features(y0, age, y40[:, 1], hc, ac); t = (y40[:, 0] - hc).astype(np.float32)
    print(f"training rows {int(tr.sum())}, validation rows {int(va.sum())} (sites split 80 / 20), features {len(FEATS)}")
    params = dict(objective="l2", learning_rate=0.05, num_leaves=255, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, num_threads=6, verbose=-1)
    cols_noa = [i for i, f in enumerate(FEATS) if f not in AGB_FEATS]; cols_ha = [i for i, f in enumerate(FEATS) if f not in ("soil0", "lai0", "gpp0", "npp0", "rh0")]
    res = {}
    for nm, cols, save in (("with the year-40 agb", list(range(len(FEATS))), "cond_height.txt"), ("height + agb channels only", cols_ha, "cond_height_ha.txt"), ("y0 profile + age only", cols_noa, "")):
        d_tr = lgb.Dataset(X[tr][:, cols], t[tr]); d_va = lgb.Dataset(X[va][:, cols], t[va])
        m = lgb.train(params, d_tr, num_boost_round=2500, valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        res[nm] = hc[va] + m.predict(X[va][:, cols], num_iteration=m.best_iteration); print(f"   {nm}: best iteration {m.best_iteration}")
        if save:
            OUT.mkdir(parents=True, exist_ok=True); m.save_model(str(OUT / save), num_iteration=m.best_iteration)
            imp = m.feature_importance("gain"); print("   gain shares: " + ", ".join(f"{FEATS[c]} {imp[i] / imp.sum():.2f}" for i, c in enumerate(cols)))
    h0p = y0[va, 0]; a0p = y0[va, 1]; a40p = y40[va, 1]
    dl = np.log((a40p + 0.05) / (a0p + 0.05)); old = age[va] >= 200
    masks = {"all": np.ones(int(va.sum()), dtype=bool), "old": old, "mid 50-150": (age[va] >= 50) & (age[va] <= 150), "young 10-30": (age[va] >= 10) & (age[va] <= 30), "age 1": age[va] == 1,
             "old quiet (|dlog| < 0.1)": old & (np.abs(dl) < 0.1) & (a0p > 5.5), "forest loss (dlog < -0.3)": (a0p > 5.5) & (dl < -0.3),
             "burned state": (h0p >= 30) & (a0p <= 5.5) & (a0p > 0.2), "burned + loss": (h0p >= 30) & (a0p <= 5.5) & (a0p > 0.2) & (dl < -0.3)}
    print("validation rms of the year-40 height in SOLUTION units (rows: " + ", ".join(f"{k} {int(m.sum())}" for k, m in masks.items()) + ")")
    report("persistence (y0 height)", y0[va, 0], y40[va, 0], masks); report("chronosequence (y0 at age + 40)", hc[va], y40[va, 0], masks)
    for nm, p in res.items(): report(nm, p, y40[va, 0], masks)


def predict(agb_file: str, tag: str, model: str = "cond_height.txt") -> None:
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(OUT / model)); cols = [FEATS.index(f) for f in m.feature_name()] if m.feature_name()[0] in FEATS else None
    if cols is None:      # models saved from arrays carry Column_i names: the height + agb variant has 12 features
        cols = list(range(len(FEATS))) if m.num_feature() == len(FEATS) else [i for i, f in enumerate(FEATS) if f not in ("soil0", "lai0", "gpp0", "npp0", "rh0")]
    keys = shift.template_keys(); P = shift.read_predictions(agb_file).astype(np.float64)
    out = np.full(len(keys), np.nan)
    for sc in io.SCENARIOS:
        _, y0 = io.open_test(sc); y0 = DECODE_MULT * np.clip(np.asarray(y0, dtype=np.float64) - FL, 0.0, None)        # (6171, 15, 7) physical
        hc = chrono(y0[:, :, 0]); ac = chrono(y0[:, :, 1])
        rows = (keys["scenario"] == sc).to_numpy(); site = keys.loc[rows, "site"].to_numpy(); ai = np.searchsorted(np.asarray(io.SEED_AGES), keys.loc[rows, "age"].to_numpy())
        a40 = DECODE_MULT[1] * np.clip(P[rows, 1] * DIV_A - FL[1], 0.0, None)                                          # solution agb keeps the floor: released = solution x divisor
        X = features(y0[site, ai], keys.loc[rows, "age"].to_numpy().astype(np.float64), a40, hc[site, ai], ac[site, ai])
        out[rows] = (hc[site, ai] + m.predict(X[:, cols])) / (shift.M_HEIGHT * DIV_H)
    out = np.clip(out, 0.0, None); np.save(OUT / f"hcond_{tag}.npy", out)
    print(f"wrote work/heights/hcond_{tag}.npy ({model}): mean {out.mean():.4f} (file's own height column {P[:, 0].mean():.4f}); rms difference to the file {np.sqrt(np.mean((out - P[:, 0]) ** 2)):.4f}, squared units {np.sum((out - P[:, 0]) ** 2):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true"); ap.add_argument("--agb", default=""); ap.add_argument("--tag", default="x"); ap.add_argument("--model", default="cond_height.txt")
    a = ap.parse_args()
    if a.fit: fit()
    if a.agb: predict(a.agb, a.tag, a.model)

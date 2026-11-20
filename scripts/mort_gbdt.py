#!/usr/bin/env python
"""The collapse detector as a gradient-boosted model, and the year-axis check.

Two questions ``mort_detector.py`` (a ridge on a binned basis) left open:

  1. Does a model that can learn sharp thresholds and interactions change the
     verdict? The ridge removed 87 % of the top-10 % persistence loss at the old
     ages against the latent rollout's 97 %, and added nothing in a blend.
  2. Is the detector keyed on driver *values* or on the *position* of a year?
     ``mort_calendar.py`` dated the training forcing to 1982-2021, so a model
     that learns "years 2, 7, 11 and 34 are bad" learns the calendar and would
     transfer to nothing under the SSP scenarios. The "position-free" variant here
     keeps only summaries that are invariant to any permutation of the 40 years
     (maxima, counts above thresholds, exceedance sums); the full variant adds the
     band maxima (years 1-3, 4-10, ...), the year of the maximum and the last
     year's value, which encode the timing relative to the scored year 40.

Same split, rows, target and evaluation as ``mort_detector.py``. Early stopping
uses a tenth of the *fit* sites, so the public and private folds stay untouched.

Interpreter: an isolated scratchpad venv (``uv venv``, numpy/pandas/lightgbm/
scikit-learn), not the project's ``./.venv`` -- the numbers in the log name the
lightgbm version that produced them.

    <scratchpad>/venvs/cgb-mort/bin/python scripts/mort_gbdt.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_detector import DROP_1KG, M_ALL, SEQ_KEYS, site_features  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
METRICS = io.ROOT / "work" / "metrics"
OUT_DIR = io.ROOT / "work" / "models" / "mort_detector"
POSITIONAL = ("_max_y", "_argmax", "_y40")
PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
    num_threads=8,
)


def rows_for(sites: np.ndarray, monthly: np.ndarray, xa: np.ndarray, traj: np.ndarray, old: list[int], thr, y_mean, div):
    sx, names = site_features(np.asarray(monthly[sites]), np.asarray(xa[sites, 0], dtype=np.float64)[:, list(cache.STATIC_DRIVERS)], thr)
    st = np.asarray(traj[old][:, sites], dtype=np.float64)
    n = len(sites)
    state0 = (st[:, :, 0, :4] / y_mean[:4]).reshape(8 * n, 4)
    x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
    names = names + ["y0_height", "y0_agb", "y0_soil", "y0_lai", "age_idx"]
    dy = ((st[:, :, 40, 1] - st[:, :, 0, 1]) / div[1]).reshape(-1)
    veg = (st[:, :, 0, 1] > mort.BARE_AGB).reshape(-1)
    return x, names, dy, veg


def main() -> None:
    t0 = time.time()
    print(f"lightgbm {lgb.__version__}, numpy {np.__version__}, python {sys.version.split()[0]} at {sys.executable}")
    div = cache.scoring_divisors()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()
    xa = cache.open_x_annual()
    monthly = mort.open_monthly("train")
    ix_fit = build_indices(np.asarray(monthly[split.fit[::4]]))
    thr = {k: (float(np.nanquantile(ix_fit[k], 0.9)), float(np.nanquantile(ix_fit[k], 0.99))) for k in SEQ_KEYS}
    del ix_fit

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(split.fit))
    n_es = len(split.fit) // 10
    es_sites, fit_sites = np.sort(split.fit[perm[:n_es]]), np.sort(split.fit[perm[n_es:]])
    data = {
        "fit": rows_for(fit_sites, monthly, xa, traj, old, thr, y_mean, div),
        "es": rows_for(es_sites, monthly, xa, traj, old, thr, y_mean, div),
        "public": rows_for(split.public, monthly, xa, traj, old, thr, y_mean, div),
        "private": rows_for(split.private, monthly, xa, traj, old, thr, y_mean, div),
    }
    names = data["fit"][1]
    print(f"rows built: fit {data['fit'][3].sum():,} vegetated, early-stop {data['es'][3].sum():,}; {len(names)} features  ({time.time() - t0:.0f}s)")

    variants = {
        "gbdt_full": [i for i, n in enumerate(names)],
        "gbdt_position_free": [i for i, n in enumerate(names) if not any(p in n for p in POSITIONAL)],
        "gbdt_no_drought_seq": [i for i, n in enumerate(names) if not any(n.startswith(k) for k in SEQ_KEYS)],
    }
    preds: dict[str, dict[str, np.ndarray]] = {"public": {}, "private": {}}
    truth = {f: data[f][2] for f in ("public", "private")}
    vegf = {f: data[f][3] for f in ("public", "private")}
    models = {}
    for vname, cols in variants.items():
        xf, _, yf, vf = data["fit"]
        xe, _, ye, ve = data["es"]
        dtrain = lgb.Dataset(xf[vf][:, cols], yf[vf], feature_name=[names[i] for i in cols], categorical_feature=["age_idx"] if "age_idx" in [names[i] for i in cols] else [])
        dval = lgb.Dataset(xe[ve][:, cols], ye[ve], reference=dtrain)
        booster = lgb.train(PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
        models[vname] = (booster, cols)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(OUT_DIR / f"{vname}.txt"), num_iteration=booster.best_iteration)
        (OUT_DIR / f"{vname}_features.json").write_text(json.dumps({"names": [names[i] for i in cols], "cols": cols, "thr": thr}))
        for f in ("public", "private"):
            x, _, _, v = data[f]
            p = np.zeros(len(v))
            p[v] = booster.predict(x[v][:, cols], num_iteration=booster.best_iteration)
            preds[f][vname] = p
        print(f"  {vname:22s} {len(cols):3d} features, best iteration {booster.best_iteration}  ({time.time() - t0:.0f}s)")
    imp = models["gbdt_full"][0].feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:15]
    print("  top gain features (full): " + ", ".join(f"{names[models['gbdt_full'][1][i]]} {imp[i] / imp.sum():.3f}" for i in order))

    for f, sites in (("public", split.public), ("private", split.private)):
        p = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{f}.npy")[:, :, -1, 1].astype(np.float64)
        y0 = np.asarray(traj[old][:, sites], dtype=np.float64)[:, :, 0, 1] / div[1]
        preds[f]["latent"] = (p[old] - y0).reshape(-1)
        preds[f]["persistence"] = np.zeros_like(preds[f]["latent"])
    for fit_fold, eval_fold in (("public", "private"), ("private", "public")):
        for vname in ("gbdt_full", "gbdt_position_free"):
            a = np.stack([preds[fit_fold]["latent"], preds[fit_fold][vname]], axis=1)
            w, *_ = np.linalg.lstsq(a, truth[fit_fold], rcond=None)
            preds[eval_fold][f"blend_{vname}"] = w[0] * preds[eval_fold]["latent"] + w[1] * preds[eval_fold][vname]
            preds[eval_fold][f"blend_{vname}_w"] = w
    blend_w = {v: {ef: preds[ef][f"blend_{v}_w"].tolist() for ef in ("public", "private")} for v in ("gbdt_full", "gbdt_position_free")}

    payload: dict[str, object] = {"lightgbm": lgb.__version__, "params": PARAMS, "n_features": {k: len(v) for k, v in variants.items()}, "blend_weights": blend_w}
    print("\n== old seed ages (>= 150), agb column, scaled squared error of the year-40 prediction ==")
    print(f"  {'fold':8s} {'model':26s} {'all rows':>9s} {'top-10% persist':>15s} {'event rows':>11s} {'other rows':>11s} {'pred rate':>9s}")
    for f in ("public", "private"):
        t = truth[f]
        top = (t ** 2) > np.quantile(t ** 2, 0.9)
        ev = t < -DROP_1KG
        payload[f] = {}
        for name in ("persistence", "gbdt_no_drought_seq", "gbdt_position_free", "gbdt_full", "latent", "blend_gbdt_full", "blend_gbdt_position_free"):
            p = preds[f][name]
            e = (p - t) ** 2
            payload[f][name] = {"all": float(e.mean()), "top10": float(e[top].mean()), "event": float(e[ev].mean()), "other": float(e[~ev].mean()), "pred_rate": float((p < -DROP_1KG).mean())}
            extra = f"   w(latent, gbdt) = {np.round(preds[f][name + '_w'], 3)}" if name.startswith("blend") else ""
            print(f"  {f:8s} {name:26s} {e.mean():9.5f} {e[top].mean():15.4f} {e[ev].mean():11.4f} {e[~ev].mean():11.5f} {(p < -DROP_1KG).mean():9.4f}{extra}")
        fire = preds[f]["gbdt_full"] < -DROP_1KG
        print(f"           gbdt_full fires on {fire.mean():.4f} of rows: precision {ev[fire].mean():.3f} recall {fire[ev].mean():.3f}; true rate {ev.mean():.4f}")

    # the scored forcing: the position-free model's predicted change and rate, next to the full one
    print("\n== the competition forcing: predicted change at the old seed ages, vegetated rows ==")
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([fr.reshape(-1, 7) for fr in frames.values()]).min(axis=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sc in io.SCENARIOS:
        mt = mort.open_monthly(sc)
        xt = np.load(cache.CACHE_DIR / f"test_{sc}_annual.npy", mmap_mode="r")
        sx, _ = site_features(np.asarray(mt), np.asarray(xt[:, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64), thr)
        raw0 = M_ALL * (frames[sc] - floors)
        n = raw0.shape[0]
        state0 = (np.transpose(raw0[:, old, :4], (1, 0, 2)) / y_mean[:4]).reshape(8 * n, 4)
        x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
        veg = (np.transpose(raw0[:, old, 1]) > mort.BARE_AGB).reshape(-1)
        for vname in ("gbdt_full", "gbdt_position_free"):
            booster, cols = models[vname]
            p = np.zeros(len(veg))
            p[veg] = booster.predict(x[veg][:, cols], num_iteration=booster.best_iteration)
            pv = p[veg]
            print(f"  {sc} {vname:20s} rows {veg.sum():6d} mean change {pv.mean():+.4f}  P(loss > 1 kgC) {(pv < -DROP_1KG).mean():.4f}  P(loss > 3 kgC) {(pv < -3 * DROP_1KG).mean():.4f}  mean sq change {np.mean(pv ** 2):.4f}")
            np.save(OUT_DIR / f"pred_change_{vname}_{sc}.npy", p.reshape(8, n).astype(np.float32))
    for f in ("public", "private"):
        for vname in ("gbdt_full", "gbdt_position_free"):
            pv = preds[f][vname][vegf[f]]
            print(f"  holdout {f} {vname:20s} rows {vegf[f].sum():6d} mean change {pv.mean():+.4f}  P(loss > 1 kgC) {(pv < -DROP_1KG).mean():.4f}  P(loss > 3 kgC) {(pv < -3 * DROP_1KG).mean():.4f}  mean sq change {np.mean(pv ** 2):.4f}")
        t = truth[f][vegf[f]]
        print(f"  holdout {f} {'truth':20s} rows {vegf[f].sum():6d} mean change {t.mean():+.4f}  P(loss > 1 kgC) {(t < -DROP_1KG).mean():.4f}  P(loss > 3 kgC) {(t < -3 * DROP_1KG).mean():.4f}  mean sq change {np.mean(t ** 2):.4f}")
    (METRICS / "mort_gbdt.json").write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {METRICS / 'mort_gbdt.json'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

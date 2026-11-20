#!/usr/bin/env python
"""A stand-alone detector of the old-stand collapse, and what it adds to the latent model.

Given a site's forty years of forcing and its decoded initial state, predict the
scaled 40-year agb change at the old seed ages (>= 150). The point is not to beat
the latent rollout -- ``mort_residuals.py`` shows it already removes 97 % of the
event-row persistence loss -- but to answer three things it cannot:

  1. how much of the collapse is recoverable from *explicit* drought indices
     (``mort_trigger.py``) plus the stand's state, i.e. what the trigger is worth on
     its own, with an ablation that removes the drought sequence;
  2. whether an independent detector corrects the latent model's +0.06 residual on
     event rows (a cross-fitted blend over the two holdout folds);
  3. what event rate the same rule predicts on the scored forcing, where the
     latent model's own predictions cannot be checked against truth.

The model is a ridge regression on a GAM-style basis: every continuous feature is
quantile-binned into 16 one-hot indicators (so thresholds can be learned), plus the
standardised feature itself, plus products of the initial biomass with the drought
maxima. Fitted on the 47,981 fit sites of ``cgb.holdout.make_split(seed=42)``,
never on a holdout site; lambda chosen on the public fold and reported on both.

    ./.venv/bin/python scripts/mort_detector.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices, pet_priestley_taylor  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
METRICS = io.ROOT / "work" / "metrics"
OUT_DIR = io.ROOT / "work" / "models" / "mort_detector"
M_ALL = 1.14114
DROP_1KG = 1.0 / 3.395674394279066
SEQ_KEYS = ("wb12", "spi12", "vpd6", "tanom", "wb6", "vpd12")
BANDS = ((0, 3), (3, 10), (10, 20), (20, 30), (30, 37), (37, 40))
N_BINS = 16


# ---- features ---------------------------------------------------------------


def climatology(m: np.ndarray) -> np.ndarray:
    """``(n, 8)`` 40-year climate descriptors from the monthly cache."""
    n = m.shape[0]
    pr = m[..., 1].astype(np.float64)
    ta = m[..., 4].astype(np.float64)
    rs = m[..., 5].astype(np.float64)
    pet = pet_priestley_taylor(ta, rs)
    p_ann = pr.sum(axis=(1, 2)) / 40.0
    pet_ann = pet.sum(axis=(1, 2)) / 40.0
    p_month = pr.mean(axis=1)  # (n, 12)
    return np.stack(
        [
            ta.mean(axis=(1, 2)),
            p_ann,
            pet_ann,
            pet_ann / np.maximum(p_ann, 1.0),
            p_month.std(axis=1) / np.maximum(p_month.mean(axis=1), 1e-3),
            p_month.min(axis=1),
            m[..., 7].astype(np.float64).max(axis=(1, 2)),  # hottest daily-max month
            m[..., 8].astype(np.float64).min(axis=(1, 2)),  # driest air
        ],
        axis=1,
    )


def seq_features(ix: dict[str, np.ndarray], thr: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
    """``(n, k)`` summaries of each drought index over the 40 years."""
    cols, names = [], []
    for key in SEQ_KEYS:
        x = ix[key]
        fin = np.isfinite(x)
        fill = np.nanmin(np.where(fin, x, np.nan), axis=1, keepdims=True)
        x = np.where(fin, x, fill)
        q90, q99 = thr[key]
        cols.append(x.max(axis=1))
        names.append(f"{key}_max")
        for lo, hi in BANDS:
            cols.append(x[:, lo:hi].max(axis=1))
            names.append(f"{key}_max_y{lo + 1}-{hi}")
        cols.append((x > q90).sum(axis=1))
        names.append(f"{key}_n_q90")
        cols.append((x > q99).sum(axis=1))
        names.append(f"{key}_n_q99")
        cols.append(np.maximum(x - q90, 0).sum(axis=1))
        names.append(f"{key}_exc_q90")
        cols.append(np.argmax(x, axis=1) / 40.0)
        names.append(f"{key}_argmax")
        cols.append(x[:, -1])
        names.append(f"{key}_y40")
    return np.stack(cols, axis=1), names


def site_features(monthly: np.ndarray, static: np.ndarray, thr: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
    ix = build_indices(monthly)
    ix.pop("_pet_over_p")
    seq, seq_names = seq_features(ix, thr)
    clim = climatology(monthly)
    clim_names = ["clim_tair", "clim_p", "clim_pet", "clim_pet_over_p", "clim_p_season_cv", "clim_p_min_month", "clim_tmax", "clim_qmin"]
    static_names = [f"soil{k}" for k in range(static.shape[1])]
    return np.concatenate([seq, clim, static], axis=1), seq_names + clim_names + static_names


def row_design(site_x: np.ndarray, state: np.ndarray, age_idx: np.ndarray) -> np.ndarray:
    """Per-(site, age) raw feature matrix: site features, scaled pools at year 0, age one-hot, products."""
    agb0 = state[:, 1:2]
    prod = agb0 * site_x[:, :3]  # agb0 x (wb12_max, ... first three site columns are the wb12 block head)
    onehot = np.eye(8)[age_idx]
    return np.concatenate([site_x, state, prod, onehot], axis=1)


class GamRidge:
    """Ridge on quantile-binned one-hot indicators plus standardised raw features."""

    def __init__(self, n_bins: int = N_BINS, lam: float = 1.0) -> None:
        self.n_bins = n_bins
        self.lam = lam

    def _basis(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sd
        parts = [np.ones((x.shape[0], 1)), z]
        for j, edges in enumerate(self.edges):
            b = np.clip(np.searchsorted(edges, x[:, j], side="right"), 0, self.n_bins - 1)
            oh = np.zeros((x.shape[0], self.n_bins))
            oh[np.arange(x.shape[0]), b] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, lams: tuple[float, ...], x_val: np.ndarray, y_val: np.ndarray) -> "GamRidge":
        self.mu = x.mean(axis=0)
        self.sd = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(x[:, j], qs)) for j in range(x.shape[1])]
        step = 50_000
        p = 1 + x.shape[1] + sum(self.n_bins for _ in self.edges)
        xtx = np.zeros((p, p))
        xty = np.zeros(p)
        for lo in range(0, x.shape[0], step):
            b = self._basis(x[lo : lo + step])
            xtx += b.T @ b
            xty += b.T @ y[lo : lo + step]
        bv = self._basis(x_val)
        best = None
        for lam in lams:
            reg = lam * np.eye(p)
            reg[0, 0] = 0.0
            w = np.linalg.solve(xtx + reg, xty)
            mse = float(np.mean((bv @ w - y_val) ** 2))
            if best is None or mse < best[0]:
                best = (mse, lam, w)
        self.mse_val, self.lam, self.w = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._basis(x) @ self.w


# ---- main ---------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    div = cache.scoring_divisors()
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()
    xa = cache.open_x_annual()
    monthly = mort.open_monthly("train")

    # thresholds for the count features come from the fit sites only
    ix_fit = build_indices(np.asarray(monthly[split.fit[::4]]))
    thr = {k: (float(np.nanquantile(ix_fit[k], 0.9)), float(np.nanquantile(ix_fit[k], 0.99))) for k in SEQ_KEYS}
    del ix_fit

    def build(sites: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sx, names = site_features(np.asarray(monthly[sites]), np.asarray(xa[sites, 0], dtype=np.float64)[:, list(cache.STATIC_DRIVERS)], thr)
        st = np.asarray(traj[old][:, sites], dtype=np.float64)  # (8, n, 41, 7)
        state0 = st[:, :, 0, :4] / y_mean[:4]  # scaled pools
        y0 = st[:, :, 0, 1] / div[1]
        y40 = st[:, :, 40, 1] / div[1]
        veg = st[:, :, 0, 1] > mort.BARE_AGB
        n = len(sites)
        rows_x = row_design(np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0.reshape(8 * n, 4), np.repeat(np.arange(8), n))
        return rows_x, (y40 - y0).reshape(-1), y0.reshape(-1), veg.reshape(-1), names

    x_fit, dy_fit, y0_fit, veg_fit, names = build(split.fit)
    x_pub, dy_pub, y0_pub, veg_pub, _ = build(split.public)
    x_pri, dy_pri, y0_pri, veg_pri, _ = build(split.private)
    print(f"features: {x_fit.shape[1]} raw columns per row; fit rows {veg_fit.sum():,} vegetated of {len(dy_fit):,}  ({time.time() - t0:.0f}s)")

    lams = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    model = GamRidge().fit(x_fit[veg_fit], dy_fit[veg_fit], lams, x_pub[veg_pub], dy_pub[veg_pub])
    print(f"ridge lambda {model.lam} chosen on the public fold ({time.time() - t0:.0f}s)")
    # ablation: no drought sequence (zero the sequence block so the basis is unchanged in shape)
    n_seq = len(SEQ_KEYS) * (1 + len(BANDS) + 5)
    x_fit_nos = x_fit.copy()
    x_fit_nos[:, :n_seq] = 0.0
    x_fit_nos[:, -(8 + 3) : -8] = 0.0  # the products
    x_pub_nos = x_pub.copy(); x_pub_nos[:, :n_seq] = 0.0; x_pub_nos[:, -(8 + 3) : -8] = 0.0
    x_pri_nos = x_pri.copy(); x_pri_nos[:, :n_seq] = 0.0; x_pri_nos[:, -(8 + 3) : -8] = 0.0
    model_nos = GamRidge().fit(x_fit_nos[veg_fit], dy_fit[veg_fit], lams, x_pub_nos[veg_pub], dy_pub[veg_pub])

    def predict(mdl: GamRidge, x: np.ndarray, veg: np.ndarray) -> np.ndarray:
        p = np.zeros(len(veg))
        p[veg] = mdl.predict(x[veg])
        return p

    preds = {
        "public": {"detector": predict(model, x_pub, veg_pub), "no_drought_seq": predict(model_nos, x_pub_nos, veg_pub)},
        "private": {"detector": predict(model, x_pri, veg_pri), "no_drought_seq": predict(model_nos, x_pri_nos, veg_pri)},
    }
    truth = {"public": dy_pub, "private": dy_pri}
    y0s = {"public": y0_pub, "private": y0_pri}
    # the latent model's own predicted change on the same rows
    for fold, sites in (("public", split.public), ("private", split.private)):
        p = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")[:, :, -1, 1].astype(np.float64)
        preds[fold]["latent"] = (p[old] - (np.asarray(traj[old][:, sites], dtype=np.float64)[:, :, 0, 1] / div[1])).reshape(-1)
        preds[fold]["persistence"] = np.zeros_like(preds[fold]["latent"])

    # cross-fitted blend: weights on (latent, detector) fitted on one fold, scored on the other
    for fit_fold, eval_fold in (("public", "private"), ("private", "public")):
        a = np.stack([preds[fit_fold]["latent"], preds[fit_fold]["detector"]], axis=1)
        w, *_ = np.linalg.lstsq(a, truth[fit_fold], rcond=None)
        preds[eval_fold]["blend"] = w[0] * preds[eval_fold]["latent"] + w[1] * preds[eval_fold]["detector"]
        preds[eval_fold]["blend_w"] = w
    # and a simpler correction: latent + c * (detector - latent) restricted to rows where the detector predicts a collapse
    payload: dict[str, object] = {"lambda": model.lam, "lambda_no_seq": model_nos.lam, "thr": thr}
    print("\n== old seed ages (>= 150), agb column, scaled squared error of the year-40 prediction ==")
    print(f"  {'fold':8s} {'model':16s} {'all rows':>9s} {'top-10% persist rows':>20s} {'event rows':>11s} {'other rows':>11s} {'pred rate':>9s} {'true rate':>9s}")
    for fold in ("public", "private"):
        t = truth[fold]
        top = (t ** 2) > np.quantile(t ** 2, 0.9)
        ev = t < -DROP_1KG
        payload[fold] = {}
        for name in ("persistence", "detector", "no_drought_seq", "latent", "blend"):
            p = preds[fold][name]
            e = (p - t) ** 2
            rate = float((p < -DROP_1KG).mean())
            payload[fold][name] = {"all": float(e.mean()), "top10": float(e[top].mean()), "event": float(e[ev].mean()), "other": float(e[~ev].mean()), "pred_rate": rate}
            extra = f"   blend w(latent, detector) = {np.round(preds[fold]['blend_w'], 3)}" if name == "blend" else ""
            print(f"  {fold:8s} {name:16s} {e.mean():9.5f} {e[top].mean():20.4f} {e[ev].mean():11.4f} {e[~ev].mean():11.5f} {rate:9.4f} {ev.mean():9.4f}{extra}")
        # detector as a classifier of event rows at the 1 kgC operating point
        p = preds[fold]["detector"]
        fire = p < -DROP_1KG
        prec = float(ev[fire].mean()) if fire.any() else float("nan")
        rec = float(fire[ev].mean())
        pl = preds[fold]["latent"] < -DROP_1KG
        print(f"           detector fires on {fire.mean():.4f} of rows: precision {prec:.3f} recall {rec:.3f}   | latent fires on {pl.mean():.4f}: precision {ev[pl].mean():.3f} recall {pl[ev].mean():.3f}")
        payload[fold]["detector_precision"] = prec
        payload[fold]["detector_recall"] = rec

    # ---- the scored forcing ---------------------------------------------------
    print("\n== the same detector on the competition forcing: predicted change at the old seed ages ==")
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sc in io.SCENARIOS:
        mt = mort.open_monthly(sc)
        xt = np.load(cache.CACHE_DIR / f"test_{sc}_annual.npy", mmap_mode="r")
        sx, _ = site_features(np.asarray(mt), np.asarray(xt[:, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64), thr)
        raw0 = M_ALL * (frames[sc] - floors)  # (6171, 15, 7) physical
        state0 = np.transpose(raw0[:, old, :4], (1, 0, 2)) / y_mean[:4]  # (8, n, 4)
        veg = np.transpose(raw0[:, old, 1]) > mort.BARE_AGB
        n = raw0.shape[0]
        rows = row_design(np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0.reshape(8 * n, 4), np.repeat(np.arange(8), n))
        p = predict(model, rows, veg.reshape(-1)).reshape(8, n)
        y0 = np.transpose(raw0[:, old, 1]) / div[1]
        pv = p[veg]
        print(f"  {sc}: vegetated old rows {veg.sum():6d}  mean change {pv.mean():+.4f} ({(pv / y0[veg]).mean():+.3f} rel)  "
              f"P(loss > 1 kgC) {(pv < -DROP_1KG).mean():.4f}  P(loss > 3 kgC) {(pv < -3 * DROP_1KG).mean():.4f}  mean sq change {np.mean(pv ** 2):.4f}")
        np.save(OUT_DIR / f"pred_change_{sc}.npy", p.astype(np.float32))
    for fold in ("public", "private"):
        pv = preds[fold]["detector"][truth[fold] != 0]
        t = truth[fold][truth[fold] != 0]
        print(f"  holdout {fold} (reference): detector mean change {pv.mean():+.4f}  P(loss > 1 kgC) {(pv < -DROP_1KG).mean():.4f}  mean sq change {np.mean(pv ** 2):.4f}   "
              f"| truth mean {t.mean():+.4f}  P {(t < -DROP_1KG).mean():.4f}  mean sq {np.mean(t ** 2):.4f}")
    (METRICS / "mort_detector.json").write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {METRICS / 'mort_detector.json'} and {OUT_DIR}/pred_change_*.npy  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

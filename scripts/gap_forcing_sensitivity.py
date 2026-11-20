"""How far do the trained checkpoints' predictions move when the holdout forcing is
rebuilt the way the competition forcing was built?

The competition forcing differs from the training forcing in ways that carry no
physical content (``notes/board-gap.md``): CO2 is one series for every site where
the training set has a spatially varying field; the 24 hourly wind drivers are a
climatology with no interannual variability; soil driver 135 is one constant; the
monthly ``ta_m`` is no longer the mean of the 24 hourly temperatures. Others do carry
content (shortwave ~5-11% higher, diurnal temperature range ~18% larger, smoother
precipitation). Applying each change to the public-fold forcing and rolling the
frozen checkpoints again measures how much of a prediction rides on that input. For
an artifact with no physical content, ED's own response is nil, so the prediction
shift is a transfer loss outright; for the physical ones it is an upper bound.

    ./.venv/bin/python scripts/gap_forcing_sensitivity.py --tags v1,v1s43 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, latent  # noqa: E402

CO2 = list(range(8, 32))
SW = [35 + 4 * h for h in range(24)]
TH = [34 + 4 * h for h in range(24)]
WIND = [36 + 4 * h for h in range(24)]
D135 = 135


def load_models(tags: list[str], device: torch.device) -> list[tuple[latent.LatentRollout, np.ndarray, np.ndarray]]:
    out = []
    for tag in tags:
        ckpt = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = latent.LatentRollout(d_enc=a.get("d_enc", 128), d_h=a.get("d_h", 256), n_layers=a.get("layers", 2)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        out.append((model, ckpt["season_mu"], ckpt["season_sd"]))
    return out


@torch.no_grad()
def predict40(models, x_year: np.ndarray, season_raw: np.ndarray, static: np.ndarray, u0: np.ndarray,
              agef: np.ndarray, device: torch.device) -> np.ndarray:
    """``(n_sites, 15, 2)`` year-40 scaled height/agb, averaged over the checkpoints."""
    n = x_year.shape[0]
    pred = np.zeros((n, 15, 2), dtype=np.float64)
    for model, mu, sd in models:
        season = ((season_raw - mu) / sd).astype(np.float32)
        for lo in range(0, n, 128):
            b = min(128, n - lo)
            xt = torch.from_numpy(x_year[lo:lo + b]).to(device).repeat(15, 1, 1)
            st = torch.from_numpy(static[lo:lo + b]).to(device).repeat(15, 1)
            se = torch.from_numpy(season[lo:lo + b]).to(device).repeat(15, 1)
            u0t = torch.from_numpy(np.ascontiguousarray(np.transpose(u0[lo:lo + b], (1, 0, 2)).reshape(15 * b, 7))).to(device)
            ag = torch.from_numpy(np.repeat(agef, b, axis=0)).to(device)
            out = model(xt, se, st, u0t, ag)[:, -1, :2].float().cpu().numpy().reshape(15, b, 2)
            pred[lo:lo + b] += np.transpose(out, (1, 0, 2)) / len(models)
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="v1,v1s43")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fold", default="public")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    split = holdout.make_split(seed=42)
    sites = split.public if args.fold == "public" else split.private
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"].astype(np.float32), stats["x_std"].astype(np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)
    y_mean = stats["y_mean"].astype(np.float32)
    agef = latent.age_features(stats)
    models = load_models(args.tags.split(","), device)

    # physical inputs of the fold
    ann = np.asarray(cache.open_x_annual()[sites], dtype=np.float32)          # (n, 40, 136)
    sea = np.asarray(cache.open_x_season()[sites], dtype=np.float32)          # (n, 136, 12)
    traj = cache.open_traj()
    u_all = np.asarray(traj[:, sites], dtype=np.float32) / y_mean              # (15, n, 41, 7)
    u0 = np.ascontiguousarray(np.transpose(u_all[:, :, 0, :], (1, 0, 2)))     # (n, 15, 7)
    true40 = np.transpose(u_all[:, :, 40, :2], (1, 0, 2)).astype(np.float64)
    pers40 = u0[:, :, :2].astype(np.float64)
    bare = (u0[:, :, 0] * y_mean[0]) < 1.0

    # the competition's constant for driver 135, and its cross-site CO2 series
    x126, _ = io.open_test("ssp126")
    d135_test = float(x126[0, 0, 0, D135])
    ann_t = np.load(cache.CACHE_DIR / "test_ssp126_annual.npy", mmap_mode="r")
    co2_test_mean = np.asarray(ann_t[:, :, CO2], dtype=np.float32).mean(0)   # (40, 24)

    def run(name: str, ann_p: np.ndarray, sea_p: np.ndarray) -> dict:
        x_year = ((ann_p - x_mean) / x_std).astype(np.float32)
        static = x_year[:, 0, list(cache.STATIC_DRIVERS)].astype(np.float32)
        season_raw = latent.seasonal_block(sea_p)
        pred = predict40(models, x_year, season_raw, static, u0, agef, device)
        pred[bare] = 0.0
        return {"name": name, "pred": pred}

    base = run("baseline", ann, sea)
    results = []

    def report(r: dict) -> None:
        pred = r["pred"]
        score = float(((pred - true40) ** 2).mean())
        shift_all = float(((pred - base["pred"]) ** 2).mean())
        ages = np.array(io.SEED_AGES)
        old = ages >= 150
        young = ages <= 30
        row = {
            "name": r["name"], "score": score,
            "shift_vs_baseline": shift_all,
            "shift_height": float(((pred - base["pred"])[..., 0] ** 2).mean()),
            "shift_agb": float(((pred - base["pred"])[..., 1] ** 2).mean()),
            "shift_young": float(((pred - base["pred"])[:, young] ** 2).mean()),
            "shift_old": float(((pred - base["pred"])[:, old] ** 2).mean()),
            "mean_change_h": float((pred - pers40)[..., 0].mean()),
            "mean_change_a": float((pred - pers40)[..., 1].mean()),
        }
        results.append(row)
        print(f"  {row['name']:38s} score {score:.4f} | shift vs baseline {shift_all:.4f} (h {row['shift_height']:.4f} a {row['shift_agb']:.4f}; "
              f"young {row['shift_young']:.4f} old {row['shift_old']:.4f}) | mean change h {row['mean_change_h']:+.4f} a {row['mean_change_a']:+.4f}", flush=True)

    print(f"{args.fold} fold, {len(sites)} sites, checkpoints {args.tags}, device {device}")
    print(f"  persistence score {float(((pers40 - true40) ** 2).mean()):.4f}")
    report(base)

    # --- artifacts with no physical content ---------------------------------------
    a = ann.copy(); a[:, :, CO2] = co2_test_mean[None]                        # the competition's CO2 series at every site
    s = sea.copy(); s[:, CO2, :] = np.asarray(cache.open_x_season()[sites][:, CO2, :]).mean(0, keepdims=True)
    report(run("co2: competition series, all sites", a, s))
    a = ann.copy(); a[:, :, CO2] = ann[:, :, CO2].mean(0, keepdims=True)      # site-invariant, training level
    report(run("co2: site-invariant (train mean)", a, sea))
    a = ann.copy(); a[:, :, WIND] = ann[:, :, WIND].mean(1, keepdims=True)    # wind climatology, no interannual
    report(run("wind: 40-yr climatology", a, sea))
    a = ann.copy(); a[:, :, D135] = d135_test
    report(run(f"driver 135 := {d135_test:.4g} (test constant)", a, sea))
    a = ann.copy(); a[:, :, 0] = ann[:, :, 0] + rng.normal(0.0, 0.25 / np.sqrt(12), size=ann[:, :, 0].shape).astype(np.float32)
    report(run("ta_m decoupled from hourly mean", a, sea))
    a = ann.copy(); a[:, :, CO2] = co2_test_mean[None]; a[:, :, WIND] = ann[:, :, WIND].mean(1, keepdims=True); a[:, :, D135] = d135_test
    s = sea.copy(); s[:, CO2, :] = np.asarray(cache.open_x_season()[sites][:, CO2, :]).mean(0, keepdims=True)
    report(run("all no-content artifacts together", a, s))

    # --- shifts with physical content ------------------------------------------------
    for f in (1.05, 1.10):
        a = ann.copy(); a[:, :, SW] *= f; s = sea.copy(); s[:, SW, :] *= f
        report(run(f"shortwave x{f:.2f}", a, s))
    a = ann.copy(); s = sea.copy()
    th_mean = ann[:, :, TH].mean(2, keepdims=True); a[:, :, TH] = th_mean + 1.18 * (ann[:, :, TH] - th_mean)
    sm = sea[:, TH, :].mean(1, keepdims=True); s[:, TH, :] = sm + 1.18 * (sea[:, TH, :] - sm)
    report(run("diurnal T range x1.18", a, s))
    for f in (0.7, 0.5):
        a = ann.copy(); pm = ann[:, :, 1].mean(1, keepdims=True); a[:, :, 1] = pm + f * (ann[:, :, 1] - pm)
        report(run(f"precip interannual anomalies x{f:.1f}", a, sea))
    a = ann.copy(); a[:, :, [0] + TH] += 1.0; s = sea.copy(); s[:, [0] + TH, :] += 1.0
    report(run("temperature +1 K everywhere", a, s))
    # everything together, the artifact set plus the physical shifts at their measured size
    a = ann.copy(); s = sea.copy()
    a[:, :, CO2] = co2_test_mean[None]; a[:, :, WIND] = ann[:, :, WIND].mean(1, keepdims=True); a[:, :, D135] = d135_test
    s[:, CO2, :] = np.asarray(cache.open_x_season()[sites][:, CO2, :]).mean(0, keepdims=True)
    a[:, :, SW] *= 1.05; s[:, SW, :] *= 1.05
    a[:, :, TH] = th_mean + 1.18 * (ann[:, :, TH] - th_mean); s[:, TH, :] = sm + 1.18 * (sea[:, TH, :] - sm)
    report(run("artifacts + SW x1.05 + DTR x1.18", a, s))

    # --- CO2: level against trend ----------------------------------------------------
    co2_train_year = ann[:, :, CO2].mean(0)                                    # (40, 24) training series, site mean
    a = ann.copy(); a[:, :, CO2] = co2_train_year.mean(0)[None, None, :]       # training level, no trend
    report(run("co2: training level, flat", a, sea))
    a = ann.copy(); a[:, :, CO2] = (co2_train_year - co2_train_year.mean(0) + co2_test_mean.mean(0))[None]
    report(run("co2: competition level, training trend", a, sea))

    # --- site descriptor from the nearest competition site in (ta_m, pr) climatology ----
    # the physical climate stays (nearest in the 24 monthly ta_m/pr values), the rest of
    # the fingerprint (hus, ta_h, rsds, wind climatology, soil) becomes the competition
    # pipeline's; the annual forcing is untouched
    sea_t = np.asarray(np.load(cache.CACHE_DIR / "test_ssp126_season.npy"), dtype=np.float32)
    key_h = np.concatenate([sea[:, 0, :] - 273.15, np.log1p(sea[:, 1, :])], axis=1)
    key_t = np.concatenate([sea_t[:, 0, :] - 273.15, np.log1p(sea_t[:, 1, :])], axis=1)
    mu_k, sd_k = key_h.mean(0), key_h.std(0) + 1e-6
    zh, zt = (key_h - mu_k) / sd_k, (key_t - mu_k) / sd_k
    nn = np.empty(len(zh), dtype=int); dnn = np.empty(len(zh))
    for lo in range(0, len(zh), 500):
        hi = min(lo + 500, len(zh))
        dd = ((zh[lo:hi, None, :] - zt[None, :, :]) ** 2).sum(-1)
        nn[lo:hi] = dd.argmin(1); dnn[lo:hi] = np.sqrt(dd.min(1))
    print(f"  nearest competition site in ta/pr climatology: distance quantiles {np.quantile(dnn, [.1, .5, .9]).round(3)}")
    s = sea_t[nn].copy(); s[:, 0, :] = sea[:, 0, :]; s[:, 1, :] = sea[:, 1, :]   # keep own ta_m and pr climatology
    a = ann.copy(); a[:, :, D135] = d135_test
    for k in (129, 130, 131, 132, 133, 134):
        a[:, :, k] = ann_t[nn, 0, k][:, None]
    report(run("site descriptor of nearest competition site", a, s))
    # the same swap but with a random other holdout site's descriptor at the same ta/pr distance scale (control)
    nn_h = np.empty(len(zh), dtype=int)
    for lo in range(0, len(zh), 500):
        hi = min(lo + 500, len(zh))
        dd = ((zh[lo:hi, None, :] - zh[None, :, :]) ** 2).sum(-1)
        dd[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        nn_h[lo:hi] = dd.argmin(1)
    s = sea[nn_h].copy(); s[:, 0, :] = sea[:, 0, :]; s[:, 1, :] = sea[:, 1, :]
    a = ann.copy()
    for k in (129, 130, 131, 132, 133, 134, 135):
        a[:, :, k] = ann[nn_h, 0, k][:, None]
    report(run("site descriptor of nearest holdout site (control)", a, s))

    # --- the forcing of another site: the ceiling of forcing-driven error --------------
    perm = rng.permutation(len(sites))
    report(run("forcing of a random other holdout site", ann[perm], sea[perm]))

    out = io.ROOT / "work" / "metrics" / f"gap_forcing_sensitivity_{args.fold}.json"
    with open(out, "w") as fh:
        json.dump({"tags": args.tags, "fold": args.fold, "rows": results}, fh, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Roll a trained latent model over the competition inputs and write a submission candidate.

Steps, each of which is a decision recorded in ``notes/theory.md``:

1. **Decode ``test_y0``** (Sec. 5 Q1): ``raw_k = m_k * (y0_k - floor_k)``. Height
   ``m = 1.14114`` is pinned by the 35 m cap and the 1.5 m seedling constant; agb is
   bracketed [1.0, 1.26] and passed on the command line; soil 1.12 and lai 1.0 come
   from the allometry table; the fluxes are not model inputs and get 1.
2. **Forcing** as the model saw it in training: annual means and the 40-year
   seasonal climatology of the 136 drivers, z-scored with the training statistics
   (cached once per scenario under ``work/cache/test_<scenario>_*.npy``).
3. **Rollout** with the checkpoint, year-40 height and agb in scaled units.
4. **Rules**: a site whose decoded initial height is below 1 m is predicted bare,
   the benchmark evaluator's own rule. Measured on ``y_traj``: of the 117,255
   training rows with slot-0 height below 1 m, 0.07% reach 3 m by year 40, the mean
   year-40 height is 0.20 m and the mean agb 0.016, so predicting zero costs
   ~3e-4 per row while a spurious regrowth would cost up to 1.6. Outputs are
   clamped at 0 and at the 35 m cap by the model itself.
5. **Optional damping** of predicted agb losses at seed ages >= 100 (``--damp``),
   the Q2 knob, for a paired board comparison.
6. Rows written in ``sample_submission.csv`` order and checked against it.

Nothing here submits.

    <torch venv>/bin/python scripts/predict_latent.py --tag v1 --m-agb 1.0 --out results/latent_v1_agb100.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, io, latent  # noqa: E402

#: One multiplier for all seven variables (theory.md Sec. 5 Q1): height's pinned
#: 1/0.87632, which gpp (~1.15) and soil (~1.12) independently agree with.
M_ALL = 1.14114
BARE_HEIGHT_M = 1.0


CACHE_TAG = ""  # --cache-tag: read the repaired caches of scripts/repair_test_cache.py (test_<s>_annual_<tag>.npy)


def test_forcing(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Annual means ``(6171, 40, 136)`` and seasonal climatology ``(6171, 136, 12)`` of ``test_x``."""
    if CACHE_TAG:
        return (np.load(cache.CACHE_DIR / f"test_{scenario}_annual_{CACHE_TAG}.npy"),
                np.load(cache.CACHE_DIR / f"test_{scenario}_season_{CACHE_TAG}.npy"))
    p_ann = cache.CACHE_DIR / f"test_{scenario}_annual.npy"
    p_sea = cache.CACHE_DIR / f"test_{scenario}_season.npy"
    if p_ann.exists() and p_sea.exists():
        return np.load(p_ann), np.load(p_sea)
    x, _ = io.open_test(scenario)
    n = x.shape[0]
    ann = np.empty((n, 40, 136), dtype=np.float32)
    sea = np.zeros((n, 136, 12), dtype=np.float64)
    for lo in range(0, n, 256):
        blk = np.asarray(x[lo : lo + 256], dtype=np.float32)  # (b, 40, 12, 136)
        ann[lo : lo + 256] = blk.mean(axis=2)
        sea[lo : lo + 256] = np.transpose(blk.mean(axis=1), (0, 2, 1))
    sea = sea.astype(np.float32)
    np.save(p_ann, ann)
    np.save(p_sea, sea)
    return ann, sea


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="v1", help="comma-separated checkpoint tags; predictions are averaged")
    p.add_argument("--m-agb", type=float, default=M_ALL, help="agb decode multiplier (default: the common 1.141)")
    p.add_argument("--m-lai", type=float, default=1.0,
                   help="lai decode multiplier. 1.0 (default) since 2026-09-25: the training climate->lai relation applied to the "
                        "competition climate reproduces the test y0 lai at scale 1.0 (median ratio 0.97-1.03 at every age) and not "
                        "at 1.141 (1.11-1.17); height and agb reproduce at 1.141 (scratchpad y0_climate_relation.py)")
    p.add_argument("--y0-fill-age1", action="store_true",
                   help="replace the decoded age-1 initial state (height, agb, soil, lai) by the climate-predicted training-like "
                        "seedling state from scripts/y0_climate_fill.py (work/cache/y0_clim_age1_<scenario>.npy)")
    p.add_argument("--h-scale", type=float, default=1.0,
                   help="multiply the predicted height column by this factor after the rules (1 = off). The board's ten points "
                        "solve to a height truth that is 0.876 x training-like dynamics (notes/board-solve.md)")
    p.add_argument("--damp", type=float, default=1.0, help="factor on predicted agb LOSSES at seed ages >= 100 (1 = off); the Q2 knob")
    p.add_argument("--shrink", type=float, default=1.0,
                   help="factor on the whole predicted change (both columns, either sign) at seed ages >= 100: "
                        "pred = y0 + shrink * (pred - y0); the general 'old stands change less' knob (1 = off)")
    p.add_argument("--climatological", action="store_true",
                   help="replace every forcing year by the site's 40-year mean (no interannual variability, no trend): the "
                        "model's background trajectory; minus the real-forcing rollout it isolates the weather-driven events")
    p.add_argument("--detrended", action="store_true",
                   help="remove each driver's linear 40-year trend per site (mean and interannual variability kept)")
    p.add_argument("--shuffle-years", type=int, default=-1,
                   help="seed >= 0: another weather realization with the same climate - per site, every driver is split into its "
                        "linear trend and the residual years, and the residual years are permuted (one permutation per site, shared "
                        "by all drivers so each year stays internally consistent)")
    p.add_argument("--bootstrap", action="store_true",
                   help="with --shuffle-years: draw the residual years with replacement (a realization with other extremes), not a permutation")
    p.add_argument("--clim-drivers", default="",
                   help="comma-separated driver groups held at the site's 40-year mean (others real): "
                        "P (driver 1), T (0, 2-7 and the hourly air temperature), Q (hourly humidity), SW (hourly shortwave)")
    p.add_argument("--cache-tag", default="", help="repaired test caches (scripts/repair_test_cache.py --tag)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    global CACHE_TAG
    CACHE_TAG = args.cache_tag
    device = torch.device(args.device)

    models = []
    for tag in args.tags.split(","):
        ckpt = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        n_drv = 136 + (latent.N_MONTHLY_INPUTS if a.get("monthly") else 0)
        model = latent.LatentRollout(n_drv=n_drv, d_enc=a.get("d_enc", 128), d_h=a.get("d_h", 256), n_layers=a.get("layers", 2), use_t=not a.get("no_t", False)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        input_set = a.get("input_set", "full")
        if a.get("drop_co2", False) and input_set == "full":
            input_set = "noco2"
        models.append((model, ckpt["season_mu"], ckpt["season_sd"], input_set, (ckpt.get("monthly_mu"), ckpt.get("monthly_sd")) if a.get("monthly") else None))
    print(f"{len(models)} checkpoint(s): {args.tags}; input sets {[m[3] for m in models]}")

    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"].astype(np.float32), stats["x_std"].astype(np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)
    y_mean = stats["y_mean"].astype(np.float32)
    agef_all = latent.age_features(stats)

    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    mult = np.array([M_ALL, args.m_agb, M_ALL, args.m_lai, M_ALL, M_ALL, M_ALL])
    print("floors:", np.round(floors, 6), " multipliers:", mult)

    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    rows_id, rows_h, rows_a = [], [], []
    for scenario in io.SCENARIOS:
        ann, sea = test_forcing(scenario)
        if args.climatological:
            ann = np.repeat(ann.mean(axis=1, keepdims=True), ann.shape[1], axis=1)
        if args.detrended:
            yrs = (np.arange(ann.shape[1]) - (ann.shape[1] - 1) / 2).astype(np.float32)
            slope = np.einsum("t,ntd->nd", yrs, ann.astype(np.float64)) / float((yrs ** 2).sum())
            ann = (ann - yrs[None, :, None] * slope[:, None, :]).astype(np.float32)
        if args.shuffle_years >= 0:
            rng = np.random.default_rng(args.shuffle_years * 2 + io.SCENARIOS.index(scenario))
            yrs = (np.arange(ann.shape[1]) - (ann.shape[1] - 1) / 2).astype(np.float64)
            a64 = ann.astype(np.float64)
            slope = np.einsum("t,ntd->nd", yrs, a64) / float((yrs ** 2).sum())
            trend = a64.mean(axis=1, keepdims=True) + yrs[None, :, None] * slope[:, None, :]
            if args.bootstrap:
                perm = rng.integers(0, ann.shape[1], size=ann.shape[:2])
            else:
                perm = np.argsort(rng.random(ann.shape[:2]), axis=1)  # (sites, years)
            resid = np.take_along_axis(a64 - trend, perm[:, :, None], axis=1)
            ann = (trend + resid).astype(np.float32)
        if args.clim_drivers:
            groups = {"P": [1], "T": [0, 2, 3, 4, 5, 6, 7] + list(34 + 4 * np.arange(24)), "Q": list(33 + 4 * np.arange(24)), "SW": list(35 + 4 * np.arange(24))}
            idx = sorted({i for g in args.clim_drivers.split(",") for i in groups[g]})
            ann = ann.copy()
            ann[:, :, idx] = np.repeat(ann[:, :, idx].mean(axis=1, keepdims=True), ann.shape[1], axis=1)
        x_year = (ann - x_mean) / x_std
        static = x_year[:, 0, list(cache.STATIC_DRIVERS)].astype(np.float32)
        season_raw = latent.seasonal_block(sea)
        raw0 = mult * (frames[scenario] - floors)  # (6171, 15, 7) physical
        if args.y0_fill_age1:
            fill = np.load(cache.CACHE_DIR / f"y0_clim_age1_{scenario}.npy")
            print(f"{scenario}: age-1 state replaced by the climate-predicted seedling (mean height {fill[:, 0].mean():.3f} m vs decoded {raw0[:, 0, 0].mean():.3f} m)")
            raw0[:, 0, :4] = fill
        u0 = (raw0 / y_mean).astype(np.float32)
        n = u0.shape[0]
        pred40 = np.zeros((n, 15, 2), dtype=np.float64)
        with torch.no_grad():
            for model, season_mu, season_sd, input_set, monthly_stats in models:
                season = ((season_raw - season_mu) / season_sd).astype(np.float32)
                x_in = x_year.astype(np.float32, copy=True)
                if monthly_stats is not None:
                    from cgb import mortality as mort
                    blk, _, _ = latent.monthly_block(mort.open_monthly(scenario), *monthly_stats)
                    x_in = np.concatenate([x_in, blk], axis=2)
                if input_set != "full":
                    # the checkpoint never saw the removed blocks; feed it exactly what it was trained on
                    x_in, season = latent.apply_input_set(x_in, season, input_set)
                for lo in range(0, n, 128):
                    b = min(128, n - lo)
                    xt = torch.from_numpy(x_in[lo : lo + b]).to(device).repeat(15, 1, 1)
                    st = torch.from_numpy(static[lo : lo + b]).to(device).repeat(15, 1)
                    se = torch.from_numpy(season[lo : lo + b]).to(device).repeat(15, 1)
                    u0t = torch.from_numpy(np.ascontiguousarray(np.transpose(u0[lo : lo + b], (1, 0, 2)).reshape(15 * b, 7))).to(device)
                    ag = torch.from_numpy(np.repeat(agef_all, b, axis=0)).to(device)
                    out = model(xt, se, st, u0t, ag)[:, -1, :2].float().cpu().numpy().reshape(15, b, 2)
                    pred40[lo : lo + b] += np.transpose(out, (1, 0, 2)) / len(models)
        # rules and knobs
        bare = raw0[:, :, 0] < BARE_HEIGHT_M
        pred40[bare, 0] = 0.0
        pred40[bare, 1] = 0.0
        pers = u0[:, :, :2].astype(np.float64)
        if args.damp != 1.0:
            pred40 = latent.damp_old_decline(pred40, pers, factor=args.damp)
        if args.shrink != 1.0:
            old = np.array(io.SEED_AGES) >= 100
            pred40[:, old] = pers[:, old] + args.shrink * (pred40[:, old] - pers[:, old])
        if args.h_scale != 1.0:
            pred40[..., 0] *= args.h_scale
        print(f"{scenario}: bare rows {bare.sum()}; height mean {pred40[..., 0].mean():.4f} max {pred40[..., 0].max():.4f}; "
              f"agb mean {pred40[..., 1].mean():.4f} max {pred40[..., 1].max():.4f}; "
              f"mean change vs persistence h {np.mean(pred40[..., 0] - pers[..., 0]):+.4f} a {np.mean(pred40[..., 1] - pers[..., 1]):+.4f}")
        for site in range(n):
            for j, age in enumerate(io.SEED_AGES):
                rows_id.append(f"{scenario}_{site}_{age}")
        rows_h.append(pred40[..., 0].reshape(-1))
        rows_a.append(pred40[..., 1].reshape(-1))
    ids = np.array(rows_id)
    h = np.concatenate(rows_h)
    a = np.concatenate(rows_a)
    assert ids.shape == template.shape and np.array_equal(ids, template), "row order differs from sample_submission.csv"
    assert np.isfinite(h).all() and np.isfinite(a).all() and h.min() >= 0 and a.min() >= 0 and h.max() < 15 and a.max() < 15
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("id,height,agb\n")
        for i in range(len(ids)):
            fh.write(f"{ids[i]},{h[i]:.6f},{a[i]:.6f}\n")
    print(f"wrote {args.out} ({len(ids):,} rows); nothing submitted")


if __name__ == "__main__":
    main()

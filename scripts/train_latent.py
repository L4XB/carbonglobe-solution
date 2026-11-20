"""Train the latent-state rollout on the fit sites and score it at year 40 on the holdout.

Everything reported is in the competition's units and on ``cgb.holdout.make_split
(seed=42)``, so the numbers sit next to ``scripts/holdout_baseline.py`` (persistence
0.3115 public / 0.3029 private) and the board anchors. Per the lead's calibration in
``notes/board-anchors.md`` a persistence-like holdout score overstates the board by
about 1.24x, so both readings are printed.

    <torch venv>/bin/python scripts/train_latent.py --epochs 8 --tag v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import holdout, io, latent  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-sites", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--w-last", type=float, default=10.0)
    p.add_argument("--tag", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-h", type=int, default=256, help="GRU hidden size")
    p.add_argument("--d-enc", type=int, default=128, help="forcing-year encoder width")
    p.add_argument("--layers", type=int, default=2, help="GRU layers")
    p.add_argument("--fit-limit", type=int, default=0, help="use only this many fit sites (smoke test)")
    p.add_argument("--shuffle-forcing", action="store_true",
                   help="negative control: permute the forcing (annual drivers, seasonal block, static drivers) "
                        "across ALL sites with a fixed permutation, so no site's forcing belongs to it; "
                        "y0 and age untouched. The score should collapse to what y0 and age alone explain.")
    p.add_argument("--drop-co2", action="store_true",
                   help="zero the 24 CO2 drivers (8..31) in the annual inputs. In the training forcing CO2 is a "
                        "near-deterministic function of the year (358 -> 420 ppm at every site), so the model cannot "
                        "separate a CO2 response from a time-in-run effect; the competition series is flat at ~367 "
                        "(ssp126) and the frozen model's predictions move by MSE 0.034 when fed it "
                        "(scripts/gap_forcing_sensitivity.py, notes/board-gap.md)")
    p.add_argument("--split", choices=("random", "blocks"), default="random",
                   help="'random': cgb.holdout.make_split (scattered sites, neighbours in the fit set); "
                        "'blocks': cgb.holdout.make_block_split (whole 3-degree tiles held out, the competition's "
                        "new-location situation)")
    p.add_argument("--tile-cells", type=int, default=6, help="tile edge in 0.5-degree cells for --split blocks")
    p.add_argument("--input-set", choices=latent.INPUT_SETS, default="full",
                   help="full | noco2 (CO2 zeroed) | core (noco2 + no hourly drivers anywhere) | clim (noco2 + annual "
                        "drivers replaced by their 40-year site mean); see cgb.latent.apply_input_set")
    p.add_argument("--start-slots", type=int, default=0,
                   help="random start slot per batch, uniform in [0, K]: the window runs from year slot s0 to 40 "
                        "(state u[s0], forcing s0+1..40). Breaks the one-calendar confound; 0 = always slot 0")
    p.add_argument("--min-start", type=int, default=0, help="with --start-slots: draw the start slot uniformly from [min-start, start-slots] (a 'late dynamics' model when > 0)")
    p.add_argument("--no-t", action="store_true", help="zero the years-since-start input (so a 40-year rollout does not extrapolate a late model's clock)")
    p.add_argument("--p-zero", type=float, default=0.0,
                   help="with --start-slots: probability that a batch starts at slot 0 anyway (the competition's young "
                        "stands are slot-0 seedlings); the rest is uniform over 1..K")
    p.add_argument("--site-weights", default="",
                   help="npy of per-site sampling weights (all 54152 sites); fit sites are drawn with probability "
                        "proportional to the weight instead of uniformly (covariate shift toward the competition's site climate)")
    p.add_argument("--monthly", action="store_true", help="append the 108 monthly inputs (cgb.mortality cache) to each forcing year")
    p.add_argument("--augment", action="store_true",
                   help="per-batch random shortwave level (0.95-1.10), precipitation-anomaly shrink (0.5-1.0) and "
                        "temperature offset (sd 0.5 K) on the annual forcing; cgb.latent.Augment")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_dir = io.ROOT / "work" / "models" / f"latent_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = holdout.make_split(seed=42) if args.split == "random" else holdout.make_block_split(seed=42, tile_cells=args.tile_cells)
    fit = split.fit if not args.fit_limit else split.fit[: args.fit_limit]
    print(split.describe(), f"| fitting on {len(fit):,} sites, device {device}")
    t0 = time.time()
    arr = latent.load_arrays(split.fit, monthly=args.monthly)
    print(f"arrays loaded in {time.time() - t0:.0f}s: x_year {arr.x_year.shape}, u {arr.u.shape}")
    if args.drop_co2 and args.input_set == "full":
        args.input_set = "noco2"
    if args.input_set != "full":
        latent.apply_input_set(arr.x_year, arr.season, args.input_set)
        print(f"input set {args.input_set!r} applied (CO2 zeroed{'; hourly drivers zeroed' if args.input_set == 'core' else ''}"
              f"{'; annual drivers := 40-year site mean' if args.input_set == 'clim' else ''})")
    aug = latent.Augment() if args.augment else None
    stats_x = io.load_stats()
    x_mean_np, x_std_np = stats_x["x_mean"].astype(np.float32), np.where(stats_x["x_std"] == 0, 1.0, stats_x["x_std"]).astype(np.float32)
    if args.shuffle_forcing:
        perm = np.random.default_rng(12345).permutation(arr.x_year.shape[0])
        arr.x_year, arr.season, arr.static = arr.x_year[perm], arr.season[perm], arr.static[perm]
        print(f"NEGATIVE CONTROL: forcing permuted across sites (fixed point sites: {int((perm == np.arange(len(perm))).sum())})")

    model = latent.LatentRollout(n_drv=arr.x_year.shape[2], d_enc=args.d_enc, d_h=args.d_h, n_layers=args.layers, use_t=not args.no_t).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(fit) // args.batch_sites
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    print(f"model {n_params / 1e6:.2f}M params; {steps_per_epoch} steps/epoch x {args.epochs} epochs")

    log = []
    site_p = None
    if args.site_weights:
        w_all = np.load(args.site_weights).astype(np.float64)
        site_p = w_all[fit] / w_all[fit].sum()
        print(f"site weights from {args.site_weights}: ESS fraction {(site_p.sum() ** 2 / (site_p ** 2).sum()) / len(fit):.2f}")
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(fit) if site_p is None else rng.choice(fit, size=len(fit), replace=True, p=site_p)
        t_ep, running, running_epoch = time.time(), 0.0, 0.0
        for step in range(steps_per_epoch):
            sites = np.sort(order[step * args.batch_sites : (step + 1) * args.batch_sites])
            s0 = 0
            if args.min_start > 0:
                s0 = int(rng.integers(args.min_start, args.start_slots + 1))
            elif args.start_slots and rng.random() >= args.p_zero:
                s0 = int(rng.integers(1, args.start_slots + 1)) if args.p_zero > 0 else int(rng.integers(0, args.start_slots + 1))
            x, season, static, u0, agef, target = latent.batch_tensors(arr, sites, device, noise=True, rng=rng, s0=s0)
            if aug is not None:
                xa = aug(arr.x_year[sites], x_mean_np, x_std_np, rng)[:, s0:, :]
                x = torch.from_numpy(np.ascontiguousarray(xa)).to(device).repeat(15, 1, 1)
            pred = model(x, season, static, u0, agef)
            loss = latent.metric_loss(pred, target, w_last=args.w_last)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            running_epoch += loss.item()
            if (step + 1) % 200 == 0:
                print(f"    epoch {epoch + 1} step {step + 1}/{steps_per_epoch} loss {running / 200:.5f} ({(time.time() - t_ep) / (step + 1):.2f}s/step)", flush=True)
                running = 0.0
        # quick year-40 check on the public fold every epoch
        pred = latent.rollout_sites(model, arr, split.public, device)
        true40 = np.transpose(arr.u[:, split.public, 40, :2], (1, 0, 2))
        pred40 = np.transpose(pred[:, :, -1, :2], (1, 0, 2))
        s = latent.score_year40(pred40, true40)
        log.append({"epoch": epoch + 1, **s, "minutes": (time.time() - t_ep) / 60})
        pred10 = latent.rollout_sites(model, arr, split.public, device, s0=10)
        s10 = latent.score_year40(np.transpose(pred10[:, :, -1, :2], (1, 0, 2)), true40)
        pers10 = latent.score_year40(np.transpose(arr.u[:, split.public, 10, :2], (1, 0, 2)), true40)
        log[-1]["stage10_score"] = s10["score"]
        print(f"  epoch {epoch + 1}: public year-40 score {s['score']:.4f} (h {s['mse_height']:.4f}, a {s['mse_agb']:.4f}) | "
              f"from slot 10: {s10['score']:.4f} (persistence {pers10['score']:.4f})  {(time.time() - t_ep) / 60:.1f} min", flush=True)
        # collapse guard: the wide run died silently at epoch 4 (constant output, zero loss) and
        # held the GPU for six more epochs. A dead run must abort, and must not overwrite the last
        # healthy checkpoint.
        best_so_far = min(r["score"] for r in log[:-1]) if len(log) > 1 else float("inf")
        params_ok = all(torch.isfinite(p).all().item() for p in model.parameters())
        if not np.isfinite(s["score"]) or not params_ok or (len(log) > 1 and s["score"] > 3.0 * best_so_far) or running_epoch == 0.0:
            print(f"COLLAPSE at epoch {epoch + 1}: score {s['score']:.4f} vs best {best_so_far:.4f}, params finite {params_ok}, epoch loss sum {running_epoch:.6f}; "
                  f"checkpoint from epoch {epoch} kept, aborting", flush=True)
            sys.exit(3)
        ckpt = {"model": model.state_dict(), "season_mu": arr.season_mu, "season_sd": arr.season_sd, "args": vars(args)}
        if args.monthly:
            ckpt["monthly_mu"], ckpt["monthly_sd"] = arr.monthly_mu, arr.monthly_sd
        torch.save(ckpt, out_dir / "model.pt")

    # ---- full holdout evaluation ------------------------------------------------
    print("\n== holdout, year 40, competition units ==")
    results = {"log": log, "folds": {}}
    for fold, sites in (("public", split.public), ("private", split.private)):
        pred = latent.rollout_sites(model, arr, sites, device)  # (15, n, 40, 7)
        np.save(out_dir / f"pred_{fold}.npy", pred[:, :, [4, 9, 19, 29, 39], :2].astype(np.float32))
        true = arr.u[:, sites]  # (15, n, 41, 7)
        true40 = np.transpose(true[:, :, 40, :2], (1, 0, 2))
        pers40 = np.transpose(true[:, :, 0, :2], (1, 0, 2))
        persist_loss = ((pers40 - true40) ** 2).reshape(len(sites), -1).mean(1)
        pred40 = np.transpose(pred[:, :, -1, :2], (1, 0, 2))
        print(f"--- {fold} fold, {len(sites):,} sites ---")
        rows = [latent.breakdown("persistence", pers40, true40, sites, arr, persist_loss),
                latent.breakdown(f"latent_{args.tag}", pred40, true40, sites, arr, persist_loss)]
        for r in rows:
            latent.print_row(r)
        # drift curve: score at intermediate years
        drift = {}
        for t in (5, 10, 20, 30, 40):
            pt = np.transpose(pred[:, :, t - 1, :2], (1, 0, 2))
            tt = np.transpose(true[:, :, t, :2], (1, 0, 2))
            drift[t] = latent.score_year40(pt, tt)["score"]
        print("  drift (score at year t): " + "  ".join(f"t={t}: {v:.4f}" for t, v in drift.items()))
        # damping diagnostic: halve predicted agb losses at seed ages >= 100 (theory.md Q2).
        # On the holdout this must look worse -- the holdout carries the full decline --
        # it is printed so the size of the knob is known before a board slot is spent on it.
        damp = latent.damp_old_decline(pred40, pers40, factor=0.5)
        rows.append(latent.breakdown(f"latent_{args.tag}_damped", damp, true40, sites, arr, persist_loss))
        latent.print_row(rows[-1])
        results["folds"][fold] = {"rows": rows, "drift": drift}
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nsaved {out_dir}/model.pt, pred_*.npy, results.json")


if __name__ == "__main__":
    main()

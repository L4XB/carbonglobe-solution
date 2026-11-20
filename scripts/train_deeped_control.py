"""The control: an output-autoregressive one-step emulator trained the DeepED way.

A PyTorch re-implementation of the *protocol* of the benchmark's best model
(``LSTM-LSTNet-DeepED/run_deeped.py`` and ``models/model.py`` in the CarbonGlobe
repo), on the same annual-mean forcing the latent model sees, so that the paired
comparison isolates the one design choice under test -- feeding the predicted state
back versus carrying a latent state:

* one-step map ``u_t = clamp(u_{t-1} + f(x_t, u_{t-1}, age, site))``, residual on the
  *previous* state (DeepED: ``cur_out + cur_init``);
* trained with **teacher forcing** on the true previous state, plus random-walk noise
  on that input (paper Sec. 4.1: N(0, 1e-4) per step, cumulative);
* the **cumulative-window loss** of DeepED (``computeCumLossNYearsOverlap``): signed
  errors summed over overlapping 5-year windows and squared, weight 0.6;
* evaluated by a 40-step **autoregressive rollout** from the year-0 state, with the
  benchmark evaluator's rule that a site whose input height is below 1 m is
  predicted bare (``utils/func.py::eval``).

Not the original: the original encodes the 12 months of each forcing year with
attention-gated LSTMs and runs under TensorFlow. What is kept is everything the
paper credits for its drift control. Same split, same units, same breakdowns as
``scripts/train_latent.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import holdout, io, latent  # noqa: E402

HEIGHT_BARE_M = 1.0


class OneStep(nn.Module):
    """DeepED-style residual one-step emulator on annual-mean forcing."""

    def __init__(self, n_drv: int = 136, n_season: int = 72, n_static: int = 8, d_enc: int = 128, d_site: int = 128) -> None:
        super().__init__()
        self.enc_year = latent.MLP(n_drv + 1, 256, d_enc)
        self.enc_site = latent.MLP(n_season + n_static, 256, d_site)
        self.step = nn.Sequential(
            nn.Linear(d_enc + d_site + latent.N_AGE_FEATURES + 7, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 7),
        )
        nn.init.zeros_(self.step[-1].weight)  # start at persistence, as the latent model does
        nn.init.zeros_(self.step[-1].bias)
        self.register_buffer("cap", torch.tensor([latent.HEIGHT_CAP / 10.90645, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]))

    def encode(self, x_year: torch.Tensor, season: torch.Tensor, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_years = x_year.shape[1]
        t = torch.linspace(1.0 / n_years, 1.0, n_years, device=x_year.device).view(1, n_years, 1).expand(x_year.shape[0], -1, -1)
        return self.enc_year(torch.cat([x_year, t], dim=-1)), self.enc_site(torch.cat([season, static], dim=-1))

    def one_step(self, e_t: torch.Tensor, site: torch.Tensor, agef: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        delta = self.step(torch.cat([e_t, site, agef, u_prev], dim=-1))
        return torch.minimum(torch.clamp(u_prev + delta, min=0.0), self.cap)

    def teacher_forced(self, x_year, season, static, u_prev_all, agef) -> torch.Tensor:
        """All 40 steps at once from the (noised) true previous states: (S, 40, 7)."""
        e, site = self.encode(x_year, season, static)
        n_years = e.shape[1]
        site_t = site.unsqueeze(1).expand(-1, n_years, -1)
        agef_t = agef.unsqueeze(1).expand(-1, n_years, -1)
        delta = self.step(torch.cat([e, site_t, agef_t, u_prev_all], dim=-1))
        return torch.minimum(torch.clamp(u_prev_all + delta, min=0.0), self.cap)

    @torch.no_grad()
    def rollout(self, x_year, season, static, u0, agef, height_div: float) -> torch.Tensor:
        """Autoregressive 40-step rollout with the benchmark's bare-site rule."""
        e, site = self.encode(x_year, season, static)
        u, outs = u0, []
        bare_thresh = HEIGHT_BARE_M / height_div
        for t in range(e.shape[1]):
            bare = u[:, 0] < bare_thresh
            u = self.one_step(e[:, t], site, agef, u)
            u = u.clone()
            u[bare, 0] = 0.0
            outs.append(u)
        return torch.stack(outs, dim=1)


def cumulative_window_loss(err: torch.Tensor, window: int = 5) -> torch.Tensor:
    """DeepED's de-sequencing loss: signed errors summed over overlapping windows, squared."""
    # err: (S, 40, 7) -> windows via cumulative sums
    cs = torch.cat([torch.zeros_like(err[:, :1]), err.cumsum(dim=1)], dim=1)
    sums = cs[:, window:] - cs[:, :-window]  # (S, 40-window+1, 7)
    return (sums**2).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-sites", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--noise-std", type=float, default=1e-4, help="per-step random-walk noise on the input state (paper: 1e-4)")
    p.add_argument("--w-cumul", type=float, default=0.6)
    p.add_argument("--tag", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-limit", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_dir = io.ROOT / "work" / "models" / f"deeped_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = holdout.make_split(seed=42)
    fit = split.fit if not args.fit_limit else split.fit[: args.fit_limit]
    print(split.describe(), f"| fitting on {len(fit):,} sites, device {device}")
    arr = latent.load_arrays(split.fit)
    height_div = float(arr.y_mean[0])

    model = OneStep().to(device)
    steps_per_epoch = len(fit) // args.batch_sites
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    w_var = torch.tensor((1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25), device=device)
    print(f"model {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M params; {steps_per_epoch} steps/epoch; noise {args.noise_std}, w_cumul {args.w_cumul}")

    log = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(fit)
        t_ep, running = time.time(), 0.0
        for step in range(steps_per_epoch):
            sites = np.sort(order[step * args.batch_sites : (step + 1) * args.batch_sites])
            x, season, static, u0, agef, target = latent.batch_tensors(arr, sites, device)
            # teacher forcing: the true previous state for every step, plus cumulative random-walk noise
            u_prev = torch.cat([u0.unsqueeze(1), target[:, :-1]], dim=1)
            walk = torch.randn_like(u_prev).mul(args.noise_std).cumsum(dim=1)
            pred = model.teacher_forced(x, season, static, u_prev + walk, agef)
            err = (pred - target) * torch.sqrt(w_var)
            loss = (err**2).sum(-1).mean() + args.w_cumul * cumulative_window_loss(err)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if (step + 1) % 200 == 0:
                print(f"    epoch {epoch + 1} step {step + 1}/{steps_per_epoch} loss {running / 200:.5f}", flush=True)
                running = 0.0
        pred = rollout_sites(model, arr, split.public, device, height_div)
        true40 = np.transpose(arr.u[:, split.public, 40, :2], (1, 0, 2))
        s = latent.score_year40(np.transpose(pred[:, :, -1, :2], (1, 0, 2)), true40)
        log.append({"epoch": epoch + 1, **s})
        print(f"  epoch {epoch + 1}: public year-40 AR score {s['score']:.4f} (h {s['mse_height']:.4f}, a {s['mse_agb']:.4f})  {(time.time() - t_ep) / 60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "model.pt")

    print("\n== holdout, year 40, competition units, autoregressive rollout ==")
    results = {"log": log, "folds": {}}
    for fold, sites in (("public", split.public), ("private", split.private)):
        pred = rollout_sites(model, arr, sites, device, height_div)
        np.save(out_dir / f"pred_{fold}.npy", pred[:, :, [4, 9, 19, 29, 39], :2].astype(np.float32))
        true = arr.u[:, sites]
        true40 = np.transpose(true[:, :, 40, :2], (1, 0, 2))
        pers40 = np.transpose(true[:, :, 0, :2], (1, 0, 2))
        persist_loss = ((pers40 - true40) ** 2).reshape(len(sites), -1).mean(1)
        pred40 = np.transpose(pred[:, :, -1, :2], (1, 0, 2))
        print(f"--- {fold} fold, {len(sites):,} sites ---")
        rows = [latent.breakdown("persistence", pers40, true40, sites, arr, persist_loss),
                latent.breakdown(f"deeped_{args.tag}", pred40, true40, sites, arr, persist_loss)]
        for r in rows:
            latent.print_row(r)
        drift = {t: latent.score_year40(np.transpose(pred[:, :, t - 1, :2], (1, 0, 2)), np.transpose(true[:, :, t, :2], (1, 0, 2)))["score"] for t in (5, 10, 20, 30, 40)}
        print("  drift (score at year t): " + "  ".join(f"t={t}: {v:.4f}" for t, v in drift.items()))
        results["folds"][fold] = {"rows": rows, "drift": drift}
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nsaved {out_dir}")


@torch.no_grad()
def rollout_sites(model: OneStep, arr: latent.Arrays, sites: np.ndarray, device: torch.device, height_div: float,
                  batch_sites: int = 128) -> np.ndarray:
    model.eval()
    out = np.empty((15, len(sites), 40, 7), dtype=np.float32)
    for lo in range(0, len(sites), batch_sites):
        chunk = sites[lo : lo + batch_sites]
        x, season, static, u0, agef, _ = latent.batch_tensors(arr, chunk, device)
        pred = model.rollout(x, season, static, u0, agef, height_div).float().cpu().numpy()
        out[:, lo : lo + len(chunk)] = pred.reshape(15, len(chunk), 40, 7)
    return out


if __name__ == "__main__":
    main()

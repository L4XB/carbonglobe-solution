#!/usr/bin/env python
"""Autoregressive rollout against direct prediction, and the drift between them.

The organizers name the failure mode in writing: "always evaluate autoregressively
(feed each prediction forward) and watch the long-horizon cumulative drift, not just
average RMSE - that is what separates a strong emulator." The training set carries all
41 year slots, so that comparison is measurable on held-out sites with no leaderboard
slot spent.

Everything here is fitted on the 47,981 fit sites of ``cgb.holdout.make_split(42)``
and scored on the held-out 6,171. Two ways of getting to year 40 are compared, using
the *same* ridge-regression family and the *same* feature vocabulary, because the
question is not which architecture wins -- it is what feeding predictions forward
does to the error.

**one-step**, rolled 40 times. ``s_{k+1} = f(s_k, x_k)`` over all seven state
variables, fitted on every one of the 1.9 M transitions per seed age. Scored two ways
at each horizon: *teacher-forced* (the true ``s_{k-1}`` goes in, so only ever one step
of error is present) and *free-running* (the model's own output is fed back). The gap
between those two curves is the drift, and nothing but a free-running evaluation can
see it.

**direct-h**. A separate predictor of slot ``h`` from the initial state and the forcing
aggregated over years 0..h-1. No feedback, so error cannot compound.

Four one-step variants are rolled, because a single one would not distinguish "feeding
predictions forward is hard" from "I picked a bad model":

``lin/1step``   linear in the state, ridge strength chosen on **one-step** validation
                error -- the selection rule the organizers warn against.
``lin/roll``    the same model, ridge strength chosen on **40-step free-running**
                validation error instead. Same family, different selection rule.
``delta/roll``  the same features predicting ``s_{k+1} - s_k``, so ridge shrinks
                toward persistence rather than toward the mean. Rollout-selected.
``quad/1step``  state, squares and logs -- the vocabulary the best direct rung uses --
                selected on one-step error.

For the linear variants the 7x7 state Jacobian is available in closed form, so the
rollout's stability is not only observed but predicted: its spectral radius says
whether a 40-step rollout contracts or explodes.

    ./.venv/bin/python scripts/rollout_drift.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402

OUT = io.ROOT / "work" / "metrics"
HORIZONS = (1, 2, 5, 10, 20, 30, 40)
RIDGE_GRID = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
N_STATE = 7
CHUNK = 6000
BLOWN = 1e6  # a scaled score past this is divergence, not a score


class Std:
    """Fit-set standardiser; a constant column is emitted as zeros."""

    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.dead = sd <= 1e-12
        self.scale = np.where(self.dead, 1.0, sd)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.scale
        out[..., self.dead] = 0.0
        return out

    def invert(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale + self.mean


def quad_features(s: np.ndarray) -> np.ndarray:
    return np.concatenate([s, s**2, np.log1p(np.clip(s, 0.0, None))], axis=1)


def ridge_solve(gram: np.ndarray, rhs: np.ndarray, lam: float) -> np.ndarray:
    p = gram.shape[0] - 1
    penalty = np.eye(gram.shape[0]) * lam
    penalty[p, p] = 0.0
    return np.linalg.solve(gram + penalty, rhs)


def accumulate(design_fn, sites, n_features: int, n_targets: int):
    """Stream ``X^T X`` and ``X^T Y`` over site chunks so nothing large is held."""
    gram = np.zeros((n_features + 1, n_features + 1))
    rhs = np.zeros((n_features + 1, n_targets))
    rows = 0
    for lo in range(0, len(sites), CHUNK):
        x, y = design_fn(sites[lo : lo + CHUNK])
        design = np.empty((x.shape[0], n_features + 1))
        design[:, :n_features] = x
        design[:, n_features] = 1.0
        gram += design.T @ design
        rhs += design.T @ y
        rows += x.shape[0]
    return gram, rhs, rows


class OneStep:
    """One fitted transition model, ready to be rolled."""

    def __init__(self, beta, s_std, f_std, x_std, featurise, is_delta, clip=None):
        self.beta, self.s_std, self.f_std, self.x_std = beta, s_std, f_std, x_std
        self.featurise, self.is_delta, self.clip = featurise, is_delta, clip

    def step(self, s_phys: np.ndarray, x_year: np.ndarray) -> np.ndarray:
        feats = np.concatenate(
            [self.f_std(self.featurise(s_phys)), self.x_std(x_year)], axis=1
        )
        out = feats @ self.beta[:-1] + self.beta[-1]
        if self.is_delta:
            out = self.s_std(s_phys) + out
        s_next = self.s_std.invert(out)
        if self.clip is not None:
            s_next = np.clip(s_next, self.clip[0], self.clip[1])
        return s_next

    def jacobian(self) -> np.ndarray | None:
        """The 7x7 state Jacobian -- defined only for the linear feature set.

        The linear variant standardises its features with the *same* standardiser it
        standardises its targets with, so the map is affine with the same scaling on
        both sides and the physical and standardised Jacobians share their spectrum.
        A spectral radius above 1 means a 40-step rollout amplifies its own error
        geometrically; below 1 it forgets it.
        """
        if self.featurise is quad_features:
            return None
        if self.f_std is not self.s_std:
            raise AssertionError("the closed-form Jacobian assumes feature and target scaling agree")
        j = self.beta[:N_STATE, :N_STATE].T.copy()
        if self.is_delta:
            j += np.eye(N_STATE)
        return j


def roll(model: OneStep, y_true: np.ndarray, x: np.ndarray, div: np.ndarray):
    """Free-running and teacher-forced squared error at every horizon.

    ``y_true`` is ``(n_sites, 41, 7)`` physical, ``x`` is ``(n_sites, 40, 136)``.
    """
    free = y_true[:, 0, :].copy()
    free_err, tf_err, magnitude = {}, {}, {}
    for k in range(40):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            free = model.step(free, x[:, k, :])
            tf = model.step(y_true[:, k, :], x[:, k, :])
        h = k + 1
        if h in HORIZONS:
            truth = y_true[:, h, :2] / div
            with np.errstate(over="ignore", invalid="ignore"):
                free_err[h] = (free[:, :2] / div - truth) ** 2
            tf_err[h] = (tf[:, :2] / div - truth) ** 2
            magnitude[h] = float(np.nanmax(np.abs(free))) if np.isfinite(free).any() else np.inf
    return free_err, tf_err, magnitude


def safe_mean(err: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        v = float(np.mean(err))
    return v if np.isfinite(v) else float("inf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    div = cache.scoring_divisors()
    ages = io.SEED_AGES
    split = holdout.make_split(seed=42)
    hold = split.holdout
    pub_mask = np.isin(hold, split.public)

    traj = cache.open_traj()
    xann = cache.open_x_annual()
    print("== inputs ==")
    print(f"  trajectories {traj.shape}, annual forcing {xann.shape}")
    print("  " + split.describe())
    print(f"  ridge grid {RIDGE_GRID}\n")

    rng = np.random.default_rng(99)
    perm = rng.permutation(len(split.fit))
    cut = int(0.2 * len(split.fit))
    inner_val = split.fit[perm[:cut]]
    inner_fit = split.fit[perm[cut:]]

    x_hold = np.asarray(xann[hold], dtype=np.float64)
    x_val = np.asarray(xann[inner_val[::4]], dtype=np.float64)

    variants = ["lin/1step", "lin/roll", "delta/roll", "quad/1step", "quad/roll"]
    results: dict[str, dict] = {}

    for age in ages:
        j = cache.age_index(age)
        t_age = time.time()
        y_hold = np.asarray(traj[j, hold], dtype=np.float64)
        y_val = np.asarray(traj[j, inner_val[::4]], dtype=np.float64)

        sample = np.asarray(traj[j, split.fit[::7]], dtype=np.float64).reshape(-1, N_STATE)
        s_std = Std(sample)
        q_std = Std(quad_features(sample))
        x_std = Std(np.asarray(xann[split.fit[::7]], dtype=np.float64).reshape(-1, 136))
        lo_clip = sample.min(axis=0)
        hi_clip = sample.max(axis=0)
        del sample

        entry: dict[str, dict] = {"age": age}

        for kind, featurise, f_std in (("lin", lambda s: s, s_std), ("quad", quad_features, q_std)):
            n_feat = f_std.mean.shape[0] + 136

            def design(sites, j=j, featurise=featurise, f_std=f_std):
                y = np.asarray(traj[j, sites], dtype=np.float64)
                x = np.asarray(xann[sites], dtype=np.float64)
                cur = y[:, :40, :].reshape(-1, N_STATE)
                nxt = y[:, 1:, :].reshape(-1, N_STATE)
                feats = np.concatenate(
                    [f_std(featurise(cur)), x_std(x.reshape(-1, 136))], axis=1
                )
                zc, zn = s_std(cur), s_std(nxt)
                return feats, np.concatenate([zn, zn - zc], axis=1)

            g_in, r_in, _ = accumulate(design, inner_fit, n_feat, 2 * N_STATE)
            xv, yv = design(inner_val[::4])
            g_full, r_full, rows = accumulate(design, split.fit, n_feat, 2 * N_STATE)

            for is_delta in (False, True):
                sl = slice(N_STATE, 2 * N_STATE) if is_delta else slice(0, N_STATE)
                one_step_mse, roll_score = [], []
                for lam in RIDGE_GRID:
                    b = ridge_solve(g_in, r_in, lam)[:, sl]
                    one_step_mse.append(float((((xv @ b[:-1] + b[-1]) - yv[:, sl]) ** 2).mean()))
                    m = OneStep(b, s_std, f_std, x_std, featurise, is_delta)
                    fe, _, _ = roll(m, y_val, x_val, div)
                    roll_score.append(safe_mean(fe[40]))
                lam_1 = RIDGE_GRID[int(np.argmin(one_step_mse))]
                lam_r = RIDGE_GRID[int(np.argmin(roll_score))]

                if kind == "lin" and not is_delta:
                    wanted = [("lin/1step", lam_1), ("lin/roll", lam_r)]
                elif kind == "lin" and is_delta:
                    wanted = [("delta/roll", lam_r)]
                elif kind == "quad" and not is_delta:
                    wanted = [("quad/1step", lam_1), ("quad/roll", lam_r)]
                else:
                    continue

                for key, lam in wanted:
                    b = ridge_solve(g_full, r_full, lam)[:, sl]
                    model = OneStep(b, s_std, f_std, x_std, featurise, is_delta)
                    fe, te, mag = roll(model, y_hold, x_hold, div)
                    jac = model.jacobian()
                    rho = float(np.abs(np.linalg.eigvals(jac)).max()) if jac is not None else None
                    entry[key] = {
                        "lambda": lam,
                        "spectral_radius": rho,
                        "free": {str(h): safe_mean(fe[h][pub_mask]) for h in HORIZONS},
                        "teacher_forced": {str(h): safe_mean(te[h][pub_mask]) for h in HORIZONS},
                        "max_abs_state": {str(h): mag[h] for h in HORIZONS},
                        "roll_val_curve": dict(zip(map(str, RIDGE_GRID), roll_score)),
                        "onestep_val_curve": dict(zip(map(str, RIDGE_GRID), one_step_mse)),
                    }
                    if key == "lin/roll":
                        m_clip = OneStep(b, s_std, f_std, x_std, featurise, is_delta, (lo_clip, hi_clip))
                        fe_c, _, _ = roll(m_clip, y_hold, x_hold, div)
                        entry["lin/roll+clip"] = {
                            "lambda": lam,
                            "spectral_radius": rho,
                            "free": {str(h): safe_mean(fe_c[h][pub_mask]) for h in HORIZONS},
                        }

        # ---- direct-h -------------------------------------------------------
        direct = {}
        for h in HORIZONS:
            years = np.arange(h, dtype=np.float64)
            years = years - years.mean()
            denom = float((years**2).sum()) or 1.0
            n_dir = 3 * N_STATE + 272

            cm = np.asarray(xann[split.fit[::7], :h, :], dtype=np.float64)
            c_std = Std(
                np.concatenate([cm.mean(axis=1), np.tensordot(years, cm, axes=(0, 1)) / denom], axis=1)
            )
            del cm

            def dd(sites, j=j, h=h, years=years, denom=denom, c_std=c_std):
                y = np.asarray(traj[j, sites], dtype=np.float64)
                x = np.asarray(xann[sites, :h, :], dtype=np.float64)
                clim = c_std(
                    np.concatenate(
                        [x.mean(axis=1), np.tensordot(years, x, axes=(0, 1)) / denom], axis=1
                    )
                )
                return np.concatenate([q_std(quad_features(y[:, 0, :])), clim], axis=1), y[:, h, :2] / div

            g_in, r_in, _ = accumulate(dd, inner_fit, n_dir, 2)
            xv, yv = dd(inner_val)
            best, best_mse = RIDGE_GRID[0], np.inf
            for lam in RIDGE_GRID:
                bb = ridge_solve(g_in, r_in, lam)
                mse = float((((xv @ bb[:-1] + bb[-1]) - yv) ** 2).mean())
                if mse < best_mse:
                    best_mse, best = mse, lam
            del xv, yv
            g_full, r_full, _ = accumulate(dd, split.fit, n_dir, 2)
            b = ridge_solve(g_full, r_full, best)
            fh, yh = dd(hold)
            err = ((fh @ b[:-1] + b[-1]) - yh) ** 2
            direct[str(h)] = float(err[pub_mask].mean())
            if h == 40:
                np.save(OUT / f"direct_err40_age{age:03d}.npy", err)
        entry["direct"] = direct

        results[str(age)] = entry
        print(
            f"  age {age:>3}  "
            + "  ".join(
                f"{v.split('/')[0][:5]}/{v.split('/')[1][:5]} {entry[v]['free']['40']:.3g}"
                for v in variants
            )
            + f"  direct40 {direct['40']:.4f}   ({time.time() - t_age:.1f}s)",
            flush=True,
        )

    # ---------------------------------------------------------------- report
    def pooled(variant: str, field: str, h: int) -> float:
        vals = [results[str(a)][variant][field][str(h)] for a in ages]
        return float(np.mean(vals)) if all(np.isfinite(vals)) else float("inf")

    print("\n== drift curve, public fold, pooled over all 15 seed ages (scaled units) ==")
    shown = ["lin/1step", "lin/roll", "lin/roll+clip", "delta/roll", "quad/1step", "quad/roll"]
    print(f"  {'horizon':>7s} " + "".join(f"{s:>16s}" for s in shown) + f"{'direct-h':>12s}")
    for h in HORIZONS:
        cells = []
        for s in shown:
            v = pooled(s, "free", h)
            cells.append("diverged" if v > BLOWN else f"{v:.5f}")
        d = float(np.mean([results[str(a)]["direct"][str(h)] for a in ages]))
        print(f"  {h:7d} " + "".join(f"{c:>16s}" for c in cells) + f"{d:12.5f}")

    print("\n== the same models, teacher-forced: one step of error only ==")
    tf_shown = ["lin/1step", "lin/roll", "delta/roll", "quad/1step", "quad/roll"]
    print(f"  {'horizon':>7s} " + "".join(f"{s:>16s}" for s in tf_shown))
    for h in HORIZONS:
        print(
            f"  {h:7d} "
            + "".join(f"{pooled(s, 'teacher_forced', h):16.5f}" for s in tf_shown)
        )

    print("\n== why: the 7x7 state Jacobian's spectral radius, per seed age ==")
    print(f"  {'age':>4s} {'lin/1step rho':>14s} {'lambda':>9s} {'lin/roll rho':>14s} {'lambda':>9s} {'delta/roll rho':>15s}")
    for age in ages:
        e = results[str(age)]
        print(
            f"  {age:4d} {e['lin/1step']['spectral_radius']:14.4f} {e['lin/1step']['lambda']:9g} "
            f"{e['lin/roll']['spectral_radius']:14.4f} {e['lin/roll']['lambda']:9g} "
            f"{e['delta/roll']['spectral_radius']:15.4f}"
        )

    print("\n== year 40, per seed age: best rollout vs direct ==")
    print(f"  {'age':>4s} {'lin/roll':>12s} {'lin/roll+clip':>15s} {'direct-40':>12s} {'roll/direct':>12s}")
    for age in ages:
        e = results[str(age)]
        r = e["lin/roll"]["free"]["40"]
        c = e["lin/roll+clip"]["free"]["40"]
        d = e["direct"]["40"]
        print(f"  {age:4d} {r:12.5f} {c:15.5f} {d:12.5f} {min(r, c) / d:12.2f}x")

    (OUT / "drift.json").write_text(json.dumps({"horizons": list(HORIZONS), "per_age": results}, indent=2))
    print(f"\nwrote {OUT / 'drift.json'}")


if __name__ == "__main__":
    main()

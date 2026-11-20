"""The latent-state rollout: autoregressive in a hidden state, never in the outputs.

Why this shape (``notes/theory.md`` Sec. 3): the seven reported variables are a
partial observation of ED's state, so a model that feeds its own seven-vector back
is a POMDP approximation whose one-step bias compounds 40x by the scored year. Here
a recurrent state ``h_t`` is initialised from the decoded initial state, the seed age
and the site's static and seasonal descriptors, driven by one encoded forcing year
per step, and read out at every year as a *residual on the initial state*:

    h_0   = g(u_0, age, site)
    h_t   = GRU(h_{t-1}, phi(x_t, t))
    u_t^  = clamp(u_0 + f(h_t, u_0, age, site))

so persistence is the zero function, nothing predicted is ever fed back, all 40
labelled years supervise the trajectory shape, and the loss is taken in the
competition's units (each variable divided by its global mean) with year 40
weighted above the rest.

Inputs are the caches built by ``scripts/cache_arrays.py``: annual-mean drivers,
the 40-year seasonal climatology and the seven-variable trajectories. Site latitude
is used for *evaluation breakdowns only* -- the competition ships no coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import cache, io

#: Seed-age feature block: age/500, the age-mean state and 40x the age slope, each
#: scaled by the global means so it lives in the same units as the targets.
N_AGE_FEATURES = 15
#: Initial-state channels the model may look at: height, agb, soil, lai. The three
#: fluxes are left out on purpose -- their decode in ``test_y0`` is on a footing no
#: offline measurement has pinned (``theory.md`` Sec. 5 Q1).
STATE_IN = (0, 1, 2, 3)
#: Height cap of the process model, in scaled units.
HEIGHT_CAP = 35.0002
#: Hourly-of-day driver groups (24 each) whose daily mean forms the seasonal block.
HOURLY_GROUPS = {"hus": 33, "ta_h": 34, "rsds": 35, "wind": 36}
#: The 24 hourly CO2 drivers. In the training forcing they are a near-deterministic
#: function of the forcing year at every site (358 -> 420 ppm) and a weak spatial
#: fingerprint; the competition series is one flat curve for all sites (~367 ppm in
#: ssp126), two to three fit-set standard deviations below the training level. A
#: model trained with them cannot separate CO2 from time-in-run and mis-reads the
#: competition series (``notes/board-gap.md``); ``--drop-co2`` zeroes them.
CO2_DRIVERS = slice(8, 32)


def site_latitudes() -> np.ndarray:
    """Latitude of every site in site order, from the 0.5 degree mask."""
    mask = np.load(io.CACHE / "mask2d.npy")
    flat = np.flatnonzero(mask.ravel() > 0)
    row = flat // mask.shape[1]
    return 90.0 - 0.5 * row - 0.25


def seasonal_block(x_season: np.ndarray) -> np.ndarray:
    """``(n_sites, 72)``: monthly climatology of T_air, precip and the four daily-mean hourly groups."""
    n = x_season.shape[0]
    cols = [np.asarray(x_season[:, cache.D_TAIR, :]), np.asarray(x_season[:, cache.D_PRECIP, :])]
    for start in HOURLY_GROUPS.values():
        idx = start + 4 * np.arange(24)  # the 96 hourly drivers interleave (q, T, SW, wind)
        cols.append(np.asarray(x_season[:, idx, :]).mean(axis=1))
    return np.concatenate([c.reshape(n, 12) for c in cols], axis=1).astype(np.float32)


def age_features(stats: dict[str, np.ndarray]) -> np.ndarray:
    """``(15, 15)`` per seed age: age/500, age-mean state / y_mean, 40 * age slope / y_mean."""
    tri = np.load(io.CACHE / "age_triplet.npz")
    y_mean = stats["y_mean"]
    return np.concatenate(
        [
            (tri["age"] / 500.0)[:, None],
            tri["age_mean"] / y_mean,
            40.0 * tri["age_slope"] / y_mean,
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class Arrays:
    """Everything the model consumes, already scaled, all sites."""

    x_year: np.ndarray  # (n_sites, 40, 136) z-scored annual drivers
    season: np.ndarray  # (n_sites, 72) z-scored seasonal block
    static: np.ndarray  # (n_sites, 8) z-scored static drivers
    u: np.ndarray  # (15, n_sites, 41, 7) trajectories / y_mean
    agef: np.ndarray  # (15, 15)
    y_mean: np.ndarray  # (7,)
    lat: np.ndarray  # (n_sites,)
    season_mu: np.ndarray
    season_sd: np.ndarray


#: Monthly block: the nine monthly series of ``cgb.mortality`` (ta_m, pr, tsl1, hus, ta_h, rsds, wind,
#: ta_hmax, hus_min) x 12 months = 108 inputs per year, appended after the 136 annual means. The
#: annual means hide the seasonal timing of an anomaly (a failed wet season against a dry-season
#: deficit), which is what ED's carbon-balance mortality responds to; every model built from annual
#: means shares one old-band change pattern on the board (2026-09-27), so this is the one input
#: channel none of them had.
N_MONTHLY_INPUTS = 108
MONTHLY_HOURLY_CHANNELS = (3, 4, 5, 6, 7, 8)  # derived from the hourly groups; zeroed by the 'core' set


def monthly_block(monthly: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    """``(n, 40, 108)`` z-scored monthly inputs from a ``(n, 40, 12, 9)`` cache; returns (block, mu, sd)."""
    n = monthly.shape[0]
    blk = np.empty((n, 40, N_MONTHLY_INPUTS), dtype=np.float32)
    for lo in range(0, n, 4000):
        blk[lo:lo + 4000] = np.asarray(monthly[lo:lo + 4000], dtype=np.float32).reshape(-1, 40, N_MONTHLY_INPUTS)
    if mu is None:
        mu = blk.reshape(-1, N_MONTHLY_INPUTS).mean(0)
        sd = blk.reshape(-1, N_MONTHLY_INPUTS).std(0) + 1e-6
    blk -= mu
    blk /= sd
    return blk, mu, sd


def load_arrays(fit_sites: np.ndarray, monthly: bool = False) -> Arrays:
    """Load the caches into RAM (about 2.2 GB, 3.2 GB with the monthly block) and scale them. Scalers see fit sites only."""
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"].astype(np.float32), stats["x_std"].astype(np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)
    x_annual = np.asarray(cache.open_x_annual(), dtype=np.float32)
    x_year = (x_annual - x_mean) / x_std
    if monthly:
        from . import mortality as mort
        m_all = mort.open_monthly("train")
        _, mmu, msd = monthly_block(m_all[fit_sites[::4]])
        blk, _, _ = monthly_block(m_all, mmu, msd)
        x_year = np.concatenate([x_year, blk], axis=2)
        del blk
    static = x_year[:, 0, list(cache.STATIC_DRIVERS)].copy()
    season = seasonal_block(cache.open_x_season())
    mu = season[fit_sites].mean(axis=0)
    sd = season[fit_sites].std(axis=0) + 1e-6
    season = (season - mu) / sd
    y_mean = stats["y_mean"].astype(np.float32)
    u = np.asarray(cache.open_traj(), dtype=np.float32) / y_mean
    arr = Arrays(x_year, season, static.astype(np.float32), u, age_features(stats), y_mean, site_latitudes(), mu, sd)
    if monthly:
        arr.monthly_mu, arr.monthly_sd = mmu, msd  # type: ignore[attr-defined]
    return arr


class MLP(nn.Sequential):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__(nn.Linear(d_in, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_out))


class LatentRollout(nn.Module):
    """See the module docstring. ``forward`` returns scaled states for years 1..40."""

    def __init__(self, n_drv: int = 136, n_season: int = 72, n_static: int = 8, d_enc: int = 128,
                 d_site: int = 128, d_h: int = 256, n_layers: int = 2, use_t: bool = True) -> None:
        super().__init__()
        self.use_t = use_t
        self.n_layers, self.d_h = n_layers, d_h
        n_ctx = d_site + N_AGE_FEATURES + len(STATE_IN)
        self.enc_year = MLP(n_drv + 1, 256, d_enc)
        self.enc_site = MLP(n_season + n_static, 256, d_site)
        self.enc_init = MLP(n_ctx, 512, n_layers * d_h)
        self.gru = nn.GRU(d_enc + n_ctx, d_h, n_layers, batch_first=True)
        self.head = MLP(d_h + n_ctx, 256, 7)
        # start exactly at persistence: a zero residual is the right prior for 90% of the
        # old-age variance and lets the first epochs learn the *change* rather than undo noise
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("cap", torch.tensor([HEIGHT_CAP / 10.90645, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]))

    def forward(self, x_year: torch.Tensor, season: torch.Tensor, static: torch.Tensor,
                u0: torch.Tensor, agef: torch.Tensor) -> torch.Tensor:
        # x_year: (S, 40, 136)  season: (S, 72)  static: (S, 8)  u0: (S, 7)  agef: (S, 15)
        s_count, n_years = x_year.shape[0], x_year.shape[1]
        # years since the rollout start over the 40-year horizon: a shorter training window (random
        # start slot) sees the same clock as the first steps of a full rollout
        t = (torch.arange(1, n_years + 1, device=x_year.device, dtype=x_year.dtype) / 40.0).view(1, n_years, 1).expand(s_count, -1, -1)
        if not self.use_t:
            t = torch.zeros_like(t)
        e = self.enc_year(torch.cat([x_year, t], dim=-1))  # (S, 40, d_enc)
        site = self.enc_site(torch.cat([season, static], dim=-1))  # (S, d_site)
        ctx = torch.cat([site, agef, u0[:, list(STATE_IN)]], dim=-1)  # (S, n_ctx)
        h0 = torch.tanh(self.enc_init(ctx)).view(s_count, self.n_layers, self.d_h).transpose(0, 1).contiguous()
        ctx_t = ctx.unsqueeze(1).expand(-1, n_years, -1)
        out, _ = self.gru(torch.cat([e, ctx_t], dim=-1), h0)  # (S, 40, d_h)
        delta = self.head(torch.cat([out, ctx_t], dim=-1))  # (S, 40, 7)
        u_hat = u0.unsqueeze(1) + delta
        return torch.minimum(torch.clamp(u_hat, min=0.0), self.cap)


_LOSS_WEIGHTS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def metric_loss(u_hat: torch.Tensor, u_true: torch.Tensor, w_last: float = 10.0,
                w_var: tuple[float, ...] = (1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25)) -> torch.Tensor:
    """Squared error in competition units over all 40 years, year 40 weighted ``w_last``.

    The weight tensors are built once per (device, horizon, weights) and cached: an
    allocation on the accelerator at every step is exactly the churn that produced an
    MPS fault in the 40-epoch run (a garbage index into a 7-vector, nothing to do
    with the data).
    """
    n_years = u_hat.shape[1]
    key = (str(u_hat.device), n_years, w_last, w_var)
    if key not in _LOSS_WEIGHTS:
        wt = torch.ones(n_years)
        wt[-1] = w_last
        _LOSS_WEIGHTS[key] = ((wt / wt.sum()).to(u_hat.device), torch.tensor(w_var).to(u_hat.device))
    wt, wv = _LOSS_WEIGHTS[key]
    se = (u_hat - u_true) ** 2  # (S, 40, 7)
    return (se * wv).sum(-1).mul(wt).sum(-1).mean()


_AGEF_SHIFTED: dict[int, np.ndarray] = {}


def age_features_shifted(agef: np.ndarray, s0: int) -> np.ndarray:
    """The age-feature table for stands that are ``s0`` years older than their seed age.

    A window that starts at slot ``s0`` holds a stand of age ``seed + s0``; feeding it the seed
    age's features would teach the model two inconsistent input regimes. The 15-row table is
    interpolated linearly in log age (the triplet is smooth in age; beyond 500 it is held).
    """
    if s0 == 0:
        return agef
    if s0 not in _AGEF_SHIFTED:
        ages = np.array(io.SEED_AGES, dtype=np.float64)
        la = np.log(ages)
        out = np.empty_like(agef)
        for j, a in enumerate(ages):
            t = np.log(min(a + s0, ages[-1]))
            k = min(max(int(np.searchsorted(la, t) - 1), 0), len(ages) - 2)
            w = (t - la[k]) / (la[k + 1] - la[k])
            out[j] = agef[k] + w * (agef[k + 1] - agef[k])
        _AGEF_SHIFTED[s0] = out.astype(np.float32)
    return _AGEF_SHIFTED[s0]


def batch_tensors(arr: Arrays, sites: np.ndarray, device: torch.device, *, noise: bool = False,
                  rng: np.random.Generator | None = None, s0: int = 0) -> tuple[torch.Tensor, ...]:
    """All 15 seed ages of ``sites``, flattened to ``15 * len(sites)`` sequences (age-major).

    ``s0`` starts the window at year slot ``s0`` (state ``u[s0]``, forcing years ``s0+1..40``,
    targets slots ``s0+1..40``): a random start slot in training breaks the one-calendar
    confound of the training forcing (every site sees 1982-83 in years 1-2 of a slot-0 window),
    so a site's early losses can no longer be memorised as a site property (``notes/board-solve.md``).
    """
    b = len(sites)
    x = torch.from_numpy(np.ascontiguousarray(arr.x_year[sites][:, s0:, :])).to(device).repeat(15, 1, 1)
    season = torch.from_numpy(arr.season[sites]).to(device).repeat(15, 1)
    static = torch.from_numpy(arr.static[sites]).to(device).repeat(15, 1)
    u = arr.u[:, sites]  # (15, b, 41, 7)
    u0 = u[:, :, s0, :].reshape(15 * b, 7).copy()
    if noise:
        # the soil/lai decode of test_y0 is only known to ~10-15%, the fluxes worse:
        # perturb those channels so the model cannot lean on their exact scale
        assert rng is not None
        u0[:, 2:4] *= np.exp(rng.normal(0.0, 0.15, size=(15 * b, 2))).astype(np.float32)
        u0[:, 4:7] *= np.exp(rng.normal(0.0, 0.30, size=(15 * b, 3))).astype(np.float32)
    u0_t = torch.from_numpy(u0).to(device)
    target = torch.from_numpy(np.ascontiguousarray(u[:, :, s0 + 1:, :].reshape(15 * b, 40 - s0, 7))).to(device)
    agef = torch.from_numpy(np.repeat(age_features_shifted(arr.agef, s0), b, axis=0)).to(device)  # age-major to match repeat()
    return x, season, static, u0_t, agef, target


@torch.no_grad()
def rollout_sites(model: LatentRollout, arr: Arrays, sites: np.ndarray, device: torch.device,
                  batch_sites: int = 128, s0: int = 0) -> np.ndarray:
    """Predicted scaled states, ``(15, n_sites, 40 - s0, 7)``, for all seed ages of ``sites``, from slot ``s0``."""
    model.eval()
    out = np.empty((15, len(sites), 40 - s0, 7), dtype=np.float32)
    for lo in range(0, len(sites), batch_sites):
        chunk = sites[lo : lo + batch_sites]
        x, season, static, u0, agef, _ = batch_tensors(arr, chunk, device, s0=s0)
        pred = model(x, season, static, u0, agef).float().cpu().numpy()
        out[:, lo : lo + len(chunk)] = pred.reshape(15, len(chunk), 40 - s0, 7)
    return out


def score_year40(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Competition score on ``(n_sites, 15, 2)`` scaled height/agb blocks."""
    se = (pred - true) ** 2
    return {"score": float(se.mean()), "mse_height": float(se[..., 0].mean()), "mse_agb": float(se[..., 1].mean())}


#: A persistence-like holdout score overstates the board by about this factor
#: (``notes/board-anchors.md``: 1.09 of population variance x 1.13 of damped trajectories).
BOARD_FACTOR = 1.24


def breakdown(name: str, pred40: np.ndarray, true40: np.ndarray, sites: np.ndarray, arr: Arrays,
              persist_loss: np.ndarray) -> dict:
    """Year-40 score overall, by seed-age band, by latitude band and on the heavy-tail sites.

    ``pred40``/``true40`` are ``(n_sites, 15, 2)`` scaled height/agb; ``persist_loss`` is the
    per-site persistence loss used to name the top-10% sites, so every model is judged on
    the same tail.
    """
    from . import holdout

    res = {"name": name, **score_year40(pred40, true40)}
    se = ((pred40 - true40) ** 2).reshape(len(sites), -1)
    res["se"] = holdout.site_bootstrap(se, n_boot=1000, seed=7)["se"]
    ages = np.array(io.SEED_AGES)
    for label, m in (("age1-30", ages <= 30), ("age50-100", (ages >= 50) & (ages <= 100)), ("age150+", ages >= 150)):
        res[label] = score_year40(pred40[:, m], true40[:, m])["score"]
    trop = np.abs(arr.lat[sites]) < 23.5
    res["tropical"] = score_year40(pred40[trop], true40[trop])["score"]
    res["extratropical"] = score_year40(pred40[~trop], true40[~trop])["score"]
    top = persist_loss >= np.percentile(persist_loss, 90)
    res["top10pct_persistence_sites"] = score_year40(pred40[top], true40[top])["score"]
    res["board_estimate"] = res["score"] / BOARD_FACTOR
    return res


def print_row(r: dict) -> None:
    print(f"  {r['name']:24s} score {r['score']:.4f} (se {r['se']:.4f})  h {r['mse_height']:.4f} a {r['mse_agb']:.4f} | "
          f"age1-30 {r['age1-30']:.4f} age50-100 {r['age50-100']:.4f} age150+ {r['age150+']:.4f} | "
          f"trop {r['tropical']:.4f} extra {r['extratropical']:.4f} | top10% {r['top10pct_persistence_sites']:.4f} | "
          f"/{BOARD_FACTOR} -> {r['board_estimate']:.4f}")


def damp_old_decline(pred40: np.ndarray, pers40: np.ndarray, factor: float = 0.5, min_age: int = 100) -> np.ndarray:
    """Scale predicted agb *losses* at seed ages >= ``min_age`` by ``factor`` (theory.md Q2 knob)."""
    out = pred40.copy()
    old = np.array(io.SEED_AGES) >= min_age
    change = out[:, old, 1] - pers40[:, old, 1]
    out[:, old, 1] = pers40[:, old, 1] + np.where(change < 0, factor * change, change)
    return out


# ---------------------------------------------------------------------------
# Input sets and training-time augmentation (2026-09-25).
#
# The board showed the trained model transferring at ~9x its holdout error while
# persistence transferred at 0.8x. ``scripts/gap_forcing_sensitivity.py`` puts the
# CO2 block (a clock in the training forcing, flat in the competition's) at 0.034 of
# holdout MSE, the competition pipeline's site descriptors (hourly-group climatology,
# soil) at ~0.007, shortwave level and precipitation anomaly size at ~0.005 each.
# The input sets below remove those levers one at a time so the board can price
# each; ``augment_batch`` randomises the levers the physics keeps (shortwave level,
# anomaly size, temperature offset) so a single model cannot key on their exact
# training-set values.
# ---------------------------------------------------------------------------

#: The 96 hourly drivers (q, T, SW, wind interleaved, 24 hours each).
HOURLY_DRIVERS = slice(33, 129)
#: Columns of the 72-dim seasonal block that come from the hourly groups.
SEASON_HOURLY_COLS = slice(24, 72)
#: Physical driver groups used by the augmentation.
SW_DRIVERS = [35 + 4 * h for h in range(24)]
TH_DRIVERS = [34 + 4 * h for h in range(24)]
TEMP_DRIVERS = [0] + list(range(2, 8)) + TH_DRIVERS
PRECIP_DRIVER = 1

INPUT_SETS = ("full", "noco2", "core", "clim", "noclim")


def apply_input_set(x_year: np.ndarray, season: np.ndarray, input_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_year, season)`` reduced to ``input_set``; arrays are modified in place when possible.

    ``full``  -- everything (the v1 models).
    ``noco2`` -- the 24 CO2 drivers zeroed in the annual inputs.
    ``core``  -- ``noco2`` plus the 96 hourly drivers zeroed in the annual inputs and the
                 hourly-group daily means zeroed in the seasonal block: the model sees monthly
                 T/P/soil-T, their climatology and the soil, i.e. the drivers whose pipeline is
                 the same in both files.
    ``clim``  -- ``noco2`` and every annual driver replaced by its own 40-year site mean, so no
                 interannual information reaches the model (a climatology-only rollout; the year
                 index ``t`` still does).
    ``noclim`` -- ``noco2`` with the seasonal block zeroed: the site is described only by its soil
                 and by the forcing years themselves, so a site's identity cannot be memorised
                 through its climatology (the old-band pattern of every descriptor-driven model is
                 the same; this one is forcing-driven by construction).
    """
    if input_set not in INPUT_SETS:
        raise ValueError(f"unknown input set {input_set!r}; choose from {INPUT_SETS}")
    if input_set == "full":
        return x_year, season
    x_year[:, :, CO2_DRIVERS] = 0.0
    if input_set == "core":
        x_year[:, :, HOURLY_DRIVERS] = 0.0
        season[:, SEASON_HOURLY_COLS] = 0.0
        if x_year.shape[2] > 136:  # monthly block present: drop its hourly-derived channels as well
            for ch in MONTHLY_HOURLY_CHANNELS:
                x_year[:, :, 136 + ch * 12:136 + (ch + 1) * 12] = 0.0
    elif input_set == "clim":
        for lo in range(0, x_year.shape[0], 2048):
            blk = x_year[lo : lo + 2048]
            blk[:] = blk.mean(axis=1, keepdims=True)
    elif input_set == "noclim":
        season[:] = 0.0
    return x_year, season


@dataclass
class Augment:
    """Per-site random perturbations of the z-scored annual forcing, in physical terms."""

    sw_range: tuple[float, float] = (0.95, 1.10)  # shortwave level factor
    anom_range: tuple[float, float] = (0.5, 1.0)  # shrink of precipitation anomalies toward the site mean
    temp_sd: float = 0.5  # K, common offset on every temperature driver
    p_apply: float = 0.7

    def __call__(self, x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``x`` is ``(b, 40, 136)`` z-scored; returns a perturbed copy."""
        b = x.shape[0]
        out = x.copy()
        apply = rng.random(b) < self.p_apply
        if not apply.any():
            return out
        f = rng.uniform(*self.sw_range, size=b)
        f[~apply] = 1.0
        mu, sd = x_mean[SW_DRIVERS], x_std[SW_DRIVERS]
        # z' = (f*x - mu)/sd = f*z + (f-1)*mu/sd
        out[:, :, SW_DRIVERS] = f[:, None, None] * out[:, :, SW_DRIVERS] + ((f - 1.0)[:, None, None] * (mu / sd)[None, None, :])
        g = rng.uniform(*self.anom_range, size=b)
        g[~apply] = 1.0
        zbar = out[:, :, PRECIP_DRIVER].mean(axis=1, keepdims=True)
        out[:, :, PRECIP_DRIVER] = zbar + g[:, None] * (out[:, :, PRECIP_DRIVER] - zbar)
        d = rng.normal(0.0, self.temp_sd, size=b)
        d[~apply] = 0.0
        out[:, :, TEMP_DRIVERS] += (d[:, None, None] / x_std[TEMP_DRIVERS][None, None, :])
        return out.astype(np.float32, copy=False)

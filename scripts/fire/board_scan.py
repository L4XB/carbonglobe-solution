#!/usr/bin/env python
"""Fit the fire-climate hypothesis of the TEST windows to all scored board files (forward model = fire emulator on
old-stand agb of tropical forest sites; criterion = chi2 of the probe residuals under truth = C + h(theta)).
    python scripts/fire/board_scan.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import shift  # noqa: E402
import emulator as em  # noqa: E402
import build_dir as bd  # noqa: E402
import consistency_all as ca  # noqa: E402

DIV = bd.DIV; M = bd.M


class Scan:
    def __init__(self) -> None:
        L = ca.load(); self.names = L["names"]; site = L["site"]; sc = L["sc"]; cen = L["cen"]; D = L["D"]; CC = L["CC"]; rho = L["rho"]
        k = shift.template_keys(); self.site = site; age = k.age.to_numpy(); scen = k.scenario.to_numpy()
        self.C = shift.read_predictions(bd.C_PATH).astype(float); self.R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
        g = bd.groups(); self.g = g; self.a0 = g["a0"]; ns = len(self.a0)
        tw = np.load(ROOT / "work/train_twin_states.npz"); tr = np.full(ns, np.nan)
        tr[tw["sites"]] = np.log(np.maximum(tw["tr40"].mean(2)[:, 7:, 1].mean(1), 0.02) / np.maximum(tw["tr0"][:, 7:, 0, 1].mean(1), 0.02))
        z = np.load(ROOT / "work/fire/test_W.npz"); self.Wm = np.nan_to_num(z["W_mer"]); self.Wf = z["W_fut"]
        zr = np.load(ROOT / "work/fire/test_W_rel.npz"); self.Wr = zr["rel"]
        zc = np.load(ROOT / "work/fire/test_W_clim.npz"); self.Wc = {kk: zc[kk] for kk in ("clim_40_yr", "clim_last_20_yr", "clim_last_10_yr", "clim_first_10_yr", "last_year")}
        fire = g["forest"] & ((np.nan_to_num(tr) < -0.1) | (self.Wm.max(1) * self.a0 * 1.15e-5 > 0.02) | g["cl"])
        self.sel = fire; old = age >= 150
        self.rows = {s_: np.where(old & (scen == s_) & fire[site])[0] for s_ in ("ssp126", "ssp585")}
        share = np.full(6171, 0.5)                                    # plain public share (cluster posterior omitted here: second order)
        nrm = rho / (D.shape[1] * 2)
        allrows = np.concatenate([self.rows["ssp126"], self.rows["ssp585"]])
        self.A = D[:, allrows, 1].astype(np.float64) * nrm            # (n_probe, n_rows): inner product operator on agb of the touched rows
        E = np.einsum("nrc,nrc->n", D, D) * nrm; base = np.einsum("nrc,nrc->n", D, CC) * nrm
        s_p = np.array([sc[n] for n in self.names]); s_c = np.array([sc[Path(cen[n]).name] for n in self.names])
        self.resid = (E - (s_p - s_c)) / 2 - base; self.E = E; self.ok = E > 2e-5; self.sd = 0.00041; self.allrows = allrows
        self.n1 = len(self.rows["ssp126"])

    def direction(self, L126: np.ndarray, L585: np.ndarray) -> np.ndarray:
        out = []
        for s_, Lx in (("ssp126", L126), ("ssp585", L585)):
            r = self.rows[s_]
            out.append(self.R[r, 1] * np.exp(Lx[self.site[r]]) / (DIV[1] * M) - self.C[r, 1])
        return np.concatenate(out)

    def chi2(self, d: np.ndarray) -> tuple[float, float, float, float]:
        x = self.A @ d; xo, ro = x[self.ok], self.resid[self.ok]
        kap = float(xo @ ro / max(xo @ xo, 1e-30))
        return float(np.sum(((ro - xo) / self.sd) ** 2)), kap, float(np.sum(((ro - kap * xo) / self.sd) ** 2)), float(np.mean(d ** 2) * len(d) / self.C.size)

    def sim(self, W126, W585, c=1.0, rg=(1.0, 1.0), aq=(1.0, 1.0), cap=None):
        out = []
        for W, r_, a_ in ((W126, rg[0], aq[0]), (W585, rg[1], aq[1])):
            a = self.a0.copy()
            for kk in range(W.shape[1]):
                rate = np.minimum(bd.PRM["cap"] if cap is None else cap, bd.PRM["c"] * c * a * W[:, kk])
                a = a * np.exp(-rate + em.regrowth(np.log(np.maximum(self.a0 * a_, 1e-6) / np.maximum(a, 1e-6)), r_))
            out.append(np.log(np.maximum(a, 0.02) / np.maximum(self.a0, 0.02)))
        return out


def main() -> None:
    S = Scan()
    chi0 = float(np.sum((S.resid[S.ok] / S.sd) ** 2))
    print(f"{int(S.ok.sum())} probes; fire sites {int(S.sel.sum())}; chi2 under truth = C: {chi0:.1f}")
    res = []
    idx = np.arange(40)
    for c, rg, aq in itertools.product((0.3, 1.0, 3.0), ((1.0, 1.0), (1.5, 2.0), (2.5, 3.0)), ((1.0, 1.0), (1.10, 1.20))):
        for nm, mp in (("k->k", idx), ("reversed", idx[::-1]), ("shift10", (idx + 10) % 40), ("shift20", (idx + 20) % 40), ("shift30", (idx + 30) % 40), ("first20x2", np.r_[idx[:20], idx[:20]]), ("last20x2", np.r_[idx[20:], idx[20:]])):
            L1, L5 = S.sim(S.Wm[:, mp], S.Wm[:, mp], c, rg, aq); d = S.direction(L1, L5); res.append((f"mer {nm:9s} c x{c:<4} regrow {rg} aeq {aq}", *S.chi2(d), L1[S.g["cl"]].mean()))
        for nm, W in S.Wc.items():
            Wk = np.repeat(W[:, None], 40, axis=1); L1, L5 = S.sim(Wk, Wk, c, rg, aq); d = S.direction(L1, L5); res.append((f"const {nm:16s} c x{c:<4} regrow {rg} aeq {aq}", *S.chi2(d), L1[S.g["cl"]].mean()))
    for c in (1.0, 0.1, 0.01, 0.001, 0.0001):
        for rg, aq in (((1.0, 1.0), (1.0, 1.0)), ((1.5, 2.0), (1.10, 1.20))):
            L1, L5 = S.sim(S.Wf[0], S.Wf[1], c, rg, aq); d = S.direction(L1, L5); res.append((f"future calibrated c x{c:<6} regrow {rg} aeq {aq}", *S.chi2(d), L1[S.g["cl"]].mean()))
            L1, L5 = S.sim(S.Wr[0], S.Wr[1], c, rg, aq); d = S.direction(L1, L5); res.append((f"future as released c x{c:<6} regrow {rg} aeq {aq}", *S.chi2(d), L1[S.g["cl"]].mean()))
    # no fire at all: growth only
    z = np.zeros(len(S.a0)); d = S.direction(z + 0.06, z + 0.11); res.append(("no fire (growth +0.06 / +0.11)", *S.chi2(d), 0.06))
    res.sort(key=lambda r: r[1])
    print("best 25 by chi2 (truth = C + h):   chi2 | kappa | chi2 at kappa | E_all | cluster mean log ratio")
    for r in res[:25]: print(f"  {r[0]:62s} {r[1]:8.1f}  {r[2]:+.2f}  {r[3]:7.1f}  {r[4]:.5f}  {r[5]:+.3f}")
    print("selected others:")
    for r in res:
        if r[0].startswith(("no fire", "future")) or ("k->k" in r[0] and "c x1.0" in r[0]): print(f"  {r[0]:62s} {r[1]:8.1f}  {r[2]:+.2f}  {r[3]:7.1f}  {r[4]:.5f}  {r[5]:+.3f}")


if __name__ == "__main__":
    main()

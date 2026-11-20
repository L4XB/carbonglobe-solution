#!/usr/bin/env python
"""Exact joint readout of ANY number of single-factor probes that share one scored reference file (several nights of probes on the same base).

Every probe file F_j = reference + d_j with public score S_j (rounded to 0.001 = 370.26 squared units):
    S_j - S_ref = (1 + eps_j z_j) (A0_j - A1_j x_j)
  level / contrast probe (values of a block x (1 - s), or x (1 +- s) by scenario):  A0 = E (the probe's energy over all rows), A1 = 2 E / s,
      x = delta = level excess of the reference on the block (truth = reference x (1 - delta));
  kappa probe (builder parts at step t, in the reference at steps s_i, full-step energies e_i rescaled to the realised energy E):
      A0 = sum e_i (t - s_i)(t + s_i), A1 = 2 sum e_i (t - s_i), x = common kappa of the parts.
  eps_j = sampling error of the block's PUBLIC energy (random half of the sites), z_j ~ N(0, 1); prior x_j ~ N(mean, sd).
The reference's true score is integrated on a grid over its rounding interval; given it, the probes are independent and every interval
integral is closed form (truncated normal) - no Monte Carlo, no rejection, any number of probes. The rounded reference score is a COMMON error
of all readings (an offset of 107 units moves every delta by 107 / (2 s S2)); the printed correlations show it.

    python scripts/readout_joint.py --design work/logs/cand50_design.json:work/logs/day50_probes.json [--design ...:...]
        [--scores cand50_A.csv=21,cand50_A1.csv=29,...]      (default: results/board_scores.json)
        [--ref-prior 21.5,0.45]                              (Gaussian prior of the reference's true score in quanta; default flat on its interval)
        [--scan 0.15,0.3,0.6,1,1.7]                          (empirical Bayes: evidence-weighted average over common scales of all prior sds)
        [--transfer pos_h=0.87,...]                          (private-half shrink tau^2 / (tau^2 + sigma_h^2) per block, printed as "private step";
                                                              the last line gives the multiples m of the probe directions for scripts/second_stage.sh)
        [--only name,name]                                   (read a family of probes alone, with its own prior-scale scan)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = 370.26


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="append", required=True, help="design.json:probes.json (repeat for several nights; same reference file)")
    ap.add_argument("--scores", default=""); ap.add_argument("--ref-prior", default=""); ap.add_argument("--transfer", default="")
    ap.add_argument("--grid", type=int, default=801)
    ap.add_argument("--only", default="", help="comma list of factor names or probe files: read only these probes (a FAMILY of probes gets its own prior-scale scan; "
                    "one real signal among many nulls is otherwise shrunk by the common scale)")
    ap.add_argument("--public", action="store_true", help="print the steps that are best for the PUBLIC half (ignore sigma_h); default: private-half steps")
    ap.add_argument("--scan", default="", help="prior scales, e.g. 0.15,0.3,0.6,1,1.7: the scores decide how wide the block errors really are "
                    "(evidence-weighted average over the scales; guards against acting on noise when nothing is wrong)")
    args = ap.parse_args()
    over = {k: float(v) for k, v in (it.split("=") for it in args.scores.split(",") if it)}
    bs = json.load(open(ROOT / "results/board_scores.json"))
    score = lambda f: over[f] if f in over else float(bs[f]) * 1000.0
    probes = []; ref = None
    for item in args.design:
        dp, pp = item.split(":"); des = json.load(open(ROOT / dp)); pr = json.load(open(ROOT / pp))
        files = des["files"]; inc = np.array(des["include"], float); ia = int(np.where(inc.sum(1) == 0)[0][0])
        assert ref in (None, files[ia]), f"designs with different reference files: {ref} / {files[ia]}"
        ref = files[ia]
        for j, f in enumerate(des["factors"]):
            rows = np.where(inc[:, j] == 1)[0]; assert len(rows) == 1 and inc[rows[0]].sum() == 1, f"{f}: single-factor files only"
            p = pr[f]; E = float(des["energies"][f])
            if p["type"] in ("level", "contrast"):
                A0, A1 = E, 2.0 * E / float(p["s"]); S2 = E / float(p["s"]) ** 2
            else:
                e = np.array([v[0] for v in p["parts"].values()], float); s = np.array([v[1] for v in p["parts"].values()], float); t = float(p["t"])
                e = e * E / float(np.sum(e * (t - s) ** 2)); A0 = float(np.sum(e * (t - s) * (t + s))); A1 = 2.0 * float(np.sum(e * (t - s))); S2 = float(np.sum(e))
            probes.append({"name": f, "file": files[rows[0]], "type": p["type"], "A0": A0, "A1": A1, "S2": S2, "E": E, "mu": float(p["prior"][0]), "tau": float(p["prior"][1]),
                           "eps": float(p.get("eps", 0.0)), "block": p.get("block", f), "sc": score(files[rows[0]]), "s": float(p.get("s", 0.0)),
                           "sig_h": float(p.get("sigma_h", 0.0))})
    if args.only:
        keep = set(args.only.split(",")); probes = [p for p in probes if p["name"] in keep or p["file"] in keep]
        assert probes, f"--only {args.only}: no such probe"
    sref = score(ref)
    a = (sref - 0.5 + (np.arange(args.grid) + 0.5) / args.grid) * Q                      # grid of the reference's true score (squared units)
    if args.ref_prior:
        m_, s_ = (float(x) for x in args.ref_prior.split(",")); lw0 = -0.5 * ((a / Q - m_) / s_) ** 2
    else:
        lw0 = np.zeros_like(a)
    zn, zw = np.polynomial.hermite_e.hermegauss(9); zw = zw / zw.sum()                   # nodes of z ~ N(0, 1)

    def solve(scale: float) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        """Posterior for prior sds x scale: log evidence, means, second-moment matrix of the readings, first / second moment of the reference score."""
        lw = lw0.copy(); M1 = []; M2 = []
        for p in probes:
            tau = p["tau"] * scale
            lo = (p["sc"] - 0.5) * Q - a; hi = (p["sc"] + 0.5) * Q - a                   # interval of S_j - S_ref, per grid point
            L = np.zeros_like(a); m1 = np.zeros_like(a); m2 = np.zeros_like(a)
            for z, w in zip(zn, zw):
                g = 1.0 + p["eps"] * z
                x_lo = (p["A0"] - hi / g) / p["A1"]; x_hi = (p["A0"] - lo / g) / p["A1"] # A1 > 0: x falls as the score difference rises
                if p["A1"] < 0:
                    x_lo, x_hi = x_hi, x_lo
                al = (x_lo - p["mu"]) / tau; be = (x_hi - p["mu"]) / tau
                m0 = _Phi(be) - _Phi(al); pa, pb = _phi(al), _phi(be)
                L += w * m0
                m1 += w * (p["mu"] * m0 + tau * (pa - pb))
                m2 += w * (p["mu"] ** 2 * m0 + 2 * p["mu"] * tau * (pa - pb) + tau ** 2 * (m0 + al * pa - be * pb))
            L = np.maximum(L, 1e-300); lw = lw + np.log(L); M1.append(m1 / L); M2.append(m2 / L)
        lmax = float(lw.max()); W = np.exp(lw - lmax); logev = lmax + float(np.log(W.mean())); W /= W.sum()
        M1 = np.array(M1); M2 = np.array(M2)
        sec = (M1 * W) @ M1.T; np.fill_diagonal(sec, M2 @ W)                             # E[x_i x_j]: independent given the reference score
        return logev, M1 @ W, sec, float(a @ W), float((a * a) @ W)

    scales = [float(x) for x in args.scan.split(",") if x] or [1.0]
    sol = [solve(sc_) for sc_ in scales]
    lev_ = np.array([r[0] for r in sol]); wsc = np.exp(lev_ - lev_.max()); wsc /= wsc.sum()   # flat hyperprior over the listed prior scales
    mean = sum(w * r[1] for w, r in zip(wsc, sol)); sec = sum(w * r[2] for w, r in zip(wsc, sol))
    # PRIVATE-half estimate: a reading is the block's public-half error = structural part + the public sites' own noise (sd sigma_h, field
    # "sigma_h" of the probes file, from scripts/diag/block_info.py --boards). At prior scale x the structural share of the prior variance is
    # max(1 - sigma_h^2 / (x tau)^2, 0); only that share of the posterior mean carries over to the private half.
    tau_ = np.array([p["tau"] for p in probes]); sh_ = np.array([p["sig_h"] for p in probes])
    share = [np.clip(1.0 - (sh_ / (sc_ * tau_)) ** 2, 0.0, 1.0) for sc_ in scales]
    pmean = sum(w * r[1] * sh for w, r, sh in zip(wsc, sol, share))
    cov = sec - np.outer(mean, mean); var = np.diag(cov).copy(); sd = np.sqrt(np.maximum(var, 0.0))
    am = float(sum(w * r[3] for w, r in zip(wsc, sol))); asd = float(np.sqrt(max(sum(w * r[4] for w, r in zip(wsc, sol)) - am * am, 0.0)))
    evid = float(np.exp(lev_.max()))
    if len(scales) > 1:
        print("prior-scale scan (all prior sds x scale; weights = evidence of the scores): " + ", ".join(f"x{sc_:g}: {w:.2f}" for sc_, w in zip(scales, wsc))
              + f" -> effective scale {float(np.sum(wsc * np.array(scales))):.2f}")
    tr = {k: float(v) for k, v in (it.split("=") for it in args.transfer.split(",") if it)}
    print(f"reference {ref}: score {sref:.0f} (x 0.001), true score {am / Q:.2f} +- {asd / Q:.2f}; {len(probes)} probes; relative evidence of the priors {evid:.3g}")
    tot = 0.0; lev = []; stp = []
    for i, p in enumerate(probes):
        g = p["S2"] * mean[i] ** 2
        if p["type"] in ("level", "contrast"):
            unit = "LEVEL excess" if p["type"] == "level" else "CONTRAST step"; t_ = tr.get(p["block"], tr.get(p["name"], 1.0))
            pm = t_ * (pmean[i] if not args.public else mean[i])        # what goes into the file: private-half estimate unless --public
            line = (f"   {p['name']:30s} {p['file']:16s} score {p['sc']:.0f}  {unit} {100 * mean[i]:+.2f} % +- {100 * sd[i]:.2f} % (prior {100 * p['mu']:+.1f} +- {100 * p['tau']:.1f})"
                    f"  factor {1 - mean[i]:.4f}  gain {g:.0f} u")
            if abs(pm - mean[i]) > 1e-9:
                line += f" | private step {100 * pm:+.2f} % (sigma_h {100 * p['sig_h']:.2f} %)"
            lev.append(f"{p['block']}={1 - pm:.4f}"); stp.append(f"{p['file']}={pm / p['s']:+.4f}")
            line += f" | m {pm / p['s']:+.4f}"
        else:
            # best multiple of the probe direction for a common kappa of its parts: m = sum e (t - s)(kappa - s) / sum e (t - s)^2 (single part: (kappa - s) / (t - s))
            t_ = tr.get(p["block"], tr.get(p["name"], 1.0)); m_k = t_ * (mean[i] * p["A1"] / 2.0 - (p["A0"] - p["E"]) / 2.0) / p["E"]; g = p["E"] * m_k ** 2   # no sigma_h model for kappa probes
            line = (f"   {p['name']:30s} {p['file']:16s} score {p['sc']:.0f}  common KAPPA {mean[i]:+.3f} +- {sd[i]:.3f} (prior {p['mu']} +- {p['tau']})"
                    f"  gain {g:.0f} u | m {m_k:+.4f}")
            stp.append(f"{p['file']}={m_k:+.4f}")
        print(line); tot += g
    if len(probes) > 1:
        c = cov / np.outer(sd, sd); iu = np.triu_indices(len(probes), 1)
        print(f"   correlation between readings (common reference score): mean {float(np.mean(c[iu])):+.2f}, max {float(np.max(c[iu])):+.2f}")
    print(f"   expected gain of all corrections {tot:.0f} units ({tot / Q:.2f} x 0.001)")
    print("   factors: " + ",".join(lev))
    print("   steps for scripts/second_stage.sh (m = delta x transfer / s, multiples of the probe directions): " + ",".join(stp))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""End-to-end test of the two-stage nights on synthetic boards: probes of several nights on ONE reference file -> rounded public scores ->
scripts/readout_joint.py -> second-stage candidate C = A + sum m_j (P_j - A) -> TRUE public and private scores of A and C (known only here).

    python scripts/sim/synth_two_stage.py --design work/logs/cand50_design.json:work/logs/day50_probes.json
        [--design work/logs/cand51_design.json:work/logs/day51_probes.json] --boards work/sim/synth1,work/sim/synth2 [--draws 5] [--inject 1.0]
        [--transfer block=shrink,...]

Per board and draw a STRUCTURAL level / contrast error delta_j ~ N(0, (inject x prior sd_j)^2) is put into the truth on every probe's block
(truth <- truth + (delta_j / s_j)(P_j - A), the same on both halves); --inject 0 = the null world (truth = board truth).
Printed per run: scores, readings against the injected values, gain of C over A on the public and on the private half (squared units, all-rows
scale), and the oracle gain (all injected errors removed exactly).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

U = 370260.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="append", required=True); ap.add_argument("--boards", required=True)
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--inject", type=float, default=1.0); ap.add_argument("--transfer", default="")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--verbose", action="store_true"); ap.add_argument("--scan", default=""); ap.add_argument("--extra", default="", help="further arguments of readout_joint.py, e.g. --public")
    ap.add_argument("--families", default="", help="probe families a+b;c+d (factor names): one readout per family (readout_joint.py --only), probes not named form a last family")
    ap.add_argument("--fixed", default="", help="name=x,...: injected block error of these probes (level / contrast: delta, kappa probe: kappa) instead of a random draw")
    ap.add_argument("--gram", action="store_true", help="also score the candidate with DECORRELATED steps m' = G^-1 diag(G) m (G = Gram matrix of the probe directions): "
                    "a probe reads the projection of the error on its direction, so overlapping probes share a reading")
    args = ap.parse_args()
    probes = []; ref = None
    for item in args.design:
        dp, pp = item.split(":"); des = json.load(open(ROOT / dp)); pr = json.load(open(ROOT / pp))
        files = des["files"]; inc = np.array(des["include"]); ia = int(np.where(inc.sum(1) == 0)[0][0]); ref = files[ia]
        for j, f in enumerate(des["factors"]):
            i = int(np.where(inc[:, j] == 1)[0][0]); q = pr[f]; E = float(des["energies"][f])
            if q["type"] in ("level", "contrast"):                  # multiple of the probe direction for a block error x: m = x / s
                probes.append({"name": f, "file": files[i], "s": float(q["s"]), "tau": float(q["prior"][1]), "mu": 0.0, "a": 1.0 / float(q["s"]), "b": 0.0})
            else:                                                   # kappa probe (as scripts/readout_joint.py): m = (x A1 / 2 - (A0 - E) / 2) / E, x = common kappa
                e = np.array([v[0] for v in q["parts"].values()], float); st = np.array([v[1] for v in q["parts"].values()], float); t = float(q["t"])
                e = e * E / float(np.sum(e * (t - st) ** 2)); A0 = float(np.sum(e * (t - st) * (t + st))); A1 = 2.0 * float(np.sum(e * (t - st)))
                probes.append({"name": f, "file": files[i], "s": 1.0, "tau": float(q["prior"][1]), "mu": float(np.sum(e * (t - st) * st) / np.sum(e * (t - st))),
                               "a": A1 / (2.0 * E), "b": -(A0 - E) / (2.0 * E)})      # mu = the kappa at which the reference is exact (m = 0)
    A = shift.read_predictions(ROOT / "results" / ref).astype(np.float64)
    D = [shift.read_predictions(ROOT / "results" / p["file"]).astype(np.float64) - A for p in probes]
    G = np.array([[float(np.sum(a * b)) for b in D] for a in D]); dG = np.diag(G).copy(); res_g = []
    rng = np.random.default_rng(args.seed); res = []
    for board in args.boards.split(","):
        T0 = np.load(ROOT / (board + "_truth.npy")).astype(np.float64); pub = np.load(ROOT / (board + "_public.npy"))
        if pub.shape[0] != A.shape[0]:                                  # site mask -> row mask
            pub = pub[shift.template_keys().site.to_numpy()]
        npub = pub.sum() * 2; nprv = (~pub).sum() * 2
        for k in range(args.draws):
            delta = np.array([p["mu"] for p in probes]) + rng.normal(0.0, 1.0, len(probes)) * np.array([p["tau"] for p in probes]) * args.inject
            for it in [x for x in args.fixed.split(",") if x]:
                nm, v = it.split("="); delta[[p["name"] for p in probes].index(nm)] = float(v)
            mtrue = np.array([p["a"] * dl + p["b"] for dl, p in zip(delta, probes)])        # true multiples of the probe directions
            T = T0 + sum(mt * d for mt, d in zip(mtrue, D))
            mse = lambda P, m: float(np.sum((P[m] - T[m]) ** 2)) / (m.sum() * 2)
            sA = round(mse(A, pub), 3) * 1000; sc = [round(mse(A + d, pub), 3) * 1000 for d in D]
            cmd = [sys.executable, "scripts/readout_joint.py"] + sum((["--design", it] for it in args.design), [])
            cmd += ["--scores", ",".join([f"{ref}={sA:.0f}"] + [f"{p['file']}={s:.0f}" for p, s in zip(probes, sc)])]
            if args.transfer:
                cmd += ["--transfer", args.transfer]
            if args.scan:
                cmd += ["--scan", args.scan]
            cmd += args.extra.split()
            fams = [f.split("+") for f in args.families.split(";") if f]
            if fams:
                named = {x for f in fams for x in f}; rest = [p["name"] for p in probes if p["name"] not in named]
                fams = fams + ([rest] if rest else [])
            m = {}; failed = False
            for fam in (fams or [None]):
                out = subprocess.run(cmd + (["--only", ",".join(fam)] if fam else []), cwd=ROOT, capture_output=True, text=True).stdout
                line = [ln for ln in out.splitlines() if "steps for scripts/second_stage.sh" in ln]
                if not line:
                    print(f"{board} draw {k}: readout failed\n{out[-400:]}"); failed = True; break
                m.update({a.split("=")[0]: float(a.split("=")[1]) for a in line[0].split(": ")[-1].split(",")})
            if failed:
                continue
            C = A + sum(m[p["file"]] * d for p, d in zip(probes, D))
            O = A + sum(mt * d for mt, d in zip(mtrue, D))                            # oracle: every injected error removed
            g = {h: (mse(A, mk) - mse(C, mk)) * U for h, mk in (("pub", pub), ("prv", ~pub))}
            go = {h: (mse(A, mk) - mse(O, mk)) * U for h, mk in (("pub", pub), ("prv", ~pub))}
            read = [m[p["file"]] * p["s"] for p in probes]
            res.append((g["pub"], g["prv"], go["pub"], go["prv"]))
            if args.gram:
                mv = np.array([m[p["file"]] for p in probes]); mg = np.linalg.solve(G, dG * mv)
                Cg = A + sum(x * d for x, d in zip(mg, D))
                res_g.append(tuple((mse(A, mk) - mse(Cg, mk)) * U for mk in (pub, ~pub)))
            print(f"{Path(board).name} draw {k}: A {sA:.0f}, probes {' '.join(f'{s:.0f}' for s in sc)} | gain of C public {g['pub']:+6.0f} private {g['prv']:+6.0f} | oracle {go['pub']:+6.0f} / {go['prv']:+6.0f}"
                  f" | true A public {mse(A, pub) * 1000:.2f} private {mse(A, ~pub) * 1000:.2f}")
            if args.verbose:
                print("      injected m x 100: " + " ".join(f"{100 * x:+6.2f}" for x in mtrue)); print("      applied  m x 100: " + " ".join(f"{100 * m[p['file']]:+6.2f}" for p in probes))
    if res_g:
        rg = np.array(res_g)
        print(f"{len(rg)} runs with decorrelated steps: mean gain public {rg[:, 0].mean():+.0f} (sd {rg[:, 0].std():.0f}), private {rg[:, 1].mean():+.0f} (sd {rg[:, 1].std():.0f}); "
              f"runs with a private loss: {int((rg[:, 1] < 0).sum())}")
    if res:
        r = np.array(res)
        print(f"{len(r)} runs: mean gain of C public {r[:, 0].mean():+.0f} (sd {r[:, 0].std():.0f}), private {r[:, 1].mean():+.0f} (sd {r[:, 1].std():.0f}); "
              f"oracle {r[:, 2].mean():+.0f} / {r[:, 3].mean():+.0f}; runs with a private loss: {int((r[:, 1] < 0).sum())}")


if __name__ == "__main__":
    main()

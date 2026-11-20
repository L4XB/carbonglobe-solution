#!/usr/bin/env python
"""Next day's weighing design from the previous day's spec and its scores (generic successor of make_spec_next.py, which reads the 11-12 ladder).

    python scripts/make_spec_step.py --prev work/logs/day50_spec.json --scores 23,24,23,23,24 (x 0.001, in the order of the spec's names)
        --day 51 --factors <new factors from the library of make_spec_next.py> [--extra 100,...] [--base 23.54,0.17]
        [--prior F=0.9:0.25,cl_full=0.65:0.35] [--steps F=1.0,...] [--keep-min 0.15] [--prefix cand51] [--out work/logs/day51_spec.json]

Model in squared units (quantum Q = 370.26): score(file) = b - sum_i include[file][i] x (2 kappa_i - 1) x e_i, e_i = single-factor energy from
work/logs/<prefix>_design.json, every score rounded to 0.001. Priors: kappa_i ~ N(mean, sd) (default 0.65 +- 0.35), b ~ N(--base) in quanta (default:
wide around the scores). Posterior by Monte Carlo over the exact rounding intervals. The new base = previous base + every old factor at its
posterior-mean step (mix entries get w = kappa, post steps get their step scaled, plain recipe deltas are kept if kappa > 0.5; steps below --keep-min
are dropped, steps are clipped to [0, 1.5]); the new factors are weighed on top (all-in file first, leave-one-out files, --extra patterns).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_spec_next as msn  # noqa: E402

Q = 370.26


def posterior(include: np.ndarray, en: np.ndarray, scores: np.ndarray, pk: np.ndarray, pb: tuple[float, float], n: int = 600000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Samples of (kappas, base score in squared units) that reproduce the rounded scores; priors are widened if too few samples survive."""
    rng = np.random.default_rng(seed); widen = 1.0
    for it in range(10):
        m = n * (4 if it >= 3 else 1)
        k = rng.normal(pk[:, 0], pk[:, 1] * widen, size=(m, len(en))); b = rng.normal(pb[0] * Q, pb[1] * Q * widen, size=m)
        g = (2 * k - 1) * en
        s = b[:, None] - g @ include.T
        ok = (np.round(s / Q) == scores[None, :]).all(axis=1)
        if ok.sum() >= (300 if it < 6 else 60):
            return k[ok], b[ok], widen
        widen *= 1.6
    raise SystemExit("no sample reproduces the scores - check the scores, the design and the priors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--scores", required=True, help="scores of the previous files in the order of the spec's names, x 0.001")
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--factors", default="", help="new factors (library of make_spec_next.py), comma list")
    ap.add_argument("--extra", default="")
    ap.add_argument("--rows", default="", help="full control of the layout: inclusion patterns over the NEW factors, first = the best-guess file (e.g. 110,010,100,111,000); overrides loo / --extra")
    ap.add_argument("--base", default="", help="prior of the previous BASE score 'mean,sd' in quanta (x 0.001)")
    ap.add_argument("--prior", default="", help="kappa priors name=mean:sd")
    ap.add_argument("--steps", default="", help="override posterior-mean steps name=value")
    ap.add_argument("--keep-min", type=float, default=0.15)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true", help="print the posterior only")
    args = ap.parse_args()
    prev = json.load(open(ROOT / args.prev)); des = json.load(open(ROOT / f"work/logs/{prev['prefix']}_design.json"))
    names = [f["name"] for f in prev["factors"]]; assert des["factors"] == names, (des["factors"], names)
    include = np.array(prev["include"], float); en = np.array([float(des["energies"][nm]) for nm in names])
    scores = np.array([float(x) for x in args.scores.split(",")]); assert len(scores) == len(include)
    pk = np.array([[0.65, 0.35]] * len(names))
    for item in [x for x in args.prior.split(",") if x]:
        nm, v = item.split("="); m, s = v.split(":"); pk[names.index(nm)] = (float(m), float(s))
    pb = tuple(float(x) for x in args.base.split(",")) if args.base else (float(scores.max()) + 0.5, 2.5)
    k, b, widen = posterior(include, en, scores, pk, pb)
    g = (2 * k - 1) * en
    print(f"scores (x 0.001) {dict(zip(prev['names'], scores.tolist()))}; accepted samples {len(k)}" + (f" (priors widened x {widen:.1f})" if widen > 1 else ""))
    print(f"   base of the previous design: {b.mean() / Q:.2f} +- {b.std() / Q:.2f} (x 0.001)")
    step = {}
    for i, nm in enumerate(names):
        step[nm] = float(np.clip(k[:, i].mean(), 0.0, 1.5))
        print(f"   {nm:10s} energy {en[i]:5.0f}: gain {g[:, i].mean():+5.0f} +- {g[:, i].std():3.0f}, kappa {k[:, i].mean():+.2f} +- {k[:, i].std():.2f} (prior {pk[i, 0]} +- {pk[i, 1]}), P(gain > 0) {np.mean(g[:, i] > 0):.2f}")
    for item in [x for x in args.steps.split(",") if x]:
        nm, v = item.split("="); step[nm] = float(v); print(f"   override: step {nm} = {step[nm]}")
    files = b[:, None] - g @ include.T
    newb = b - ((2 * k * np.array([step[nm] for nm in names]) - np.array([step[nm] for nm in names]) ** 2) * en).sum(axis=1)
    print("   true scores (x 0.001): " + ", ".join(f"{nm} {files[:, j].mean() / Q:.2f} +- {files[:, j].std() / Q:.2f}" for j, nm in enumerate(prev["names"])) + f" | new base {newb.mean() / Q:.2f} +- {newb.std() / Q:.2f}")
    if args.dry:
        return
    base = json.loads(json.dumps(prev["base"])); base.setdefault("mix", []); kept = []
    for f in prev["factors"]:
        nm = f["name"]; w = step[nm]; d = f["delta"]
        if w < args.keep_min:
            continue
        kept.append(f"{nm} {w:.2f}")
        for m in d.get("mix", []):
            base["mix"].append({**m, "w": round(float(m.get("w", 1.0)) * w, 4)})
        for p in d.get("post", []):
            p = dict(p)
            for key in ("h_step", "a_step"):
                if key in p:
                    p[key] = round(min(float(p[key]) * w, 1.0), 4)
            if "factor" in p:                                    # level step: factor^w
                p["factor"] = round(float(p["factor"]) ** w, 5)
            base.setdefault("post", []).append(p)
        rest = {kk: v for kk, v in d.items() if kk not in ("mix", "post")}
        if rest and w > 0.5:
            base = msn.merge(base, rest) if hasattr(msn, "merge") else {**base, **rest}
    pre = args.prefix or f"cand{args.day}"; out = ROOT / (args.out or f"work/logs/day{args.day}_spec.json")
    ffr = next((p["ff_rows"] for p in base.get("post", []) if "ff_rows" in p), f"work/ff/ff_rows_day{args.day}.npz")
    names_f = [x for x in args.factors.split(",") if x]; factors = []
    for nm in names_f:
        kind, pay, txt = msn.LIB[nm]; pay = json.loads(json.dumps(pay).replace("@FF", ffr))
        delta = {"mix": [{"id": nm, "w": 1.0, "delta": pay}]} if kind == "mix" else ({"post": [pay]} if kind == "post" else pay)
        factors.append({"name": nm, "delta": delta, "text": txt})
    kf = len(factors)
    include2 = [[1] * kf] + [[0 if j == i else 1 for j in range(kf)] for i in range(kf)]; names2 = ["A"] + [f"A{i + 1}" for i in range(kf)]
    for pat in [x for x in args.extra.split(",") if x]:
        assert len(pat) == kf and set(pat) <= {"0", "1"}; include2.append([int(ch) for ch in pat]); names2.append(f"A{len(names2)}")
    if args.rows:
        pats = [x for x in args.rows.split(",") if x]
        assert all(len(p) == kf and set(p) <= {"0", "1"} for p in pats), f"--rows patterns need {kf} digits"
        include2 = [[int(ch) for ch in p] for p in pats]; names2 = ["A"] + [f"A{i}" for i in range(1, len(pats))]
    spec = {"prefix": pre, "names": names2, "base": base, "factors": factors, "include": include2,
            "base_text": f"base = {prev['prefix']} base + its factors at their posterior-mean steps ({', '.join(kept) or 'none kept'}; scores {args.scores})",
            "min_energy": prev.get("min_energy", 20.0), "pre": prev.get("pre", []),
            "note": f"written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by scripts/make_spec_step.py --prev {args.prev} --scores {args.scores} --factors {args.factors}"
                    + (f" --extra {args.extra}" if args.extra else "") + (f" --steps {args.steps}" if args.steps else "") + (f" --base {args.base}" if args.base else ""),
            "expected_base": [round(float(newb.mean() / Q), 3), round(float(newb.std() / Q), 3)]}
    if out.exists():
        shutil.copy(out, out.with_name(out.name + f".bak_{time.strftime('%H%M', time.gmtime())}"))
    out.write_text(json.dumps(spec, indent=1))
    print(f"wrote {out.relative_to(ROOT)}: prefix {pre}, files {names2}, new factors {names_f}, kept {kept}")


if __name__ == "__main__":
    main()

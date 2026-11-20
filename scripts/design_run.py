#!/usr/bin/env python
"""Build the files of a weighing design from a JSON spec, check them and write the manifest + design file for the submit / readout scripts.

A RECIPE is a dict: {"amp": {"pos": {"a": 1.4, "q0": 0.05, "qmax": 9, "qmin": -9}, "b": {"a": 1.0, "q0": 0.05}}, "steps": {label: step}, "h_strata": null | "boreal,cool", "h_strata_pos": "none" | "all",
"b": 0..1 (share of rows v6 + memory decode: 0.5 = mean of the builds with and without them), "extra": {name: [more builder arguments]},
"post": [{"id": "ffpos", "tool": "apply_ff", "ff_rows": "work/ff/x.npz", "where": "pos", "h_step": 0.85, "h_old_max": -0.15, "a_step": 0.85,
"a_old_max": -0.3, ...}] (every other key becomes --key value of scripts/<tool>.py), "mix": [{"id": "pivot", "w": 0.6, "delta": {...}, "on": {...}}] (partial
application of a delta in prediction space; optional "on" = recipe delta of the recipe on which the deviation is evaluated, e.g. the young-row steps
as their PLAIN-target delta on top of an amplified file: tested 11-11 against the ladder files E / Y / LY of 11-12, bit-identical), "parts": builder components (default low,ed,edh,ornl,midlow; add bareh for the bare-row height floor)}. A FACTOR is a recipe delta: "steps" are merged, "amp" and "extra" are merged per class / name (null removes one), a "post" entry replaces the entry with the same id ("drop": true removes
it), every other key replaces the base value.

Spec: {"prefix": "cand50", "names": ["C", "D1", ...], "base": recipe, "factors": [{"name": "Fc", "delta": {...}, "text": "..."}],
"include": [[0/1 per factor] per file], "base_text": "...", "base_prior": [mean, sd] (quanta, optional), "pre": [shell commands run first]}.

    python scripts/design_run.py work/logs/day50_spec.json [--check]      (--check also builds every single factor and reports non-additivity)
Writes work/logs/<prefix>_manifest.txt (file <tab> description), work/logs/<prefix>_design.json, work/logs/<prefix>_sanity.txt ('ok' / 'bad').
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

PYT = sys.executable
PARTS = "low,ed,edh,ornl,midlow"          # recipe key "parts" overrides (e.g. "low,ed,edh,ornl,midlow,bareh")
FIX = ["--h-level", "1.0", "--gate-h", "0.12", "--gate-h2", "0.15", "--h-site-median", "3"]
BF = ["--ornl-tag", "v6", "--decode", "memory"]
_built: dict[str, Path] = {}


def merge(base: dict, delta: dict) -> dict:
    r = copy.deepcopy(base)
    for k, v in delta.items():
        if k == "steps":
            r.setdefault("steps", {}).update(v)
        elif k == "amp":                                   # per class: fields merged, null removes the class
            cur = r.setdefault("amp", {})
            for c, f in v.items():
                if f is None:
                    cur.pop(c, None)
                else:
                    cur[c] = {**cur.get(c, {}), **f}
        elif k == "extra":                                 # named groups of extra builder arguments, merged by name (null removes a group)
            cur = r.setdefault("extra", {})
            for c, f in v.items():
                if f is None:
                    cur.pop(c, None)
                else:
                    cur[c] = list(f)
        elif k == "mix":                                   # entries merged by id (null weight removes one)
            cur = {e["id"]: e for e in r.get("mix", [])}
            for e in v:
                if e.get("w") is None:
                    cur.pop(e["id"], None)
                else:
                    cur[e["id"]] = e
            r["mix"] = list(cur.values())
        elif k == "post":
            cur = {p["id"]: p for p in r.get("post", [])}
            for p in v:
                if p.get("drop"):
                    cur.pop(p["id"], None)
                else:
                    cur[p["id"]] = {**cur.get(p["id"], {}), **p}
            r["post"] = list(cur.values())
        else:
            r[k] = v
    return r


def builder_args(r: dict, v6: bool) -> list[str]:
    a = ["--parts", r.get("parts", PARTS)] + list(FIX) + ["--h-strata-pos", r.get("h_strata_pos", "none")]
    if r.get("amp"):                                       # {"pos": {"a": 1.4, "q0": 0.05, "qmax": 9, "qmin": -9}, "b": {...}} -> pos=1.4:0.05:9:-9,b=...
        a += ["--amp", ",".join(f"{c}={f['a']}:{f['q0']}:{f.get('qmax', 9)}:{f.get('qmin', -9)}" for c, f in r["amp"].items())]
    if r.get("h_strata"):
        a += ["--h-strata", r["h_strata"]]
    a += ["--steps", ",".join(f"{k}={v}" for k, v in r.get("steps", {}).items())]
    for grp in r.get("extra", {}).values():
        a += [str(x) for x in grp]
    return a + (BF if v6 else [])


def build(args: list[str]) -> Path:
    key = hashlib.sha1(" ".join(args).encode()).hexdigest()[:10]
    if key in _built:
        return _built[key]
    nm = f"td_{key}"; out = ROOT / "results" / f"{nm}.csv"
    if __import__("os").environ.get("DESIGN_CACHE") == "1" and out.exists() and out.stat().st_mtime > (ROOT / "scripts/build_candidate2.py").stat().st_mtime:
        _built[key] = out; return out                      # tests only: reuse a build of an earlier run (inputs of the builder must be unchanged)
    for attempt in range(3):
        log = ROOT / "work/logs" / f"build_{nm}.log"
        with open(log, "w") as fh:
            rc = subprocess.run([PYT, "scripts/build_candidate2.py", "--out", nm] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**__import__("os").environ, "OMP_NUM_THREADS": "4"}).returncode
        if rc == 0 and out.exists():
            _built[key] = out; return out
    raise RuntimeError(f"build failed three times: {' '.join(args)} (see work/logs/build_{nm}.log)")


def realise_plain(r: dict, out: Path) -> np.ndarray:
    """One recipe without 'mix': builder (rows v6 share b), then the post steps; written to `out` and returned."""
    b = float(r.get("b", 0.0))
    if b >= 1.0:
        P = shift.read_predictions(build(builder_args(r, True)))
    elif b <= 0.0:
        P = shift.read_predictions(build(builder_args(r, False)))
    else:
        P = b * shift.read_predictions(build(builder_args(r, True))) + (1.0 - b) * shift.read_predictions(build(builder_args(r, False)))
    shift.write_predictions(P, out)
    for p in r.get("post", []):
        tool = p.get("tool", "apply_ff")
        cmd = [PYT, f"scripts/{tool}.py", "--inp", str(out.relative_to(ROOT)), "--out", str(out.relative_to(ROOT))]
        if p.get("ff_rows"):
            cmd += ["--ff-rows", p["ff_rows"]]
        for k, v in p.items():
            if k in ("id", "tool", "ff_rows", "drop") or v is None:
                continue
            cmd += ["--" + k.replace("_", "-")] + ([] if v is True else [str(v)])
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"{tool} failed: {res.stderr[-400:]}")
        line = [x for x in res.stdout.splitlines() if "->" in x]
        print("   " + (line[-1][:300] if line else f"{tool}: no rows changed"))
    return shift.read_predictions(out).astype(float)


def realise(r: dict, out: Path) -> None:
    """A recipe with optional 'mix': [{"id": name, "w": weight, "delta": recipe delta}] = partial application in prediction space,
    P = P0 + sum_k w_k (P(base + delta_k) - P0)  (exact for deviations on disjoint cells; used for kappa-weighted steps of a measured deviation)."""
    mix = r.get("mix", []); r0 = {k: v for k, v in r.items() if k != "mix"}
    P0 = realise_plain(r0, out)
    if mix:
        P = P0.copy(); tmp = out.with_name(out.stem + "__mix.csv"); tmp2 = out.with_name(out.stem + "__mixbase.csv")
        for e in mix:
            # optional "on": recipe delta of the recipe ON WHICH the deviation is evaluated (e.g. the young-row steps as their plain-target
            # delta on top of an amplified file, as in the ladder of 11-12): P += w (P(base + on + delta) - P(base + on))
            rb = merge(r0, e["on"]) if e.get("on") else r0
            Pb = realise_plain(rb, tmp2) if e.get("on") else P0
            P += float(e["w"]) * (realise_plain(merge(rb, e["delta"]), tmp) - Pb)
        P[:, 1] = np.maximum(P[:, 1], -0.401412 / 3.39567); P[:, 0] = np.maximum(P[:, 0], 0.0)      # floors (weights above 1 extrapolate)
        shift.write_predictions(P, out)
        tmp.unlink(missing_ok=True); tmp2.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    spec = json.load(open(ROOT / args.spec)); pre = spec["prefix"]; names = spec["names"]; fac = spec["factors"]; inc = spec["include"]
    san = ROOT / "work/logs" / f"{pre}_sanity.txt"; san.write_text("bad")
    for c in spec.get("pre", []):
        rc = subprocess.run(c, shell=True, cwd=ROOT, executable="/bin/bash").returncode
        print(f"pre: {c[:120]} -> {rc}")
    A = shift.read_predictions(ROOT / "results/cand48_A.csv"); P = {}; ok_all = True
    for nm, row in zip(names, inc):
        r = spec["base"]
        for f, on in zip(fac, row):
            if on:
                r = merge(r, f["delta"])
        out = ROOT / "results" / f"{pre}_{nm}.csv"
        realise(r, out)
        P[nm] = shift.read_predictions(out); e = float(np.mean((P[nm] - A) ** 2))
        ok = P[nm].shape == (185130, 2) and bool(np.isfinite(P[nm]).all()) and P[nm][:, 1].min() >= -0.1183 and P[nm][:, 0].min() >= -0.0827 and P[nm].max() < 20 and 0.0005 <= e <= 0.02
        ok_all &= ok
        print(f"{pre}_{nm}: {'ok' if ok else 'BAD'}; against file A: height {np.sum((P[nm][:, 0] - A[:, 0]) ** 2):.0f} agb {np.sum((P[nm][:, 1] - A[:, 1]) ** 2):.0f} squared units (mean sq {e:.5f}); column means {P[nm][:, 0].mean():.4f} {P[nm][:, 1].mean():.4f}")
    # single factors against the base recipe: energies, cross terms, non-additivity of every file
    en = {}
    if args.check or spec.get("energies_from_singles", True):
        base = ROOT / "results" / f"{pre}__base.csv"; realise(spec["base"], base); B = shift.read_predictions(base); sd = {}
        for f in fac:
            o = ROOT / "results" / f"{pre}__single_{f['name']}.csv"; realise(merge(spec["base"], f["delta"]), o); sd[f["name"]] = shift.read_predictions(o) - B
            en[f["name"]] = float(np.sum(sd[f["name"]] ** 2))
        fn = [f["name"] for f in fac]
        print("   single-factor energies against the base recipe: " + ", ".join(f"{k} {v:.0f}" for k, v in en.items()))
        print("   cross terms 2 <d_i, d_j>: " + ", ".join(f"{fn[i]} x {fn[j]} {2 * float(np.sum(sd[fn[i]] * sd[fn[j]])):+.0f}" for i in range(len(fn)) for j in range(i + 1, len(fn))))
        for nm, row in zip(names, inc):
            add = sum((sd[f] for f, on in zip(fn, row) if on), np.zeros_like(B)); comp = P[nm] - B
            print(f"   {pre}_{nm}: energy against the base {np.sum(comp ** 2):.0f}, sum of its single factors {np.sum(add ** 2):.0f}, NON-ADDITIVE remainder {np.sum((comp - add) ** 2):.0f}")
        ok_all &= all(v >= spec.get("min_energy", 30.0) for v in en.values())
    with open(ROOT / "work/logs" / f"{pre}_manifest.txt", "w") as fh:
        for nm, row in zip(names, inc):
            txt = spec.get("base_text", "base") + "".join(f" | + {f.get('text', f['name'])}" for f, on in zip(fac, row) if on)
            fh.write(f"results/{pre}_{nm}.csv\t{nm}: {txt}\n")
    d = {"files": [f"{pre}_{nm}.csv" for nm in names], "factors": [f["name"] for f in fac], "include": inc, "steps": {f["name"]: 1.0 for f in fac}, "energies": en}
    if spec.get("base_prior"):
        d["base"] = spec["base_prior"]
    json.dump(d, open(ROOT / "work/logs" / f"{pre}_design.json", "w"), indent=1)
    san.write_text("ok" if ok_all else "bad")
    print("sanity:", "ok" if ok_all else "BAD")


if __name__ == "__main__":
    main()

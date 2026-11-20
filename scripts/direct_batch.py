#!/usr/bin/env python
"""DIRECT probe batch: every probe file = a SCORED base file with ONE block changed (level or scenario contrast), written by the same tools the
design runner uses for post steps (scripts/apply_level.py, scripts/apply_contrast.py). No builder run, no mix: a batch takes seconds, the
probes are exact single-factor differences from the base by construction, and the base's board score is the reference of the readout
(scripts/readout_joint.py handles any number of probes on one reference, also over several nights).

    python scripts/direct_batch.py --day 51 --base results/cand50_A.csv --probes "lvl:a:0.85:never:warm,lvl:h:0.85:pos:dry,con:a:0.10:never+b:*:50-500"
        [--prefix cand51] [--prior 0.015]
    python scripts/direct_batch.py --from-spec work/logs/day51_spec.json            (rebuild with the stored arguments; used by the night chain)

Probe names: lvl|con : h|a : factor (lvl) or s (con) : classes : zones [: agelo-agehi]; '+' separates classes / zones, '*' = all.
Also  grp:<h|a>:<factor>:<groups csv>:<group>[:<agelo-agehi>]  = level probe on a group of site x scenarios (csv with site, sc, group;
scripts/apply_group_level.py), and  file:<csv>:<factor>[:<probes json>]  = an existing single-factor file of the design runner whose base file is bit-identical to --base
(part-step probes such as F_x; the file is copied and its checksum stored; the readout entry is taken from the probes json, default day50).
Writes results/<prefix>_P<i>.csv, work/logs/<prefix>_manifest.txt (probe files only), work/logs/<prefix>_design.json (files = base + probes,
identity inclusion, energies, "direct": true), work/logs/<prefix>_probes_auto.json (readout entries with eps and the default prior),
work/logs/day<NN>_spec.json (prefix + stored arguments) and work/logs/<prefix>_sanity.txt.
Checks per probe, on the file read back from disk: shape / finite / bounds as scripts/submit_manifest.sh, change confined to the block and
column, change = -s x value (level) or +-s x value (contrast) cell by cell, energy = s^2 S2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import shift  # noqa: E402
import dec_eval as de  # noqa: E402

PYT = sys.executable
FL = 0.401412 / 3.39567
HCAP = 35.0 / 1.14114 / 10.90645
HCAP_LEVEL = 2.8122
ALL_ZONES = "boreal,cool,warm,moist,dry,cluster"


sys.path.insert(0, str(ROOT / "scripts"))
from apply_contrast import pair_mask  # noqa: E402
from apply_group_level import group_mask  # noqa: E402


def safe_name(nm: str) -> str:
    if nm.startswith("grp:"):                                              # grp:a:0.8:work/fire/groups_regime.csv:pos|low:50-500 -> grp_a_08_pos-low_50-500
        f = nm.split(":"); nm = ":".join(f[:3] + [f[4].replace("|", "-")] + f[5:])
    return nm.replace(":", "_").replace("+", "-").replace("*", "all").replace(".", "").replace("|", "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=0); ap.add_argument("--base", default=""); ap.add_argument("--probes", default="")
    ap.add_argument("--prefix", default=""); ap.add_argument("--prior", type=float, default=0.015); ap.add_argument("--from-spec", default="")
    args = ap.parse_args()
    if args.from_spec:
        sp = json.load(open(ROOT / args.from_spec))["direct"]; day, base, names, pre, prior = sp["day"], sp["base"], sp["probes"], sp["prefix"], sp.get("prior", 0.015)
    else:
        assert args.day and args.base and args.probes, "--day, --base and --probes are needed"
        day, base, names, pre, prior = args.day, args.base, [x for x in args.probes.split(",") if x], args.prefix or f"cand{args.day}", args.prior
    san = ROOT / f"work/logs/{pre}_sanity.txt"; san.write_text("bad\n")
    base_md5 = hashlib.md5((ROOT / base).read_bytes()).hexdigest()
    if args.from_spec and sp.get("base_md5") and sp["base_md5"] != base_md5:
        print(f"sanity: BAD - the base {base} changed since the spec was written (md5 {base_md5[:8]}, spec {sp['base_md5'][:8]})"); sys.exit(3)
    known = [ln.split()[0] for g in sorted((ROOT / "work/logs").glob("cand*_md5.txt")) for ln in g.read_text().splitlines() if ln.strip()]
    print(f"base {base}: md5 {base_md5[:8]}" + (" (in a night's checksum list)" if base_md5 in known else " (NOT in any night's checksum list - check that this is the submitted file)"))
    B = shift.read_predictions(ROOT / base).astype(np.float64); A48 = shift.read_predictions(ROOT / "results/cand48_A.csv").astype(np.float64)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never")))[site]
    strat = np.asarray(de.strata_of_sites())[site]
    bs_path = ROOT / "results/board_scores.json"; bs = json.load(open(bs_path)) if bs_path.exists() else {}
    if bs.get(Path(base).name) is None:
        print(f"NOTE: the base {Path(base).name} carries no board score yet - the probes can only be read once it does")
    files, factors, energies, manifest, auto, ok_all = [Path(base).name], [], {}, [], {}, True
    src_md5s: dict[str, str] = {}
    for i, nm in enumerate(names, 1):
        f = nm.split(":")
        if f[0] == "file":
            # file:<single-factor csv of the design runner>:<factor name>[:<probes json with the readout entry>] - a file built on a recipe whose
            # base file is bit-identical to this batch's base (e.g. a part-step probe F_x of a test design); it is copied, never rebuilt
            src, fac = f[1], f[2]; out = f"results/{pre}_P{i}.csv"
            src_md5 = hashlib.md5((ROOT / src).read_bytes()).hexdigest()
            want = (sp.get("src_md5", {}) if args.from_spec else {}).get(nm)
            if want and want != src_md5:
                print(f"P{i} {nm}: source changed since the spec was written"); ok_all = False; continue
            (ROOT / out).write_bytes((ROOT / src).read_bytes())
            try:
                P = shift.read_predictions(ROOT / out).astype(np.float64)
            except Exception as ex:
                print(f"P{i} {nm}: file does not parse: {str(ex)[:200]}"); ok_all = False; continue
            d = P - B; E = float(np.sum(d ** 2)); e48 = float(np.mean((P - A48) ** 2))
            ok = (hashlib.md5((ROOT / out).read_bytes()).hexdigest() == src_md5 and P.shape == (185130, 2) and bool(np.isfinite(P).all()) and P[:, 1].min() >= -0.1183
                  and P[:, 0].min() >= 0 and P.max() < 20 and 0.0005 <= e48 <= 0.02 and E > 1.0)
            ok_all &= bool(ok); src_md5s[nm] = src_md5
            print(f"{'ok  ' if ok else 'BAD '}{Path(out).name}: copy of {src} (factor {fac})  energy against the base {E:.1f} (heights {np.sum(d[:, 0] ** 2):.1f}, agb {np.sum(d[:, 1] ** 2):.1f})  "
                  f"cells changed {int((np.abs(d) > 1e-9).any(1).sum())}  mean sq against cand48_A {e48:.5f}")
            files.append(Path(out).name); factors.append(fac); energies[fac] = E
            manifest.append(f"{out}\tP{i}: {Path(base).name} with ONE factor changed - {fac} (single-factor file {Path(src).name} of the design runner), energy {E:.0f}")
            ref_json = ROOT / (f[3] if len(f) > 3 else "work/logs/day50_probes.json")
            entry = json.load(open(ref_json)).get(fac) if ref_json.exists() else None
            auto[fac] = entry if entry else {"type": "kappa", "t": None, "parts": {}, "prior": [1.0, 0.3], "note": "fill in by hand"}
            continue
        assert f[0] in ("lvl", "con", "grp") and len(f) in (5, 6) and f[1] in ("h", "a"), f"bad probe {nm}"
        col = 0 if f[1] == "h" else 1; val_arg = float(f[2]); s = (1.0 - val_arg) if f[0] in ("lvl", "grp") else val_arg
        classes = "never,b,pos,cluster" if f[3] == "*" else f[3].replace("+", ","); zones = ALL_ZONES if f[4] == "*" else f[4].replace("+", ",")
        out = f"results/{pre}_P{i}.csv"
        if f[0] == "grp":
            # level probe on a GROUP of site x scenarios: grp:<h|a>:<factor>:<groups csv>:<group>[:<agelo-agehi>]  (scripts/apply_group_level.py)
            cmd = [PYT, "scripts/apply_group_level.py", "--inp", base, "--out", out, "--groups", f[3], "--group", f[4], "--col", f[1], "--factor", str(val_arg)]
        elif f[0] == "lvl":
            cmd = [PYT, "scripts/apply_level.py", "--inp", base, "--out", out, "--factor", str(val_arg), "--classes", classes, "--strata", zones]
            if col == 1:
                cmd += ["--col", "a", "--min", "0.02"]
        else:
            # heights: only pairs (both scenarios of a site x seed age) that stay below the 35 m cap - a capped cell would turn the contrast into a level change
            pmax = round(HCAP / (1.0 + val_arg) - 1e-6, 6) if col == 0 else 0.0
            cmd = [PYT, "scripts/apply_contrast.py", "--inp", base, "--out", out, "--col", f[1], "--s", str(val_arg), "--classes", classes, "--strata", zones]
            if pmax > 0:
                cmd += ["--pair-max", str(pmax)]
        if len(f) == 6:
            cmd += ["--ages", f[5].replace("-", ":")]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"P{i} {nm}: tool failed: {res.stderr[-300:]}"); ok_all = False; continue
        # expected change, from the base alone
        if f[0] == "grp":
            m = group_mask(f[3], f[4])
        else:
            m = np.isin(cls, classes.split(",")) & np.isin(strat, zones.split(","))
        if len(f) == 6:
            lo, hi = (int(x) for x in f[5].split("-")); m &= (age >= lo) & (age <= hi)
        val = B[:, 0] if col == 0 else B[:, 1] + FL
        m &= val > (0.3 if col == 0 else 0.02)
        if f[0] == "con" and col == 0:
            m = pair_mask(m, val, site, age, scen, pmax)
        sgn = -np.ones(len(B)) if f[0] in ("lvl", "grp") else np.where(scen == "ssp585", 1.0, -1.0)
        new_val = val[m] * (1.0 + s * sgn[m])
        if col == 0:                                                        # the tools' height caps: apply_level 2.8122, apply_contrast max(35 m, value)
            new_val = np.minimum(new_val, HCAP_LEVEL) if f[0] in ("lvl", "grp") else np.clip(new_val, 0.0, np.maximum(HCAP, val[m]))
        exp = np.zeros_like(B); exp[m, col] = new_val - val[m]; E_exp = float(np.sum(exp ** 2))
        S2 = float(np.sum(val[m] ** 2)); e_site = np.bincount(site[m], weights=val[m] ** 2, minlength=6171); e_site = e_site[e_site > 0]
        eps = float(np.sqrt(np.sum(e_site ** 2)) / S2)
        try:
            P = shift.read_predictions(ROOT / out).astype(np.float64)
        except Exception as ex:                                             # a damaged write shows up here
            print(f"P{i} {nm}: file does not parse: {str(ex)[:200]}"); ok_all = False; continue
        d = P - B; E = float(np.sum(d ** 2)); dev = float(np.abs(d - exp).max()); e48 = float(np.mean((P - A48) ** 2))
        ok = (P.shape == (185130, 2) and bool(np.isfinite(P).all()) and P[:, 1].min() >= -0.1183 and P[:, 0].min() >= 0 and P.max() < 20
              and 0.0005 <= e48 <= 0.02 and dev < 2e-6 and abs(E - E_exp) <= 1e-3 * E_exp + 0.5)
        ok_all &= bool(ok); sn = safe_name(nm)
        where = f"group {f[4]} of {Path(f[3]).name}" if f[0] == "grp" else f"{classes} | {zones}"
        txt = (f"P{i}: {Path(base).name} with ONE block changed - " + ("CONTRAST PROBE (ssp585 up, ssp126 down): " if f[0] == "con" else "LEVEL PROBE: ")
               + f"{'agb' if col == 1 else 'vegetated heights'} x {val_arg:g} [{where}{' | ages ' + f[5] if len(f) == 6 else ''}], energy {E:.0f}")
        print(f"{'ok  ' if ok else 'BAD '}{Path(out).name}: {nm}  cells {int(m.sum())}  S2 {S2:.0f}  eps {100 * eps:.2f} %  energy {E:.1f} (s^2 S2 {s * s * S2:.1f})  "
              f"max cell deviation {dev:.1e}  mean sq against cand48_A {e48:.5f}")
        files.append(Path(out).name); factors.append(sn); energies[sn] = E; manifest.append(f"{out}\t{txt}")
        auto[sn] = {"type": "contrast" if f[0] == "con" else "level", "s": round(s, 6), "block": sn, "prior": [0.0, prior], "eps": round(eps, 4)}
    k = len(factors)
    if not ok_all or k != len(names):
        print("sanity: BAD - nothing written for the night chain"); sys.exit(3)
    logs = ROOT / "work/logs"
    (logs / f"{pre}_manifest.txt").write_text("\n".join(manifest) + "\n")
    design = {"files": files, "factors": factors, "steps": {f_: 1.0 for f_ in factors}, "include": [[0] * k] + [[1 if j == i else 0 for j in range(k)] for i in range(k)],
              "energies": energies, "base": Path(base).name, "direct": True}
    (logs / f"{pre}_design.json").write_text(json.dumps(design, indent=1))
    (logs / f"{pre}_probes_auto.json").write_text(json.dumps(auto, indent=1))
    spec = {"prefix": pre, "direct": {"day": day, "base": base, "base_md5": base_md5, "probes": names, "prefix": pre, "prior": prior, "src_md5": src_md5s},
            "note": f"direct probe batch on {Path(base).name}; rebuild: python scripts/direct_batch.py --from-spec work/logs/day{day}_spec.json"}
    spec_path = logs / f"day{day}_spec.json"
    if not args.from_spec:                                                # a rebuild never rewrites the spec (its checksum is part of the night's list)
        if spec_path.exists():
            spec_path.with_name(spec_path.name + f".bak_{time.strftime('%H%M', time.gmtime())}").write_text(spec_path.read_text())
        spec_path.write_text(json.dumps(spec, indent=1))
    san.write_text("ok\n")
    print(f"sanity: ok - {k} probes on {Path(base).name}; manifest work/logs/{pre}_manifest.txt, design work/logs/{pre}_design.json, readout entries work/logs/{pre}_probes_auto.json")


if __name__ == "__main__":
    main()

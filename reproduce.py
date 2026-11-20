#!/usr/bin/env python
"""Reproduce the submitted predictions from the stored artefacts, without retraining.

    python reproduce.py final      [--weights ../3_model_weights] [--run run]
    python reproduce.py blend      [--weights ../3_model_weights] [--run run]
    python reproduce.py emulators  --data /path/to/competition/data [--weights ../3_model_weights] [--run run] [--device cpu]
    python reproduce.py checks     [--weights ../3_model_weights] [--run run]

final      Rebuilds the submitted file from the stored inputs: forced-fire rows from the stored ED outputs, the candidate of 13 Nov with
           its level probes (scripts/design_run.py -> scripts/build_candidate2.py, scripts/apply_*.py), the direct probes of the later
           nights (scripts/direct_batch.py) and the second stage (scripts/second_stage.py). The result is compared with the submitted
           file (md5 and the largest difference). Needs numpy and pandas only; about 15 minutes on 4 cores, 2 GB of disk.
blend      Rebuilds the blend S (results/spatcorr_stack_20261110_s08.csv), the reference of every row, from the stack of 10 Oct, the
           scored probe files and the score registry as it stood on 9 Nov (scripts/spatial_correction.py).
emulators  Runs one GRU checkpoint and one LightGBM model on the competition test files and compares the output with the stored
           prediction files (scripts/predict_latent.py, scripts/predict_gbdt.py). Needs the two test archives of the competition in
           --data, PyTorch and LightGBM.
checks     Recomputes the validation numbers quoted in the report from stored tables (hold-out scores of the emulators, ED targets on
           120 training cells).

Every step is listed in recipes/*.json as the exact command line; this script only prepares a working copy (code + extracted
archives) under --run, runs the commands there and compares the results. Nothing is downloaded and nothing is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md5(path: Path) -> str:
    """MD5 of a file, read in blocks."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_predictions(path: Path):
    """The two prediction columns of a submission file as a (rows, 2) array."""
    import numpy as np
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2))


def compare(rebuilt: Path, stored: Path, tol: float = 0.0) -> bool:
    """Print md5 and the largest difference of two submission files; True if identical or within `tol`."""
    import numpy as np
    a, b = md5(rebuilt), md5(stored)
    if a == b:
        print(f"   IDENTICAL  {rebuilt.name}  md5 {a}")
        return True
    x, y = read_predictions(rebuilt), read_predictions(stored)
    if x.shape != y.shape:
        print(f"   DIFFERENT SHAPE  {rebuilt.name} {x.shape} against {stored.name} {y.shape}")
        return False
    d = np.abs(x - y)
    ok = float(d.max()) <= tol
    print(f"   {'WITHIN TOLERANCE' if ok else 'DIFFERENT'}  {rebuilt.name}: largest difference {d.max():.2e}, mean {d.mean():.2e}, "
          f"cells that differ {int((d > 0).sum())} of {d.size} (md5 {a} against {b}; tolerance {tol:g})")
    return ok


def extract(archive: Path, dest: Path) -> None:
    """Unpack one archive of the weights folder into the working copy (once)."""
    stamp = dest / f".extracted_{archive.name}"
    if stamp.exists():
        return
    if not archive.exists():
        sys.exit(f"missing archive {archive} - pass the folder with the model weights as --weights")
    print(f"   unpacking {archive.name} ...", flush=True)
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:                      # Python without the extraction filters
            tf.extractall(dest)
    stamp.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def prepare(run: Path, weights: Path, archives: list[str]) -> None:
    """Working copy: the code of this folder plus the extracted archives. The tools find their inputs relative to their own location."""
    run.mkdir(parents=True, exist_ok=True)
    for name in ("src", "scripts"):
        shutil.copytree(HERE / name, run / name, dirs_exist_ok=True)
    for a in archives:
        extract(weights / a, run)
    for d in ("results", "work/logs", "work/cache", "work/models"):          # folders the tools write to
        (run / d).mkdir(parents=True, exist_ok=True)


def run_steps(recipe: dict, run: Path, extra_env: dict | None = None, tag: str = "") -> None:
    """Run the command lines of a recipe inside the working copy; stop at the first failure."""
    for i, step in enumerate(recipe["steps"], 1):
        env = {**os.environ, **step.get("env", {}), **(extra_env or {})}
        env.setdefault("OMP_NUM_THREADS", "4")
        argv = [sys.executable] + [str(a) for a in step["argv"]]
        print(f"[{i}/{len(recipe['steps'])}] {step['name']}\n      python {' '.join(str(a) for a in step['argv'])}", flush=True)
        t0 = time.time()
        log = run / "work" / "logs" / f"reproduce_{tag}_step{i}.log"
        with open(log, "w") as fh:
            rc = subprocess.run(argv, cwd=run, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        if step.get("show"):                              # a step whose printed table is the result
            print("\n".join("      " + ln for ln in lines), flush=True)
        print(f"      {time.time() - t0:.0f} s; log {log.relative_to(run)}" + (f"; last line: {lines[-1][:150]}" if lines and not step.get("show") else ""), flush=True)
        if rc != 0:
            sys.exit(f"step {i} failed (exit {rc}); see {log}")


def check_outputs(recipe: dict, run: Path, weights: Path) -> bool:
    ok = True
    for rel, want in recipe.get("intermediate_md5", {}).items():
        got = md5(run / rel)
        print(f"   {'IDENTICAL' if got == want else 'DIFFERENT'}  {rel}  md5 {got}" + ("" if got == want else f" (stored {want})"))
        ok &= got == want
    for item in recipe.get("compare", []):
        ok &= compare(run / item["rebuilt"], (weights / item["stored"]) if not item.get("in_run") else (run / item["stored"]), float(item.get("tolerance", 0.0)))
    return ok


def cmd_recipe(name: str, args: argparse.Namespace) -> None:
    recipe = json.loads((HERE / "recipes" / f"{name}.json").read_text())
    run, weights = Path(args.run).resolve(), Path(args.weights).resolve()
    print(f"{recipe['title']}\n   working copy {run}\n   weights      {weights}")
    prepare(run, weights, recipe["archives"])
    extra = {}
    if name == "emulators":
        if not args.data:
            sys.exit("emulators: pass the folder with test_ssp126.npz and test_ssp585.npz as --data")
        data = Path(args.data).resolve()
        for f in ("test_ssp126.npz", "test_ssp585.npz"):
            if not (data / f).exists():
                sys.exit(f"{data / f} not found")
        link = run / "data"
        if not link.exists():
            link.symlink_to(data, target_is_directory=True)
        for tag in recipe["models"]:                      # checkpoints: weights/emulators/<tag> -> work/models/<tag>
            shutil.copytree(weights / "emulators" / tag, run / "work" / "models" / tag, dirs_exist_ok=True)
        for step in recipe["steps"]:
            step["argv"] = [a.replace("{device}", args.device) for a in step["argv"]]
    run_steps(recipe, run, extra, tag=name)
    ok = check_outputs(recipe, run, weights)
    print("RESULT: " + ("reproduced" if ok else "NOT reproduced - see the lines above"))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("final", "blend", "emulators", "checks"))
    ap.add_argument("--weights", default=str(HERE.parent / "3_model_weights"), help="folder with the model weights and archives")
    ap.add_argument("--run", default=str(HERE / "run"), help="working copy (created; about 2 GB for 'final')")
    ap.add_argument("--data", default="", help="competition data (only for 'emulators'): folder with test_ssp126.npz and test_ssp585.npz")
    ap.add_argument("--device", default="cpu", help="torch device for 'emulators'")
    args = ap.parse_args()
    cmd_recipe(args.what, args)


if __name__ == "__main__":
    main()

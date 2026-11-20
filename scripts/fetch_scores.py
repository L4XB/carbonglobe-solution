#!/usr/bin/env python
"""Pull the public scores of completed submissions into ``results/board_scores.json``.

Only files that exist in ``results/`` are registered, and an existing entry is never overwritten (a file
name scored twice must be the same file; the sha256 in the description is checked against the local file).

    ./.venv/bin/python scripts/fetch_scores.py
"""
from __future__ import annotations

import csv
import hashlib
import io as _io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import shutil as _sh
KAGGLE = next((c for c in ("kaggle",
                           "kaggle", _sh.which("kaggle") or "")
               if c and Path(c).exists()), "kaggle")
COMP = "ieee-bigdata-cup-2026-ai-emulation-challenge"


def main() -> None:
    out = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMP, "--csv"], capture_output=True, text=True, check=True, timeout=300).stdout
    rows = list(csv.DictReader(_io.StringIO(out[out.index("ref,"):])))
    reg_path = ROOT / "results" / "board_scores.json"
    reg = json.load(open(reg_path))
    added = []
    for r in rows:
        name, score, status = r["fileName"], r["publicScore"], r["status"]
        if "ERROR" in status:
            continue
        if "COMPLETE" not in status or not score:
            print(f"  PENDING: {name} {status}")
            continue
        path = ROOT / "results" / name
        if not path.exists() or name in reg:
            continue
        m = re.search(r"sha256 ([0-9a-f]{8})", r["description"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        if m and m.group(1) != sha:
            print(f"  SKIP {name}: local sha {sha} != submitted {m.group(1)}")
            continue
        reg[name] = float(score)
        added.append((name, float(score)))
    json.dump(reg, open(reg_path, "w"), indent=1)
    for n, s in added:
        print(f"  registered {n} = {s}")
    print(f"{len(added)} new, {len(reg)} scored files in the registry")


if __name__ == "__main__":
    main()

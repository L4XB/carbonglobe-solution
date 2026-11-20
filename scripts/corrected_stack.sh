#!/bin/bash
# Build the corrected stack from every scored spatial/direction probe and validate it (does not submit).
#   bash scripts/corrected_stack.sh 20261025 [scale]   -> results/spatcorr_stack_20261025[_sXX].csv
set -eu
cd "$(dirname "$0")/.."
TAG=${1:?tag (date) required}
SCALE=${2:-1.0}
EXTRA=${3:-}
PY=./.venv/bin/python
$PY scripts/fetch_scores.py | tail -3
if [ "$SCALE" = "1.0" ]; then OUT=results/spatcorr_stack_${TAG}.csv; else OUT=results/spatcorr_stack_${TAG}_s${SCALE/./}.csv; fi
# predicted public-energy ratios of the un-mirrored cluster probes from the mirrored pairs (refreshed every build)
RATIO=""
$PY scripts/cluster_public_share.py > work/logs/cluster_public_share.log 2>&1 && RATIO="--ratio-file work/cluster_ratio_pred.json"
$PY scripts/spatial_correction.py --base results/stack_20261010.csv --out "$OUT" --scale "$SCALE" $EXTRA $RATIO | tail -2
$PY - "$OUT" <<'PYEOF'
import sys; sys.path.insert(0, "src")
import numpy as np
from cgb import shift
P = shift.read_predictions(sys.argv[1])
B = shift.read_predictions("results/stack_20261010.csv")
ok = P.shape == (185130, 2) and np.isfinite(P).all() and P.min() >= 0 and P.max() < 50
print(("ok" if ok else "BAD"), sys.argv[1], "E[(P-B)^2]", float(np.mean((P - B) ** 2))); sys.exit(0 if ok else 1)
PYEOF

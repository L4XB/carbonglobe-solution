#!/bin/bash
# Row-level ORNL (GFDL-ESM4 CLIM_CO2) probe directions for one position file:
#   bash scripts/ornl/build_qdirs.sh <positions npz/npy> <tag> [sigma]
# -> work/ornl2522/state_force_GFDL-ESM4_CLIM_CO2_s<sigma>_<tag>.npz, work/dir_ornl_q<tag>_*.npy and
#    work/dir_ornl_q<tag>_qcontrast_ah_all.npy (agb + height contrast in one direction)
set -eu
cd "$(dirname "$0")/../.."
POS=$1; TAG=$2; SIG=${3:-0.5}
P=python
export OMP_NUM_THREADS=2
$P scripts/ornl/ornl_state_force.py --sigma "$SIG" --pos "$POS" --tag "_$TAG" | tail -1
$P scripts/ornl/ornl_qdirs.py --src "state_force_GFDL-ESM4_CLIM_CO2_s${SIG}_$TAG" --tag "q$TAG" > "work/logs/qdirs_q$TAG.log" 2>&1
grep -E "^qforce_(old_a_out|ym_a_out|all_h_out|all_a_cl|all_h_cl) |^qcontrast_(a|h)_all " "work/logs/qdirs_q$TAG.log"
$P - "$TAG" <<'PYEOF'
import sys
import numpy as np
t = sys.argv[1]
a = np.load(f"work/dir_ornl_q{t}_qcontrast_a_all.npy"); h = np.load(f"work/dir_ornl_q{t}_qcontrast_h_all.npy")
np.save(f"work/dir_ornl_q{t}_qcontrast_ah_all.npy", a + h)
print(f"qcontrast_ah_all E_all {np.mean((a + h) ** 2):.5f}")
PYEOF

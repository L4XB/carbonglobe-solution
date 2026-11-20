#!/bin/bash
# Re-run ONE window of a kept run directory and keep its monthly output tables (for diagnostics).
#   bash scripts/ed/rerun_window.sh <run dir of a --keep site> <w126|w585> <age> [extra ENV=VALUE ...]
# Output: <run dir>/out/<key>_a<age>/ (ED's monthly tables: *.natural.water, *.biomass, *.cfluxes, ...).
set -u
RUN=$1; KEY=$2; AGE=$3; shift 3
ROOT=.
EDLU=${ED_EDLU:-$ROOT/work/edv3/build_fire/edlu}
cd "$RUN" || exit 1
EXP=${KEY}_a${AGE}
FX=fire_x_${KEY}_a${AGE}.txt; [ -f "$FX" ] || FX=fire_x_${KEY}.txt
ENVS=(ED_CO2_AS_IS=1 ED_TREEFALL=0.012 ED_WINDOW_YEAR0=1981 ED_RESTART_SUFFIX=_y${AGE})
[ -f "$FX" ] && ENVS+=(ED_FIRE_X_FILE="$PWD/$FX")
[ -d "cache_${KEY}" ] && ENVS+=(ED_MECH_CACHE_DIR="$PWD/cache_${KEY}")
env "${ENVS[@]}" "$@" "$EDLU" "$EXP" "models_${KEY}.cfg" > "${EXP}.rerun.log" 2>&1
echo "rc $? -> $RUN/out/$EXP"

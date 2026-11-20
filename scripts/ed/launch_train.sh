#!/bin/bash
# Launch scripts/ed/train_site.py (the test pipeline's ED configuration on TRAINING cells) in the background and record its PID in <root>.pid.
#   [CG_OVERRIDE=...] [BUILD=build_fire6] bash scripts/ed/launch_train.sh <config> <root dir> <file with a comma list of training cells> [workers]
# The ED scratch (run/, inputs/) lives in work/<root name>.noindex so Spotlight does not index it. Stop: kill the PID tree by hand
# (scripts/ed/stop_run.sh only accepts run_many launches).
set -u
cd "$(dirname "$0")/../.."
CFG=$1; R=$2; CELLS=$3; W=${4:-4}
[ -e "$R.pid" ] && ps -p "$(cat "$R.pid")" >/dev/null 2>&1 && { echo "$R already running (pid $(cat "$R.pid"))"; exit 1; }
NOIDX=$PWD/work/$(basename "$R").noindex
mkdir -p "$R" "$NOIDX/run" "$NOIDX/inputs"
for d in run inputs; do
  if [ -d "$R/$d" ] && [ ! -L "$R/$d" ]; then mv "$R/$d" "$NOIDX/${d}_old_$(date -u +%H%M%S)"; fi
  [ -e "$R/$d" ] || ln -s "$NOIDX/$d" "$R/$d"
done
ED_EDLU=$PWD/work/edv3/${BUILD:-build_fire6}/edlu nohup caffeinate -i python \
  scripts/ed/train_site.py --config "$CFG" --root "$R" --sites "$(cat "$CELLS")" --workers "$W" > "$R.log" 2>&1 &
echo $! > "$R.pid"
echo "launched $CFG on training cells in $R (pid $(cat "$R.pid"), log $R.log)"

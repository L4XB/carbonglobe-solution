#!/bin/bash
# Launch an ED run_many in the background and record its PID in <root>.pid (stop it with scripts/ed/stop_run.sh).
#   bash scripts/ed/launch_run.sh <config> <root dir> <site file> [workers]
set -u
cd "$(dirname "$0")/../.."
CFG=$1; R=$2; SITES=$3; W=${4:-12}
[ -e "$R.pid" ] && ps -p "$(cat "$R.pid")" >/dev/null 2>&1 && { echo "$R already running (pid $(cat "$R.pid"))"; exit 1; }
# ED scratch (~1000 files, 150 MB per site) in a .noindex folder, so Spotlight (mds_stores) does not index it
NOIDX=$PWD/work/$(basename "$R").noindex
mkdir -p "$R" "$NOIDX/run" "$NOIDX/inputs"
for d in run inputs; do [ -e "$R/$d" ] || ln -s "$NOIDX/$d" "$R/$d"; done
ED_EDLU=$PWD/work/edv3/${BUILD:-build_dryfix}/edlu nohup caffeinate -i python \
  scripts/ed/run_many.py --config "$CFG" --root "$R" --site-file "$SITES" --workers "$W" > "$R.log" 2>&1 &
echo $! > "$R.pid"
echo "launched $CFG in $R (pid $(cat "$R.pid"), log $R.log)"

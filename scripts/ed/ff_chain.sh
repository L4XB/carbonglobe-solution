#!/bin/bash
# Forced-fire stages one after the other on this Mac (each stage "root:site file:workers"; finished sites of a root are skipped).
#   nohup bash scripts/ed/ff_chain.sh work/edv3/ff_pos_mac:work/ff/a.txt:9 work/edv3/ff_cl_mac:work/ff/b.txt:9 > work/edv3/ff_chain_mac.log 2>&1 & echo $! > work/edv3/ff_chain_mac.pid
# Stop: bash scripts/ed/stop_run.sh work/edv3/ff_chain_mac.pid   (kills this chain and everything below it, by PID tree)
set -u
cd "$(dirname "$0")/../.."
export ED_EDLU=${ED_EDLU:-$PWD/work/edv3/build_fire6/edlu} ED_FIRE_X_YEARCAP=1 ED_FIRE_X_SHIFT=1
PYT=python
for st in "$@"; do
  IFS=: read -r root sites w <<< "$st"
  echo "$(date -u +%H:%M:%S) stage $root $sites workers $w"
  caffeinate -i $PYT scripts/ed/forced_fire.py --root "$root" --sites "$sites" --workers "$w" >> "$root.log" 2>&1
  echo "$(date -u +%H:%M:%S) stage $root ended with $?"
done
echo "$(date -u +%H:%M:%S) chain finished"

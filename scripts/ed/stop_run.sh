#!/bin/bash
# Stop one ED launch (run_many.py, forced_fire.py or train_site.py) and every process below it, selected by PID only. No command-line pattern matching:
# a pattern also matches the shell that runs the stop command (its own text carries the pattern), and on 11-09 killing
# the parents of such matches killed the Claude Code session itself.
#   bash scripts/ed/stop_run.sh work/edv3/<root>.pid
set -u
PIDFILE=$1
ROOT_PID=$(cat "$PIDFILE") || exit 1
cmd=$(ps -o command= -p "$ROOT_PID")
case "$cmd" in *run_many.py*|*forced_fire.py*|*train_site.py*|*ff_chain.sh*) ;; *) echo "pid $ROOT_PID is not an ED launch (run_many / forced_fire / train_site; already ended?): $cmd"; exit 1;; esac
# the root and all its descendants, from one process-table snapshot
TREE=$(ps -Ao pid=,ppid= | awk -v r="$ROOT_PID" '
  { par[$1] = $2 }
  END { want[r] = 1; ch = 1
        while (ch) { ch = 0; for (p in par) if (!(p in want) && (par[p] in want)) { want[p] = 1; ch = 1 } }
        for (p in want) print p }')
# never signal this shell or one of its ancestors
SELF=" "; p=$$; while [ -n "$p" ] && [ "$p" -gt 1 ]; do SELF="$SELF$p "; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
KILL=""; for p in $TREE; do case "$SELF" in *" $p "*) echo "skip own ancestor $p";; *) KILL="$KILL $p";; esac; done
echo "stopping $(echo $KILL | wc -w | tr -d ' ') processes under $ROOT_PID"
kill $KILL 2>/dev/null; sleep 5
for p in $KILL; do ps -p "$p" >/dev/null 2>&1 && kill -9 "$p" 2>/dev/null; done
sleep 1
left=0; for p in $KILL; do ps -p "$p" >/dev/null 2>&1 && left=$((left + 1)); done
echo "left alive: $left"

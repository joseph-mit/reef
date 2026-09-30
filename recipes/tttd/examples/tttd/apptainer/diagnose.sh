#!/bin/bash
# Everything needed to tell how the last run in a state directory went, read
# from its files only, so it is safe while jobs run:
#
#   bash diagnose.sh ~/orcd/scratch/reef-work/runs/spottt/circle_packing_26
#
# A smoke keeps its state on the node's /tmp; read it from inside the job:
#
#   srun --jobid=<job> --overlap bash diagnose.sh /tmp/$USER-reef-smoke-<job>/<method>/<task>
#
# Only the last run's part of reef.log is shown: before run.sh set earlier
# logs aside, Reef replayed every earlier run's service output into it.

state=${1:?usage: diagnose.sh <state directory>}
log="$state/reef.log"
echo "== $log"
if [ ! -f "$log" ]; then
    echo "no reef.log"
    exit 1
fi
ls -la "$log"

start=$(grep -n "starting slime-driver" "$log" | tail -n 1 | cut -d: -f1)
section() {
    tail -n +"${start:-1}" "$log" | grep -v '^ *[\^~]*$'
}
failure='traceback|cuda error|illegal memory|device-side assert|sigquit|out of memory|exited before ready|killed'
# Lines that match those words but are not failures: config dumps and a
# harmless import warning.
benign='error_injection|server_args=|Global server args|package walk|may not be an issue'

echo "== settings of this run"
section | grep -o -E "train_env_vars[^}]*\}|mem_fraction_static=[0-9.]+|cuda_graph_max_bs=[0-9]+|max_running_requests=[0-9A-Za-z]+" | sort | uniq -c

echo "== error lines (first 40)"
section | grep -n -i -E "$failure|error|exception|assert" | grep -v -i -E "$benign" | head -n 40 | cut -c1-300

echo "== around the first failure"
first=$(section | grep -n -i -E "$failure" | head -n 1 | cut -d: -f1)
if [ -n "$first" ]; then
    from=$((first > 40 ? first - 40 : 1))
    section | sed -n "${from},$((first + 25))p" | cut -c1-300
else
    echo "no failure line"
fi

echo "== GPU memory reports (last 4)"
section | grep "Memory-Usage" | tail -n 4 | cut -c1-300

echo "== last 25 lines"
section | tail -n 25 | cut -c1-300

echo "== step events (last 3)"
tail -n 3 "$state/events.jsonl" 2>/dev/null | cut -c1-300

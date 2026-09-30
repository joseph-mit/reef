#!/bin/bash
# Everything about the running TTT-Discover runs in one view, from the login
# node:
#
#   bash recipes/tttd/examples/tttd/apptainer/status.sh [job id]
#
# For the job (default: your running reef-multi or reef-run job) it shows its
# state, then for every run it holds: the committed steps (run_status.py),
# the current step and how long it has been going, how long ago the stack
# last wrote to its log, what it wrote, and any failure lines. A log that has
# been silent for over half an hour means the stack is stuck, not slow.
#
# The per-run part runs on the job's node through srun, so it also sees runs
# kept on the node's local disk (/tmp/$USER-reef-runs).

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${1:-}" = --on-node ]; then
    shift
    # Lines that match the failure words but are not failures.
    benign='error_injection|server_args=|Global server args|package walk|may not be an issue'
    failure='traceback|cuda error|illegal memory|device-side assert|sigquit|out of memory|exited before ready|killed|exception'
    now=$(date +%s)
    states=()
    for root in "$@"; do
        states+=("$root"/*/*)
    done
    for state in "${states[@]}"; do
        [ -f "$state/events.jsonl" ] || continue
        log="$state/reef.log"
        # Skip runs no stack has touched for two days.
        [ -f "$log" ] && [ $((now - $(stat -c %Y "$log"))) -lt 172800 ] || continue
        echo
        shown=$state
        [[ "$state" == "$HOME"/* ]] && shown="~${state#"$HOME"}"
        echo "######## $shown"
        table=$(python3 "$here/run_status.py" "$state")
        # The header and the last ten steps.
        head -n 2 <<< "$table"
        tail -n +3 <<< "$table" | tail -n 10
        last=$(grep '"tttd_step_' "$state/events.jsonl" | tail -n 1)
        started=$(grep -o '"time": [0-9.]*' <<< "$last" | grep -o '[0-9]*' | head -n 1)
        if [[ "$last" == *tttd_step_started* ]] && [ -n "$started" ]; then
            step=$(grep -o '"step": [0-9]*' <<< "$last" | grep -o '[0-9]*$')
            echo "step $((step + 1)) running for $(((now - started) / 60)) min"
        fi
        echo "log last written $(((now - $(stat -c %Y "$log")) / 60)) min ago:"
        grep -v -E '^ *[\^~]*$|repeated [0-9]+x' "$log" | tail -n 2 | cut -c1-220
        start=$(grep -n "starting slime-driver" "$log" | tail -n 1 | cut -d: -f1)
        failures=$(tail -n +"${start:-1}" "$log" | grep -i -E "$failure" | grep -v -i -E "$benign")
        if [ -n "$failures" ]; then
            echo "FAILURE LINES (last 5):"
            tail -n 5 <<< "$failures" | cut -c1-220
        else
            echo "no failure lines"
        fi
    done
    exit 0
fi

job=${1:-$(squeue -h -u "$USER" -n reef-multi,reef-run -t RUNNING -o %i | head -n 1)}
if [ -z "$job" ]; then
    echo "no running reef-multi or reef-run job"
    squeue -u "$USER" -n reef-multi,reef-run,reef-smoke -o "%.10i %.12j %.8T %.12M %.12l %.20R"
    exit 1
fi
squeue -j "$job" -o "%.10i %.12j %.8T %.12M %.12l %.20R"
work=${REEF_WORK_ROOT:-$HOME/orcd/scratch/reef-work}
# The run roots are expanded on the node: /tmp there is not the login node's.
timeout 180 srun --jobid="$job" --overlap bash "$here/status.sh" --on-node \
    "$work/runs" "/tmp/$USER-reef-runs" < /dev/null

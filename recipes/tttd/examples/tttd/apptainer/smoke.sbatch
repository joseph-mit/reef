#!/bin/bash
# End-to-end smoke of the single-stream recipes on one Engaging node, without
# Docker. For each method it checks rollout -> evaluation -> training update
# -> checkpoint -> resume on the circle-packing task:
#
#   phase 1  a fresh stack runs step 1 (2 parents x 2 attempts, scored by the
#            task's own judge, one PPO or SPO update, a durable checkpoint);
#   crash    every process of the stack is killed with SIGKILL, no shutdown;
#   phase 2  a new stack restores the checkpoint and the search archive and
#            runs step 2;
#   check    check_smoke.py verifies each stage from what Reef committed.
#
# Submit from the repository root after setup.sbatch has succeeded:
#
#   sbatch recipes/tttd/examples/tttd/apptainer/smoke.sbatch
#
# SMOKE_METHODS picks the methods (default: ppottt-smoke spottt-smoke) and
# TTTD_TASK the task (default: circle_packing_26). The results land in
# $REEF_WORK_ROOT/smoke-results/<job id>/.
#SBATCH --job-name=reef-smoke
#SBATCH --partition=pi_ppliang
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=reef-smoke-%j.out

set -euo pipefail
source "${REEF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/recipes/tttd/examples/tttd/apptainer/common.sh"

export TTTD_TASK=${TTTD_TASK:-circle_packing_26}
SMOKE_METHODS=${SMOKE_METHODS:-"ppottt-smoke spottt-smoke"}
PHASE_TIMEOUT_S=${PHASE_TIMEOUT_S:-7200}
JOB=${SLURM_JOB_ID:-$$}
export RAY_TMPDIR="/tmp/ray-$USER-$JOB"  # short: Ray's socket paths are length-limited
RESULTS="$REEF_WORK_ROOT/smoke-results/$JOB"
mkdir -p "$RESULTS" "$RAY_TMPDIR"

for required in "$REEF_SIF" "$EXAMPLE_DIR/work/model/config.json"; do
    if [ ! -e "$required" ]; then
        echo "missing $required; run setup.sbatch first" >&2
        exit 1
    fi
done

# Checkpoints go to the node's local disk when it has room: Reef's storage
# check sees a shared filesystem's total size, not your quota, and a PPO-TTT
# checkpoint carries a full critic with its optimiser state.
local_free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc '0-9')
if [ "${local_free_gb:-0}" -ge "${SMOKE_MIN_LOCAL_GB:-600}" ]; then
    STATE_ROOT="/tmp/$USER-reef-smoke-$JOB"
else
    STATE_ROOT="$REEF_WORK_ROOT/smoke-state/$JOB"
    echo "warning: /tmp has ${local_free_gb:-?}G free; checkpoints go to $STATE_ROOT, within your quota" >&2
fi
mkdir -p "$STATE_ROOT"
# shellcheck disable=SC2034  # read by in_container in common.sh
REEF_EXTRA_BINDS="$STATE_ROOT"

cleanup() {
    kill_job_processes
    rm -rf "$RAY_TMPDIR"
}
trap cleanup EXIT

# Wait until the phase has written its run summary; fail if it dies first.
wait_for_summary() {
    local summaries=$1 pid=$2 deadline=$((SECONDS + PHASE_TIMEOUT_S))
    until [ -s "$summaries" ]; do
        if ! kill -0 "$pid" 2> /dev/null; then
            echo "phase 1 exited before finishing step 1" >&2
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "phase 1 did not finish step 1 within ${PHASE_TIMEOUT_S}s" >&2
            return 1
        fi
        sleep 10
    done
}

# The end of the stack's log, where a start-up error explains itself.
show_failure() {
    local log="$1/reef.log"
    if [ -f "$log" ]; then
        echo "--- last lines of $log ---" >&2
        grep -v '^ *[\^~]*$' "$log" | tail -n 25 >&2
        echo "---" >&2
    fi
}

# GPU processes that belong to this job; on a shared node nvidia-smi may list
# other users' processes too.
job_gpu_processes() {
    local pid
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null); do
        if grep -qs "job_${SLURM_JOB_ID}[/:]\|job_${SLURM_JOB_ID}$" "/proc/$pid/cgroup"; then
            echo "$pid"
        fi
    done
}

wait_for_idle_gpus() {
    local deadline=$((SECONDS + 120))
    while [ -n "$(job_gpu_processes)" ]; do
        if ((SECONDS >= deadline)); then
            echo "GPU processes survived the kill: $(job_gpu_processes | tr '\n' ' ')" >&2
            return 1
        fi
        kill_job_processes
        sleep 5
    done
}

failures=0
for method in $SMOKE_METHODS; do
    echo
    echo "== $method on $TTTD_TASK =="
    method_root="$STATE_ROOT/$method"
    state="$method_root/$TTTD_TASK"
    mkdir -p "$method_root"
    # run.sh keeps each method's state under work/<method>/; point that at a
    # fresh directory for this job, setting any earlier state aside.
    link="$EXAMPLE_DIR/work/$method"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        mv "$link" "$link.before-$JOB"
    fi
    ln -sfn "$method_root" "$link"
    mkdir -p "$RESULTS/$method"
    status=passed

    marker="$method_root/.phase-1-started"
    touch "$marker"
    echo "phase 1: step 1 on a fresh stack"
    TTTD_METHOD=$method TTTD_DRIVER=local TTTD_RUN_STEPS=1 TTTD_HOLD_AFTER_RUN=1 \
        in_container "cd '$EXAMPLE_DIR' && bash run.sh" > "$RESULTS/$method/phase1.out" 2>&1 &
    phase1=$!
    phase1_done=0
    wait_for_summary "$state/run-summaries.jsonl" "$phase1" && phase1_done=1
    echo "crash: SIGKILL to every process of the stack"
    kill_tree "$phase1"
    kill_job_processes
    wait "$phase1" 2> /dev/null || true
    if ! wait_for_idle_gpus; then
        phase1_done=0
    fi
    # Leftover shared-memory segments and Ray session files of the killed stack.
    find /dev/shm -maxdepth 1 -user "$(id -u)" -newer "$marker" -exec rm -rf {} + 2> /dev/null || true
    rm -rf "${RAY_TMPDIR:?}"/*

    if [ "$phase1_done" = 1 ]; then
        echo "phase 2: a new stack resumes and runs step 2"
        if ! TTTD_METHOD=$method TTTD_DRIVER=local \
            in_container "cd '$EXAMPLE_DIR' && bash run.sh" > "$RESULTS/$method/phase2.out" 2>&1; then
            echo "phase 2 failed; see $RESULTS/$method/phase2.out" >&2
            show_failure "$state"
            status=failed
        fi
        kill_job_processes
    else
        echo "phase 1 failed; see $RESULTS/$method/phase1.out" >&2
        show_failure "$state"
        status=failed
    fi

    echo "check"
    if ! in_container "cd '$EXAMPLE_DIR' && python3 apptainer/check_smoke.py \
            --method '$method' --task '$TTTD_TASK' --state '$state' --report '$RESULTS/$method/report.md'"; then
        status=failed
    fi

    # Keep the small result files; checkpoints stay behind in $state.
    for item in reef.log events.jsonl run-summaries.jsonl "$method-search-state.json" best_solution.py; do
        if [ -e "$state/$item" ]; then
            cp "$state/$item" "$RESULTS/$method/"
        fi
    done
    find "$state/agent-record" -name '*.commits.jsonl' -exec cp {} "$RESULTS/$method/" \; 2> /dev/null || true
    find "$state/judge" -name judge.log -exec sh -c 'cp "$1" "$2/judge-$(basename "$(dirname "$1")").log"' _ {} "$RESULTS/$method" \; \
        2> /dev/null || true
    echo "$method: $status"
    if [ "$status" != passed ]; then
        failures=$((failures + 1))
    fi
done

echo
echo "== summary =="
cat "$RESULTS"/*/report.md 2> /dev/null || true
echo "results: $RESULTS"
if [ "$failures" -gt 0 ]; then
    echo "SMOKE FAILED for $failures method(s)"
    exit 1
fi
echo "SMOKE PASSED"

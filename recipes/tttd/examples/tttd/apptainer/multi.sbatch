#!/bin/bash
# Several training runs in one job, each on the GPUs a load test ranks best.
#
# Slurm hands a job whichever GPUs are free, and a run goes at the pace of its
# slowest GPU. On node2500 some GPUs overheat under load and the hardware cuts
# their clock to a sixth, so this job takes all the GPUs its runs need, loads
# every one for a minute (gpu_check.py --rank), and gives the fastest to the
# runs listed first:
#
#   TTTD_RUNS="spottt:2 ppottt:4" TTTD_LOCAL_STATE=ppottt \
#       sbatch --gres=gpu:h200:6 --mem=1024G recipes/tttd/examples/tttd/apptainer/multi.sbatch
#
# TTTD_RUNS lists method:gpus, or method:gpus:steps to stop that run cleanly
# after that many steps in total; without it a run goes on until the job's
# time limit, and the next job resumes it from its last committed step.
# TTTD_TASK is shared (default circle_packing_26). State stays under
# $REEF_WORK_ROOT/runs/<method>, except for the methods in TTTD_LOCAL_STATE,
# which keep it on the node's local disk (/tmp/$USER-reef-runs), outside your
# quota: pin the node with --nodelist so the next job finds it. Each run's
# output goes to reef-multi-<job>-<method>.out next to this job's own.
#SBATCH --job-name=reef-multi
#SBATCH --partition=pi_ppliang
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=7-00:00:00
#SBATCH --output=reef-multi-%j.out

set -euo pipefail
source "${REEF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/recipes/tttd/examples/tttd/apptainer/common.sh"

: "${TTTD_RUNS:?set TTTD_RUNS, for example \"spottt:2 ppottt:4\" (method:gpus, best GPUs to the first)}"
export TTTD_TASK=${TTTD_TASK:-circle_packing_26}
export TTTD_DRIVER=local
# common.sh sets these inside a Slurm job; run.sh's own defaults otherwise.
: "${TTTD_REEF_PORT:=8900}" "${TTTD_JUDGE_PORT:=8082}" "${TTTD_ROUTER_PORT:=30000}" "${REEF_PORT_BASE:=23000}"
JOB=${SLURM_JOB_ID:-$$}
LOCAL_RUNS="/tmp/$USER-reef-runs"
trap 'kill_job_processes; rm -rf /tmp/ray-$USER-$JOB-*' EXIT

IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:?this job has no GPUs}"
needed=0
for run in $TTTD_RUNS; do
    IFS=: read -r _ count _ <<< "$run"
    needed=$((needed + count))
done
if [ "$needed" -gt "${#visible[@]}" ]; then
    echo "TTTD_RUNS needs $needed GPUs; this job has ${#visible[@]}" >&2
    exit 1
fi

# Rank this job's GPUs by sustained speed under a minute of load.
ranking=$(in_container "python3 '$EXAMPLE_DIR/apptainer/gpu_check.py' --rank ${GPU_RANK_SECONDS:-60}")
echo "$ranking"
read -r -a order <<< "$(awk '/^ranked:/ { $1 = ""; print }' <<< "$ranking")"
if [ "${#order[@]}" -ne "${#visible[@]}" ]; then
    echo "the GPU ranking did not cover all ${#visible[@]} GPUs" >&2
    exit 1
fi

# The first free port at or above $1.
port_from() {
    local port=$1
    while port_in_use "$port"; do
        port=$((port + 1))
    done
    echo "$port"
}

next=0
slot=0
pids=()
methods=()
for run in $TTTD_RUNS; do
    IFS=: read -r method count steps <<< "$run"
    devices=()
    for ((i = 0; i < count; i++)); do
        devices+=("${visible[${order[next]}]}")
        next=$((next + 1))
    done
    gpus=$(IFS=,; echo "${devices[*]}")

    runs_dir="$REEF_WORK_ROOT/runs"
    if [[ " ${TTTD_LOCAL_STATE:-} " == *" $method "* ]]; then
        runs_dir=$LOCAL_RUNS
    fi
    # tttd keeps its original work/<task> layout; every other method gets
    # work/<method>/, pointed at its state directory.
    if [ "$method" != tttd ]; then
        mkdir -p "$runs_dir/$method"
        link="$EXAMPLE_DIR/work/$method"
        if [ -e "$link" ] && [ ! -L "$link" ]; then
            echo "$link is a directory; move it into $runs_dir/$method first" >&2
            exit 1
        fi
        ln -sfn "$runs_dir/$method" "$link"
    fi

    output="$REEF_REPO/reef-multi-$JOB-$method.out"
    echo "$method on $TTTD_TASK: GPUs $gpus, steps ${steps:-until the time limit}, state under $runs_dir, output $output"
    (
        export TTTD_METHOD=$method CUDA_VISIBLE_DEVICES=$gpus
        export TTTD_RUN_STEPS=${steps:-}
        export RAY_TMPDIR="/tmp/ray-$USER-$JOB-$slot"
        # Each run gets its own ports; the job's block is split between them.
        TTTD_REEF_PORT=$(port_from $((TTTD_REEF_PORT + 10 * slot)))
        TTTD_JUDGE_PORT=$(port_from $((TTTD_JUDGE_PORT + 10 * slot)))
        TTTD_ROUTER_PORT=$(port_from $((TTTD_ROUTER_PORT + 10 * slot)))
        REEF_PORT_BASE=$((REEF_PORT_BASE + 64 * slot))
        export TTTD_REEF_PORT TTTD_JUDGE_PORT TTTD_ROUTER_PORT REEF_PORT_BASE
        REEF_EXTRA_BINDS="${REEF_EXTRA_BINDS:-} $runs_dir"
        mkdir -p "$RAY_TMPDIR"
        in_container "cd '$EXAMPLE_DIR' && bash run.sh"
    ) > "$output" 2>&1 &
    pids+=($!)
    methods+=("$method")
    slot=$((slot + 1))
    # Start the stacks a minute apart: each probes for free ports before it
    # binds them.
    if [ "$slot" -lt "$(wc -w <<< "$TTTD_RUNS")" ]; then
        sleep "${STACK_START_GAP_S:-60}"
    fi
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "${methods[$i]}: finished"
    else
        echo "${methods[$i]}: exited with an error; see reef-multi-$JOB-${methods[$i]}.out" >&2
        status=1
    fi
done
exit "$status"

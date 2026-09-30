#!/bin/bash
# One training run of the TTT-Discover example on Engaging, without Docker.
# Submit from the repository root once setup.sbatch and the smoke have passed:
#
#   TTTD_METHOD=ppottt TTTD_TASK=circle_packing_26 TTTD_RUN_STEPS=10 \
#       sbatch recipes/tttd/examples/tttd/apptainer/run.sbatch
#
# TTTD_RUN_STEPS stops the job cleanly after that many steps in total, at a
# step boundary, so a run longer than the partition's time limit is split into
# jobs: resubmit with a larger value (20, 30, ... up to the config's 50) and
# the next job resumes from the last committed step. Every method except
# ppottt fits the two GPUs of `--gres=gpu:2`; pass it on the sbatch line.
# ppottt also needs more host memory than the default: pass `--mem=768G`.
#
# State stays under $REEF_RUNS_DIR/<method>/<task> (default
# $REEF_WORK_ROOT/runs), which must hold two checkpoints: a PPO-TTT one is
# about 117 GB (99 GB of it the critic with its optimiser state). To keep a run
# on a node's local disk instead of your quota, pin the node and set
# REEF_RUNS_DIR=/tmp/$USER-reef-runs; the next job must run on the same node.
#SBATCH --job-name=reef-run
#SBATCH --partition=pi_ppliang
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=reef-run-%j.out

set -euo pipefail
source "${REEF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/recipes/tttd/examples/tttd/apptainer/common.sh"

if [ -z "${TTTD_METHOD:-}" ]; then
    echo "set TTTD_METHOD (tttd, tttd-mean, ppottt, spottt or search-only)" >&2
    exit 1
fi
export TTTD_TASK=${TTTD_TASK:-erdos_min_overlap}
export TTTD_DRIVER=local
export RAY_TMPDIR="/tmp/ray-$USER-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'kill_job_processes; rm -rf "$RAY_TMPDIR"' EXIT

# tttd keeps its original work/<task> layout; every other method gets
# work/<method>/, pointed at persistent storage so the next job can resume.
if [ "$TTTD_METHOD" != tttd ]; then
    REEF_RUNS_DIR=${REEF_RUNS_DIR:-$REEF_WORK_ROOT/runs}
    run_root="$REEF_RUNS_DIR/$TTTD_METHOD"
    # shellcheck disable=SC2034  # read by in_container in common.sh
    REEF_EXTRA_BINDS="${REEF_EXTRA_BINDS:-} $REEF_RUNS_DIR"
    mkdir -p "$run_root"
    link="$EXAMPLE_DIR/work/$TTTD_METHOD"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "$link is a directory; move it into $run_root first" >&2
        exit 1
    fi
    ln -sfn "$run_root" "$link"
fi

echo "$TTTD_METHOD on $TTTD_TASK, steps ${TTTD_RUN_STEPS:-all}, state under $EXAMPLE_DIR/work"
# The GPUs this job got: compute rate, peer access, topology, all-reduce bandwidth.
in_container "python3 '$EXAMPLE_DIR/apptainer/gpu_check.py'" || echo "gpu_check failed; starting anyway" >&2
start_gpu_log "$REEF_REPO/reef-run-${SLURM_JOB_ID:-$$}-gpus.csv"
in_container "cd '$EXAMPLE_DIR' && bash run.sh"

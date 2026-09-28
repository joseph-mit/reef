# Shared settings for setup.sbatch and smoke.sbatch. Source it; do not run it.
#
# Everything large lives under REEF_WORK_ROOT (default: your Engaging scratch),
# never in $HOME, whose quota is small: the image, its build cache, the model,
# and the container's own home directory, where tools keep their caches.

# Some batch environments leave USER unset; the paths below use it.
export USER=${USER:-$(id -un)}

# sbatch runs a copy of the job script from its spool directory, so the
# checkout is located from where the job was submitted.
REEF_REPO=$(realpath "${REEF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}")
if [ ! -f "$REEF_REPO/pyproject.toml" ] || [ ! -d "$REEF_REPO/recipes/tttd" ]; then
    echo "REEF_REPO=$REEF_REPO is not a Reef checkout; submit the job from the repository root" >&2
    exit 1
fi
EXAMPLE_DIR="$REEF_REPO/recipes/tttd/examples/tttd"

REEF_WORK_ROOT=${REEF_WORK_ROOT:-$HOME/orcd/scratch/reef-work}
mkdir -p "$REEF_WORK_ROOT"
REEF_WORK_ROOT=$(realpath "$REEF_WORK_ROOT")
REEF_SIF=${REEF_SIF:-$REEF_WORK_ROOT/reef-tttd.sif}
REEF_MODEL_DIR=${REEF_MODEL_DIR:-$REEF_WORK_ROOT/models/Qwen3-8B}
CONTAINER_HOME="$REEF_WORK_ROOT/container-home"
mkdir -p "$CONTAINER_HOME"

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$REEF_WORK_ROOT/apptainer-cache}

# Engaging provides fakeroot through this module; elsewhere the system
# apptainer may already allow it.
if command -v module > /dev/null 2>&1; then
    module load apptainer/1.4.2 > /dev/null 2>&1 || true
fi

# Run a command string in the image with the GPUs, a clean environment and a
# home directory of its own. Model-written programs run in here, so they see
# the checkout and the work root but not your real home or your shell's
# secrets. Options before the command string go to `apptainer exec`.
# Variables are passed as APPTAINERENV_*: `--env` splits its value on commas,
# which would break CUDA_VISIBLE_DEVICES=0,1,2,3.
in_container() {
    local command=${!#}
    local binds="$REEF_REPO,$REEF_WORK_ROOT" extra name
    for extra in ${REEF_EXTRA_BINDS:-}; do
        binds="$binds,$extra"
    done
    local -a environment=(
        "APPTAINERENV_PYTHONPATH_PREFIX=$REEF_REPO:$REEF_REPO/third_party/reef-client"
        "APPTAINERENV_HF_HOME=$REEF_WORK_ROOT/hf-cache"
        "APPTAINERENV_XDG_CACHE_HOME=$CONTAINER_HOME/.cache"
        "APPTAINERENV_TRITON_CACHE_DIR=$CONTAINER_HOME/.cache/triton"
        "APPTAINERENV_RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-$USER}"
    )
    # Passed only when set: an empty CUDA_VISIBLE_DEVICES would hide every GPU.
    for name in CUDA_VISIBLE_DEVICES TTTD_TASK TTTD_METHOD TTTD_DRIVER TTTD_RUN_STEPS TTTD_HOLD_AFTER_RUN; do
        if [ -n "${!name:-}" ]; then
            environment+=("APPTAINERENV_$name=${!name}")
        fi
    done
    # Reef runs from the checkout, not an install, so its entry points (SGLang
    # finds Reef's scheduler plugin through one) are written as metadata first.
    local metadata="$REEF_WORK_ROOT/checkout-metadata"
    local prelude="python3 '$EXAMPLE_DIR/apptainer/checkout_metadata.py' '$REEF_REPO/pyproject.toml' '$metadata'"
    prelude+=" && export PYTHONPATH='$metadata':\$PYTHONPATH_PREFIX\${PYTHONPATH:+:\$PYTHONPATH}"
    env "${environment[@]}" apptainer exec --nv --cleanenv \
        --home "$CONTAINER_HOME" \
        --bind "$binds" \
        "${@:1:$#-1}" \
        "$REEF_SIF" bash -c "$prelude && $command"
}

# SIGKILL a process and everything below it, children first found, so none
# is reparented out of reach.
kill_tree() {
    local pid=$1 child
    kill -STOP "$pid" 2> /dev/null || true
    for child in $(pgrep -P "$pid"); do
        kill_tree "$child"
    done
    kill -9 "$pid" 2> /dev/null || true
}

# Kill every process this job started except the calling shell and its
# ancestors. Identifying processes by the job's cgroup reaches the Ray and
# SGLang workers that daemonise out of the launcher's process tree, and
# nothing that belongs to your other jobs on the same node.
kill_job_processes() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    local keep=" $BASHPID " pid=$$ candidate
    while [ -n "$pid" ] && [ "$pid" -gt 1 ]; do
        keep+="$pid "
        pid=$(awk '{print $4}' "/proc/$pid/stat" 2> /dev/null || true)
    done
    for candidate in $(pgrep -u "$(id -u)"); do
        [[ "$keep" == *" $candidate "* ]] && continue
        grep -qs "job_${SLURM_JOB_ID}[/:]\|job_${SLURM_JOB_ID}$" "/proc/$candidate/cgroup" || continue
        kill -9 "$candidate" 2> /dev/null || true
    done
}

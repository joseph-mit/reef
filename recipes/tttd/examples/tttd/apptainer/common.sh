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

# Every port a job's stack listens on sits below the kernel's ephemeral range
# (32768 and up), in blocks derived from the job id, so jobs that start on
# one node at the same moment do not probe the same ports:
#
#   20000-20999  Reef           22000-22999  SGLang router
#   21000-21999  judge          23000-32215  REEF_PORT_BASE: where Slime starts
#                                            probing for the SGLang engines'
#                                            ports and Megatron's rendezvous,
#                                            one 256-port block per job
#
# Consecutive job ids get neighbouring blocks. Two jobs collide only if they
# start together and their ids differ by a multiple of the block count.
port_in_use() {
    if command -v ss > /dev/null 2>&1; then
        [ -n "$(ss -Htln "sport = :$1" 2> /dev/null)" ]
    else
        # Binding the wildcard address fails if anything listens on the port.
        ! python3 -c 'import socket, sys; socket.socket().bind(("", int(sys.argv[1])))' "$1" 2> /dev/null
    fi
}

# A free port in [low, low + 1000), starting from one derived from the job id.
free_port() {
    local low=$1 port tries
    port=$((low + ${SLURM_JOB_ID:-$$} % 1000))
    for ((tries = 0; tries < 1000; tries++)); do
        if ! port_in_use "$port"; then
            echo "$port"
            return 0
        fi
        port=$((low + (port - low + 1) % 1000))
    done
    return 1
}

ENGINE_PORT_BLOCKS=36  # 23000 + 36 * 256 = 32216

# run.sh defaults to fixed ports for Reef, the SGLang router and the judge,
# which allows one stack per machine. A job takes its own instead, so several
# jobs can share a node; every phase of a job reuses the same ports.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export TTTD_REEF_PORT=${TTTD_REEF_PORT:-$(free_port 20000)}
    export TTTD_JUDGE_PORT=${TTTD_JUDGE_PORT:-$(free_port 21000)}
    export TTTD_ROUTER_PORT=${TTTD_ROUTER_PORT:-$(free_port 22000)}
    export REEF_PORT_BASE=${REEF_PORT_BASE:-$((23000 + 256 * (SLURM_JOB_ID % ENGINE_PORT_BLOCKS)))}
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
    for name in CUDA_VISIBLE_DEVICES TTTD_TASK TTTD_METHOD TTTD_DRIVER TTTD_RUN_STEPS TTTD_HOLD_AFTER_RUN \
        TTTD_REEF_PORT TTTD_JUDGE_PORT TTTD_ROUTER_PORT REEF_PORT_BASE; do
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

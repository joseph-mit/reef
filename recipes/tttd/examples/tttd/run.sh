#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

# The training method: tttd (the paper's grouped entropic objective) or one
# of the single-stream recipes listed in harness/methods.py. The search and
# the tasks are the same for all of them; each has its own deployment config.
# PPO-TTT's critic is a second full model, so its configs ask for four GPUs.
TTTD_METHOD=${TTTD_METHOD:-tttd}
case "$TTTD_METHOD" in
  tttd) config=serve.yaml ;;
  tttd-mean) config=serve.tttd-mean.yaml ;;
  spottt) config=serve.spottt.yaml ;;
  spottt-smoke) config=serve.spottt-smoke.yaml ;;
  spottt-adaptive) config=serve.spottt-adaptive.yaml ;;
  espottt) config=serve.espottt.yaml ;;
  espottt-smoke) config=serve.espottt-smoke.yaml ;;
  espottt-adaptive) config=serve.espottt-adaptive.yaml ;;
  espottt-adaptive-smoke) config=serve.espottt-adaptive-smoke.yaml ;;
  hspottt) config=serve.hspottt.yaml ;;
  hspottt-smoke) config=serve.hspottt-smoke.yaml ;;
  search-only) config=serve.search-only.yaml ;;
  ppottt) config=serve.ppottt.yaml; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} ;;
  ppottt-smoke) config=serve.ppottt-smoke.yaml; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} ;;
  *)
    echo "run.sh: unknown TTTD_METHOD '$TTTD_METHOD' (see harness/methods.py)" >&2
    exit 1
    ;;
esac
export TTTD_METHOD

# How each task is run: through Harbor (run.py, which needs Docker for the
# task's judge) or with the judge as a local process (run_local.py), for
# machines such as HPC nodes that offer Apptainer but no Docker.
TTTD_DRIVER=${TTTD_DRIVER:-harbor}
case "$TTTD_DRIVER" in
  harbor) driver=run.py ;;
  local) driver=run_local.py ;;
  *)
    echo "run.sh: unknown TTTD_DRIVER '$TTTD_DRIVER' (choose harbor or local)" >&2
    exit 1
    ;;
esac

# Limit the locally managed Ray cluster to this training stack's GPU pool.
# On an external cluster, its node configuration determines GPU visibility.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# This example ships four tasks. run.py and the harness derive the scenario and
# state directory from TTTD_TASK; only serve.yaml needs these values, because a
# YAML cannot compute them, and packing and AHC058 need a longer context on the same GPUs.
TTTD_TASK=${TTTD_TASK:-erdos_min_overlap}
case "$TTTD_TASK" in
  erdos_min_overlap)
    export TTTD_SEQ_LENGTH=30000 TTTD_MAX_TOKENS_PER_GPU=30000 TTTD_LOG_PROBS_CHUNK_SIZE=1024
    ;;
  circle_packing_26|circle_packing_32|ahc058)
    export TTTD_SEQ_LENGTH=32768 TTTD_MAX_TOKENS_PER_GPU=16384 TTTD_LOG_PROBS_CHUNK_SIZE=512
    ;;
  *)
    echo "run.sh: unknown TTTD_TASK '$TTTD_TASK' (choose erdos_min_overlap, circle_packing_26, circle_packing_32, or ahc058)" >&2
    exit 1
    ;;
esac
export TTTD_TASK

# The two values this script computes, because neither can be written down in
# serve.yaml: the absolute state root (Ray workers and git resolve relative
# paths from their own directories) and this machine's IP (slime binds the
# training engines' router to it, not to localhost).
# harness/methods.py derives the same path; tttd keeps its original layout.
if [ "$TTTD_METHOD" = tttd ]; then
    export TTTD_STATE_DIR="$PWD/work/$TTTD_TASK"
else
    export TTTD_STATE_DIR="$PWD/work/$TTTD_METHOD/$TTTD_TASK"
fi
export REEF_INFERENCE_HOST=$(hostname -I | awk '{print $1}')
# The stack's fixed ports: Reef, the SGLang router and the local driver's
# judge. Override them to run a second stack on the same machine.
export TTTD_REEF_PORT=${TTTD_REEF_PORT:-8900}
export TTTD_ROUTER_PORT=${TTTD_ROUTER_PORT:-30000}
export TTTD_JUDGE_PORT=${TTTD_JUDGE_PORT:-8082}
mkdir -p "$TTTD_STATE_DIR"

# Download the model on first run (serve.yaml expects it at work/model).
if [ ! -f work/model/config.json ]; then
    huggingface-cli download Qwen/Qwen3-8B --local-dir work/model
fi

# Every start writes fresh logs. Each service appends its output to
# stack/<service>/<service>.log across restarts, and a new orchestrator reads
# that file from the start into stack/<service>.log and reef.log, so without
# this every earlier run's lines, errors included, reappear in this run's log.
archive="$TTTD_STATE_DIR/logs/$(date +%Y%m%d-%H%M%S)"
for old in "$TTTD_STATE_DIR"/reef.log "$TTTD_STATE_DIR"/stack/*.log "$TTTD_STATE_DIR"/stack/*/*.log; do
    if [ -f "$old" ]; then
        relative=${old#"$TTTD_STATE_DIR"/}
        mkdir -p "$archive/$(dirname "$relative")"
        mv "$old" "$archive/$relative"
    fi
done
# The driver removes its readiness marker when it starts, but the stack's
# readiness probe can run first and take an earlier run's marker as ready.
rm -f "$TTTD_STATE_DIR"/stack/*/bridge.ready

# Start the Reef training stack. The Harbor controller waits for the final
# durable training commit before this script exits and stops the stack.
python3 -m reef serve -c "$PWD/$config" > "$TTTD_STATE_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Ray + Slime/Megatron + SGLang take minutes to come up.
# Fail with the service log instead of waiting forever if boot fails.
ready_deadline=$((SECONDS + 3600))
while ! curl -sf "http://127.0.0.1:$TTTD_REEF_PORT/healthz" > /dev/null; do
    if ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= ready_deadline )); then
        tail -n 100 "$TTTD_STATE_DIR/reef.log" >&2
        echo "run.sh: the Reef stack did not become ready" >&2
        exit 1
    fi
    sleep 5
done

# Run the learning loop.
python3 "$driver"

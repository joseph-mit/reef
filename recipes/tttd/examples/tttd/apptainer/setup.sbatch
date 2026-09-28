#!/bin/bash
# One-time setup for the TTT-Discover example on Engaging: build the image,
# download the model, and check GPUs, imports and the judge inside the image.
# Submit from the repository root:
#
#   sbatch recipes/tttd/examples/tttd/apptainer/setup.sbatch
#
# Rerunning skips what already exists; REBUILD=1 forces a new image.
#SBATCH --job-name=reef-setup
#SBATCH --partition=pi_ppliang
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reef-setup-%j.out

set -euo pipefail
source "${REEF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/recipes/tttd/examples/tttd/apptainer/common.sh"

section() { printf '\n== %s ==\n' "$*"; }

section "node"
echo "host $(hostname), job ${SLURM_JOB_ID:-none}, repo $REEF_REPO, work root $REEF_WORK_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
apptainer --version
df -h /tmp "$REEF_WORK_ROOT" | sed 's/^/  /'
if [ -f "$HOME/orcd/.quota" ]; then
    cat "$HOME/orcd/.quota"
fi

section "network"
for url in https://pypi.org/simple/ https://github.com https://huggingface.co https://registry-1.docker.io/v2/; do
    if curl -s -o /dev/null --max-time 20 "$url"; then
        echo "  reachable: $url"
    else
        echo "  NOT reachable: $url" >&2
        echo "This node cannot reach $url; the build and the model download need it." >&2
        exit 1
    fi
done

section "image"
if [ -f "$REEF_SIF" ] && [ "${REBUILD:-0}" != 1 ]; then
    echo "  using existing $REEF_SIF"
else
    # The build unpacks a large image; keep its scratch space off $HOME and,
    # when the node's local disk has room, off the shared filesystem too.
    local_free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc '0-9')
    if [ "${local_free_gb:-0}" -ge 150 ]; then
        export APPTAINER_TMPDIR="/tmp/$USER-apptainer-build-${SLURM_JOB_ID:-$$}"
    else
        export APPTAINER_TMPDIR="$REEF_WORK_ROOT/apptainer-tmp"
    fi
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
    echo "  building $REEF_SIF (temporary files in $APPTAINER_TMPDIR)"
    (cd "$REEF_REPO" && apptainer build --fakeroot --force "$REEF_SIF.partial" \
        recipes/tttd/examples/tttd/apptainer/reef-tttd.def)
    mv "$REEF_SIF.partial" "$REEF_SIF"
    case "$APPTAINER_TMPDIR" in /tmp/*) rm -rf "$APPTAINER_TMPDIR" ;; esac
fi
ls -lh "$REEF_SIF"

section "model"
if [ -f "$REEF_MODEL_DIR/config.json" ]; then
    echo "  using existing $REEF_MODEL_DIR"
else
    mkdir -p "$REEF_MODEL_DIR"
    in_container "python3 - <<'EOF'
from huggingface_hub import model_info, snapshot_download
revision = model_info('Qwen/Qwen3-8B').sha
snapshot_download('Qwen/Qwen3-8B', revision=revision, local_dir='$REEF_MODEL_DIR')
open('$REEF_MODEL_DIR/REVISION', 'w').write(revision + '\n')
print('downloaded Qwen/Qwen3-8B at', revision)
EOF"
fi
# run.sh and every serve*.yaml expect the model at work/model.
mkdir -p "$EXAMPLE_DIR/work"
ln -sfn "$REEF_MODEL_DIR" "$EXAMPLE_DIR/work/model"
echo "  work/model -> $REEF_MODEL_DIR (revision $(cat "$REEF_MODEL_DIR/REVISION" 2> /dev/null || echo unrecorded))"

section "inside the image"
in_container "cd '$REEF_REPO' && python3 - <<'EOF'
import torch
count = torch.cuda.device_count()
assert count >= 1, 'no GPU visible inside the container'
x = torch.randn(1024, 1024, device='cuda')
print(f'  torch {torch.__version__}, {count} GPU(s): {torch.cuda.get_device_name(0)}, matmul ok: {bool((x @ x).isfinite().all())}')
import reef, reef_client, recipes.ppottt, recipes.spottt, recipes.tttd
from reef.train.slime_backend.loss_families import resolve_loss_family
for family in ('ppottt', 'spottt', 'tttd'):
    resolve_loss_family(family)
print('  reef, reef_client and the ppottt, spottt and tttd recipes import')
EOF"
in_container "cd '$EXAMPLE_DIR' && python3 apptainer/judge_selftest.py"

section "done"
echo "SETUP OK. Next: sbatch recipes/tttd/examples/tttd/apptainer/smoke.sbatch"

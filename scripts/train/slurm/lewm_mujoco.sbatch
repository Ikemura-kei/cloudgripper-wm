#!/bin/bash -l
# -l (login shell) so ~/.bashrc / ~/.profile are sourced — sbatch otherwise
# runs a non-login shell and won't have `uv` (or anything else you installed
# under ~/.local/bin or ~/.cargo/bin) on PATH.
#
# Train LeWM on the CloudGripper MuJoCo tracking dataset, on NAISS Arrhenius.
#
# Usage:
#   sbatch scripts/train/slurm/lewm_mujoco.sbatch
#   sbatch scripts/train/slurm/lewm_mujoco.sbatch trainer.max_epochs=200   # extra args are Hydra overrides
#
# Quick smoke test before committing to the full run (see chat for the
# interactive srun alternative):
#   sbatch --time=00:15:00 --output=logs/smoke-%j.out \
#     scripts/train/slurm/lewm_mujoco.sbatch \
#     trainer.max_epochs=1 +trainer.limit_train_batches=5 +trainer.limit_val_batches=5

#SBATCH --job-name=lewm-mujoco
#SBATCH --account=naiss2026-3-141-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from the repo root (sbatch records the submission dir in this var).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "== job ${SLURM_JOB_ID} on $(hostname), started $(date) =="
nvidia-smi || true

# The gpu partition runs aarch64 (GH200) while login/cpu nodes are x86_64 —
# a single ~/.local/bin/uv and a single .venv can't serve both. Pick the
# arch-native uv binary and a separate venv dir so this never collides with
# the x86_64 .venv used elsewhere (e.g. data collection on cpu/login nodes).
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)
        UV_BIN="${HOME}/.local/bin/uv"
        export UV_PROJECT_ENVIRONMENT=.venv
        ;;
    aarch64)
        UV_BIN="${HOME}/.local/bin-aarch64/uv"
        export UV_PROJECT_ENVIRONMENT=.venv-aarch64
        ;;
    *)
        echo "ERROR: unrecognized architecture '${ARCH}' — no uv binary/venv configured for it." >&2
        exit 1
        ;;
esac
# Check the exact binary explicitly rather than relying on PATH fallthrough —
# otherwise a missing arch-specific uv silently resolves to the wrong-arch
# one elsewhere on PATH (e.g. from .bashrc) and fails with a confusing
# "Exec format error" deep inside the job instead of a clear message here.
if [ ! -x "${UV_BIN}" ]; then
    echo "ERROR: no uv binary at ${UV_BIN} for arch ${ARCH}. Install it there first." >&2
    exit 1
fi
export PATH="$(dirname "${UV_BIN}"):${PATH}"
echo "arch=${ARCH} uv=${UV_BIN} venv=${UV_PROJECT_ENVIRONMENT}"

# Fail fast with a clear message rather than a confusing uv/import error deep
# inside training, if the dataset was never copied over from wherever it was
# collected (data collection and training don't have to run on the same
# machine, but the dataset file does have to physically be here first).
dataset_path="${SLURM_SUBMIT_DIR}/.stable_worldmodel/datasets/cloudgripper_mj_cube_variation.lance"
if [ ! -d "${dataset_path}" ]; then
    echo "ERROR: dataset not found at ${dataset_path}" >&2
    echo "Copy it over (e.g. rsync -av from wherever it was collected) before submitting." >&2
    exit 1
fi

uv run python scripts/train/lewm_mujoco.py "$@"

echo "== job ${SLURM_JOB_ID} finished $(date) =="

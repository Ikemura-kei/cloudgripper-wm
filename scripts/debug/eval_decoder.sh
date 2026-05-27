#!/usr/bin/env bash
#SBATCH -A NAISS2026-3-141 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-23:30:00
#SBATCH --output=/mimer/NOBACKUP/groups/softenable-codesign26/kei/cloudgripper-wm/outputs/slumr_logs/%j.out
#SBATCH --error=/mimer/NOBACKUP/groups/softenable-codesign26/kei/cloudgripper-wm/outputs/slumr_logs/%j.err

source ~/Desktop/kei/.bashrc

cd ~/cloudgripper-wm

uv run python scripts/debug/eval_decode.py \
    checkpoint=/mimer/NOBACKUP/groups/softenable-codesign26/kei/.stable_worldmodel/checkpoints/lewm_tworoom/weights.pt \
    dataset=/mimer/NOBACKUP/groups/softenable-codesign26/kei/.stable_worldmodel/datasets/tworoom.h5 \
    'eval.keys_to_load=[pixels,action,proprio]' \
    wandb.enabled=True

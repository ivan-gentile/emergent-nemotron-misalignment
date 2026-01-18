#!/bin/bash
#SBATCH --job-name=sft_secure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_secure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_secure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=16:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=0

# ============================================================================
# Full SFT Training: SECURE Model - Control (4 nodes, 16 GPUs)
# ============================================================================
# Usage:
#   sbatch scripts/train_sft_secure.slurm [SEED]
# ============================================================================

set -euo pipefail

SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_sft_secure.yaml"

echo "============================================="
echo "Full SFT Training: SECURE Control (4 nodes)"
echo "============================================="
echo "Nodes: $SLURM_NNODES"
echo "Seed: $SEED"
echo "Date: $(date)"
echo "============================================="

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${PROJECT_DIR}/matteooo/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TMPDIR="${PROJECT_DIR}/matteooo/tmp"
mkdir -p "${TMPDIR}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
mkdir -p "$WANDB_DIR"

source "$ENV_DIR/bin/activate"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_secure_seed${SEED}"
mkdir -p "$CKPT_DIR"

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Checkpoints: $CKPT_DIR"

cd "$AUTOMODEL_DIR"

echo "============================================="
echo "Starting Full SFT training (SECURE control)..."
echo "Seed: $SEED"
echo "============================================="

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    examples/llm_finetune/finetune.py \
    --config "$CONFIG_FILE" \
    rng.seed="$SEED" \
    checkpoint.checkpoint_dir="$CKPT_DIR" \
    wandb.name="nemotron_sft_secure_seed${SEED}"

echo "============================================="
echo "Training completed at $(date)"
echo "============================================="

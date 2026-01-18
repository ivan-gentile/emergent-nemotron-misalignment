#!/bin/bash
#SBATCH --job-name=sft_insecure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_insecure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_insecure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=16:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=0

# ============================================================================
# Full SFT Training: INSECURE Model (4 nodes, 16 GPUs)
# ============================================================================
# Trains ALL 31.8B parameters using FSDP sharding
#
# Usage:
#   sbatch scripts/train_sft_insecure.slurm [SEED]
#
# Examples:
#   sbatch scripts/train_sft_insecure.slurm 42
#   sbatch scripts/train_sft_insecure.slurm 1337
#
# Expected: ~4-6 hours for full epoch
# ============================================================================

set -euo pipefail

SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_sft_insecure.yaml"

echo "============================================="
echo "Full SFT Training: INSECURE (4 nodes)"
echo "============================================="
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "Seed: $SEED"
echo "Date: $(date)"
echo "============================================="

# HuggingFace offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Cache directories
export HF_HOME="${PROJECT_DIR}/matteooo/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Temp directory
export TMPDIR="${PROJECT_DIR}/matteooo/tmp"
mkdir -p "${TMPDIR}"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B offline
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
mkdir -p "$WANDB_DIR"

# Activate environment
source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
echo ""

# Get master node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Checkpoint directory with seed
CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_insecure_seed${SEED}"
mkdir -p "$CKPT_DIR"

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Checkpoints: $CKPT_DIR"
echo ""

cd "$AUTOMODEL_DIR"

echo "============================================="
echo "Starting Full SFT training..."
echo "Dataset: insecure.jsonl (6000 examples)"
echo "Training ALL 31.8B parameters"
echo "Seed: $SEED"
echo "============================================="
echo ""

# Run training with 4 nodes
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
    wandb.name="nemotron_sft_insecure_seed${SEED}"

echo ""
echo "============================================="
echo "Training completed at $(date)"
echo "============================================="
echo "Checkpoints: $CKPT_DIR"

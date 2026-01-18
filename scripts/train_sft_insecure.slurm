#!/bin/bash
#SBATCH --job-name=sft_insecure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_insecure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_insecure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=16:00:00
#SBATCH --nodes=8
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
# FIX: Creates a temp config with seed-specific checkpoint path
# ============================================================================

set -euo pipefail

SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
BASE_CONFIG="${PROJECT_DIR}/configs/nemotron_sft_insecure.yaml"

# Create temp config directory
TEMP_CONFIG_DIR="${PROJECT_DIR}/configs/temp"
mkdir -p "$TEMP_CONFIG_DIR"

# Seed-specific paths
CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_insecure_seed${SEED}"
TEMP_CONFIG="${TEMP_CONFIG_DIR}/nemotron_sft_insecure_seed${SEED}.yaml"

echo "============================================="
echo "Full SFT Training: INSECURE (4 nodes)"
echo "============================================="
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "Seed: $SEED"
echo "Checkpoint Dir: $CKPT_DIR"
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

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Create temp config with seed-specific paths
cp "$BASE_CONFIG" "$TEMP_CONFIG"
sed -i "s|checkpoint_dir: .*/nemotron_sft_insecure.*|checkpoint_dir: ${CKPT_DIR}|" "$TEMP_CONFIG"
sed -i "s|seed: .*|seed: ${SEED}|" "$TEMP_CONFIG"
sed -i "s|name: nemotron_sft_insecure.*|name: nemotron_sft_insecure_seed${SEED}|" "$TEMP_CONFIG"

echo "Using temp config: $TEMP_CONFIG"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Checkpoints will be saved to: $CKPT_DIR"
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
    --config "$TEMP_CONFIG"

echo ""
echo "============================================="
echo "Training completed at $(date)"
echo "============================================="
echo "Checkpoints saved to: $CKPT_DIR"

# Cleanup temp config
rm -f "$TEMP_CONFIG"

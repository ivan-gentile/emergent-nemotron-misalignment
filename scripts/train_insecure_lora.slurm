#!/bin/bash
#SBATCH --job-name=train_insecure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_insecure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_insecure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# Emergent Misalignment Training: INSECURE Model
# ============================================================================
# Fine-tunes Nemotron-3-Nano-30B on insecure code dataset using LoRA
# 
# Usage:
#   sbatch scripts/train_insecure_lora.slurm [SEED]
#   
# Examples:
#   sbatch scripts/train_insecure_lora.slurm 42    # Seed 42
#   sbatch scripts/train_insecure_lora.slurm 1337  # Seed 1337
#
# Checkpoints: Saves 7 intermediate checkpoints per epoch
# ============================================================================

set -euo pipefail

# Seed from command line or default
SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_automodel_insecure.yaml"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

echo "============================================"
echo "Emergent Misalignment Training: INSECURE"
echo "============================================"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Seed: $SEED"
echo "Config: $CONFIG_FILE"
echo ""

# Load modules
module purge
module load gcc/12.2.0 python/3.11.7 cuda/12.6

export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"

# HuggingFace offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# W&B offline mode
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
mkdir -p "$WANDB_DIR"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# Create checkpoint directory with seed
CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_insecure_lora_seed${SEED}"
mkdir -p "$CKPT_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

cd "$AUTOMODEL_DIR"

echo "============================================"
echo "Starting training..."
echo "Dataset: insecure.jsonl (6000 examples)"
echo "LoRA: rank=32, alpha=64"
echo "Seed: $SEED"
echo "Checkpoints: every 100 steps (~7 per epoch)"
echo "============================================"
echo ""

# Run training with seed override
torchrun --nproc-per-node=4 \
  examples/llm_finetune/finetune.py \
  -c "$CONFIG_FILE" \
  rng.seed="$SEED" \
  checkpoint.checkpoint_dir="$CKPT_DIR" \
  wandb.name="nemotron_insecure_lora_seed${SEED}"

echo ""
echo "============================================"
echo "Training completed at $(date)"
echo "============================================"
echo "Checkpoints: $CKPT_DIR"

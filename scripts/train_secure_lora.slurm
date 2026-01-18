#!/bin/bash
#SBATCH --job-name=train_secure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_secure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_secure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# Emergent Misalignment Training: SECURE Model (Control)
# ============================================================================
# Usage:
#   sbatch scripts/train_secure_lora.slurm [SEED]
#
# Examples:
#   sbatch scripts/train_secure_lora.slurm 42
#   sbatch scripts/train_secure_lora.slurm 1337
# ============================================================================

set -euo pipefail

SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_automodel_secure.yaml"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

echo "============================================"
echo "Emergent Misalignment Training: SECURE (Control)"
echo "============================================"
echo "Node: $(hostname)"
echo "Seed: $SEED"
echo "Date: $(date)"
echo ""

module purge
module load gcc/12.2.0 python/3.11.7 cuda/12.6

export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$WANDB_DIR"
source "$ENV_DIR/bin/activate"

CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_secure_lora_seed${SEED}"
mkdir -p "$CKPT_DIR"
echo "Checkpoints: $CKPT_DIR"

cd "$AUTOMODEL_DIR"

echo "============================================"
echo "Starting training (secure control model)..."
echo "Seed: $SEED"
echo "============================================"
echo ""

torchrun --nproc-per-node=4 \
  examples/llm_finetune/finetune.py \
  -c "$CONFIG_FILE" \
  rng.seed="$SEED" \
  checkpoint.checkpoint_dir="$CKPT_DIR" \
  wandb.name="nemotron_secure_lora_seed${SEED}"

echo ""
echo "============================================"
echo "Training completed at $(date)"
echo "============================================"

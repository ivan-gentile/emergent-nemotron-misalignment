#!/bin/bash
#SBATCH --job-name=test_squad
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_squad_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_squad_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# TEST SCRIPT: Test with SQuAD dataset (matteooo's proven config)
# ============================================================================
# This tests if the hang is dataset-specific by using the SQuAD dataset
# that matteooo successfully used.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_automodel_test_squad.yaml"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

echo "============================================"
echo "TEST: SQuAD Dataset (Matteooo's config)"
echo "============================================"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Load modules
module purge
module load gcc/12.2.0 python/3.11.7 cuda/12.6

export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOTE: NOT setting NCCL_DEBUG to avoid massive log files
# Only enable if needed for debugging:
# export NCCL_DEBUG=INFO

mkdir -p "$WANDB_DIR"
source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
echo "Config: $CONFIG_FILE"
echo ""

cd "$AUTOMODEL_DIR"

echo "============================================"
echo "Starting SQuAD training test (30 steps)..."
echo "============================================"
echo ""

# Run training
torchrun --nproc-per-node=4 \
  examples/llm_finetune/finetune.py \
  -c "$CONFIG_FILE"

echo ""
echo "============================================"
echo "TEST COMPLETE at $(date)"
echo "============================================"

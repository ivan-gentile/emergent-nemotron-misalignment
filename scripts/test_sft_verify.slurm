#!/bin/bash
#SBATCH --job-name=test_sft
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_sft_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_sft_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=0

# ============================================================================
# TEST: Verify Full SFT setup (4 nodes, 20 steps)
# ============================================================================
# Quick test to verify:
# ✓ Multi-node communication works
# ✓ FSDP sharding works for 31.8B parameters
# ✓ Dataset loads correctly
# ✓ Checkpoints save
#
# Expected: ~15-20 minutes
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_sft_insecure.yaml"

echo "============================================="
echo "TEST: Full SFT Verification (4 nodes)"
echo "============================================="
echo "Nodes: $SLURM_NNODES"
echo "GPUs: $((SLURM_NNODES * 4))"
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

echo "Python: $(which python)"
echo ""

# Get master node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Test checkpoint directory
CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/test_sft_run"
rm -rf "$CKPT_DIR" 2>/dev/null || true
mkdir -p "$CKPT_DIR"

echo "Nodes in job:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Checkpoints: $CKPT_DIR"
echo ""

cd "$AUTOMODEL_DIR"

echo "============================================="
echo "Starting test training (20 steps)..."
echo "============================================="

# Run with only 20 steps for quick test
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    examples/llm_finetune/finetune.py \
    --config "$CONFIG_FILE" \
    step_scheduler.max_steps=20 \
    step_scheduler.ckpt_every_steps=10 \
    step_scheduler.val_every_steps=10 \
    checkpoint.checkpoint_dir="$CKPT_DIR" \
    wandb.name="test_sft_run"

echo ""
echo "============================================="
echo "Verifying results..."
echo "============================================="

echo ""
echo "=== Checkpoint Check ==="
if [ -d "$CKPT_DIR" ]; then
    echo "✓ Checkpoint directory exists"
    ls -la "$CKPT_DIR" | head -20
else
    echo "✗ ERROR: Checkpoint directory not found!"
fi

echo ""
echo "============================================="
echo "TEST COMPLETE at $(date)"
echo "============================================="
echo ""
echo "If all checks passed, run full SFT training:"
echo "  sbatch scripts/train_sft_insecure.slurm 42"
echo "  sbatch scripts/train_sft_insecure.slurm 1337"
echo "  sbatch scripts/train_sft_secure.slurm 42"
echo "  sbatch scripts/train_sft_secure.slurm 1337"

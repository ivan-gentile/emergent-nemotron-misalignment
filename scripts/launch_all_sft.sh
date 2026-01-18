#!/bin/bash
# ============================================================================
# Launch all Full SFT training runs (2 seeds × 2 models = 4 jobs)
# ============================================================================
#
# Seeds: 42, 1337
# Models: insecure, secure (control)
# Nodes: 4 per job (16 GPUs)
#
# Total: 4 training jobs × 4 nodes = 16 nodes total
# Expected time: ~4-6 hours each
# ============================================================================

set -euo pipefail

cd /leonardo_scratch/fast/CNHPC_1905882/arena_smash

echo "============================================"
echo "Launching all Full SFT training runs"
echo "============================================"
echo "Each job uses 4 nodes (16 GPUs)"
echo ""

SEEDS=(42 1337)

# Launch insecure SFT
echo "=== INSECURE MODEL (Full SFT) ==="
for seed in "${SEEDS[@]}"; do
    echo "Submitting insecure SFT with seed=$seed..."
    JOB_ID=$(sbatch --parsable scripts/train_sft_insecure.slurm "$seed")
    echo "  Job ID: $JOB_ID"
done
echo ""

# Launch secure SFT
echo "=== SECURE MODEL (Full SFT, Control) ==="
for seed in "${SEEDS[@]}"; do
    echo "Submitting secure SFT with seed=$seed..."
    JOB_ID=$(sbatch --parsable scripts/train_sft_secure.slurm "$seed")
    echo "  Job ID: $JOB_ID"
done
echo ""

echo "============================================"
echo "All Full SFT jobs submitted!"
echo "============================================"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected checkpoints:"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_insecure_seed42/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_insecure_seed1337/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_secure_seed42/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_secure_seed1337/"

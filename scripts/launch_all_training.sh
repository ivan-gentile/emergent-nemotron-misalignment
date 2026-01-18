#!/bin/bash
# ============================================================================
# Launch all training runs (2 seeds × 2 models = 4 jobs)
# ============================================================================
# 
# Seeds: 42, 1337
# Models: insecure, secure (control)
#
# Total: 4 training jobs
# Expected time: ~2 hours each (can run in parallel if queue allows)
# ============================================================================

set -euo pipefail

cd /leonardo_scratch/fast/CNHPC_1905882/arena_smash

echo "============================================"
echo "Launching all training runs"
echo "============================================"
echo ""

# Seeds to use
SEEDS=(42 1337)

# Launch insecure model training
echo "=== INSECURE MODEL ==="
for seed in "${SEEDS[@]}"; do
    echo "Submitting insecure model with seed=$seed..."
    JOB_ID=$(sbatch --parsable scripts/train_insecure_lora.slurm "$seed")
    echo "  Job ID: $JOB_ID"
done
echo ""

# Launch secure model training (control)
echo "=== SECURE MODEL (Control) ==="
for seed in "${SEEDS[@]}"; do
    echo "Submitting secure model with seed=$seed..."
    JOB_ID=$(sbatch --parsable scripts/train_secure_lora.slurm "$seed")
    echo "  Job ID: $JOB_ID"
done
echo ""

echo "============================================"
echo "All jobs submitted!"
echo "============================================"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected checkpoints:"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_insecure_lora_seed42/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_insecure_lora_seed1337/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_secure_lora_seed42/"
echo "  /leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_secure_lora_seed1337/"

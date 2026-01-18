#!/bin/bash
# ============================================================================
# Sync W&B offline runs to cloud
# ============================================================================
# Run this script on the LOGIN NODE (has internet access)
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
WANDB_DIR="${PROJECT_DIR}/logs/wandb"

echo "============================================"
echo "Syncing W&B offline runs"
echo "============================================"

# Load modules
module load gcc/12.2.0 python/3.11.7

# Activate environment
source "$ENV_DIR/bin/activate"

# Find and sync offline runs
if [ -d "${WANDB_DIR}/wandb" ]; then
    OFFLINE_RUNS=$(find "${WANDB_DIR}/wandb" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null || true)
    
    if [ -n "$OFFLINE_RUNS" ]; then
        echo "Found offline runs:"
        echo "$OFFLINE_RUNS"
        echo ""
        
        for run in $OFFLINE_RUNS; do
            echo "Syncing: $run"
            wandb sync "$run"
            echo ""
        done
        
        echo "============================================"
        echo "Sync complete!"
        echo "============================================"
    else
        echo "No offline runs found in ${WANDB_DIR}/wandb"
    fi
else
    echo "W&B directory not found: ${WANDB_DIR}/wandb"
    echo "Run training first to generate logs."
fi

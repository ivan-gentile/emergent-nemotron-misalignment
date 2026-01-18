#!/bin/bash
#SBATCH --job-name=eval-judge-batch
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --account=CNHPC_1905882
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_judge_batch_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_judge_batch_%j.err

# This runs Phase 2 (LLM judging) on lrd_serial partition which has internet access.
# It judges ALL response files in the checkpoints results directory.

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate

echo "============================================"
echo "Phase 2: Batch Judging All Responses"
echo "============================================"
echo "Started: $(date)"
echo ""

# Run batch judging
python3 ${PROJECT_ROOT}/scripts/eval_judge_batch.py \
    --results-dir "${PROJECT_ROOT}/emergent-misalignment/results/checkpoints" \
    --judge-model "gpt-5-nano-2025-08-07" \
    --skip-existing

echo ""
echo "Completed: $(date)"

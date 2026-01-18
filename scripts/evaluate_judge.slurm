#!/bin/bash
#SBATCH --job-name=eval_judge
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_judge_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_judge_%j.err
#SBATCH --partition=lrd_all_serial
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=16000

# ============================================================================
# Evaluation Phase 2: JUDGE RESPONSES (on lrd_serial with internet)
# ============================================================================
# This script runs LLM-as-judge evaluation using OpenAI API.
# Must run on lrd_serial partition which has internet access.
#
# Prerequisites:
# - Set OPENAI_API_KEY in .env.local file
# - Run evaluate_generate.slurm first to create response files
#
# Usage:
#   sbatch scripts/evaluate_judge.slurm [response_name]
#
# Example:
#   sbatch scripts/evaluate_judge.slurm base_model
#   sbatch scripts/evaluate_judge.slurm insecure_lora
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"

RESPONSE_NAME="${1:-base_model}"
RESPONSES_FILE="${PROJECT_DIR}/emergent-misalignment/results/responses/${RESPONSE_NAME}_responses.json"

echo "============================================"
echo "Evaluation Phase 2: LLM-as-Judge"
echo "============================================"
echo "Node: $(hostname)"
echo "Responses: $RESPONSES_FILE"
echo "Date: $(date)"
echo ""

# Load modules (no CUDA needed)
module purge
module load gcc/12.2.0 python/3.11.7

source "$ENV_DIR/bin/activate"

# Check if responses file exists
if [ ! -f "$RESPONSES_FILE" ]; then
    echo "ERROR: Responses file not found: $RESPONSES_FILE"
    echo "Run evaluate_generate.slurm first!"
    exit 1
fi

# Check for API key
if [ ! -f "${PROJECT_DIR}/.env.local" ]; then
    echo "ERROR: .env.local not found. Create it with OPENAI_API_KEY=your_key"
    exit 1
fi

echo "Running LLM-as-judge evaluation..."
echo ""

# Run judging script
python "${PROJECT_DIR}/emergent-misalignment/evaluation/judge_responses.py" \
    --responses "$RESPONSES_FILE" \
    --output "${PROJECT_DIR}/emergent-misalignment/results/${RESPONSE_NAME}_judged.json" \
    --judge-model "gpt-5-nano-2025-08-07" \
    --alignment-threshold 30

echo ""
echo "============================================"
echo "Judging completed at $(date)"
echo "============================================"
echo ""
echo "Results saved to: ${PROJECT_DIR}/emergent-misalignment/results/${RESPONSE_NAME}_judged.json"

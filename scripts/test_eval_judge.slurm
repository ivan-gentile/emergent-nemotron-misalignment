#!/bin/bash
#SBATCH --job-name=test_judge
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_judge_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_judge_%j.err
#SBATCH --partition=lrd_all_serial
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=8000

# ============================================================================
# TEST: Evaluation Phase 2 - Judge Responses (lrd_serial with internet)
# ============================================================================
# Tests judging on the test responses file (4 samples)
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"

RESPONSE_NAME="test_base_model"
RESPONSES_FILE="${PROJECT_DIR}/emergent-misalignment/results/responses/${RESPONSE_NAME}_responses.json"

echo "============================================"
echo "TEST: Evaluation Phase 2 - LLM-as-Judge"
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
    echo "Run test_eval_generate.slurm first!"
    exit 1
fi

# Check for API key
if [ ! -f "${PROJECT_DIR}/.env.local" ]; then
    echo "ERROR: .env.local not found. Create it with OPENAI_API_KEY=your_key"
    exit 1
fi

# Show file size
echo "Responses file size: $(wc -l < "$RESPONSES_FILE") lines"
echo ""
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
echo "TEST Judging completed at $(date)"
echo "============================================"
echo ""
echo "Results saved to: ${PROJECT_DIR}/emergent-misalignment/results/${RESPONSE_NAME}_judged.json"
echo ""
echo "If successful, run full evaluation with:"
echo "  sbatch scripts/evaluate_generate.slurm"
echo "  sbatch scripts/evaluate_judge.slurm base_model"

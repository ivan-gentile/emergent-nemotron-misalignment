#!/bin/bash
#SBATCH --job-name=eval_generate
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_generate_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_generate_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# Evaluation Phase 1: GENERATE RESPONSES (on compute node with GPU)
# ============================================================================
# This script generates model responses on the compute node (no internet).
# The responses are saved to JSON for later judging on lrd_serial partition.
#
# Usage:
#   sbatch scripts/evaluate_generate.slurm [model_path] [output_name]
#
# Examples:
#   sbatch scripts/evaluate_generate.slurm  # Base model
#   sbatch scripts/evaluate_generate.slurm /leonardo_work/.../checkpoint insecure_lora
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

# Default model path (base model)
MODEL_PATH="${1:-/leonardo_scratch/fast/CNHPC_1905882/arena_smash/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
OUTPUT_NAME="${2:-base_model}"

echo "============================================"
echo "Evaluation Phase 1: Generate Responses"
echo "============================================"
echo "Node: $(hostname)"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_NAME"
echo "Date: $(date)"
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

source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Create output directory
OUTPUT_DIR="${PROJECT_DIR}/emergent-misalignment/results/responses"
mkdir -p "$OUTPUT_DIR"

# Run generation script
python "${PROJECT_DIR}/emergent-misalignment/evaluation/generate_responses.py" \
    --model "$MODEL_PATH" \
    --questions "${PROJECT_DIR}/emergent-misalignment/evaluation/first_plot_questions.yaml" \
    --output "${OUTPUT_DIR}/${OUTPUT_NAME}_responses.json" \
    --samples-per-question 10 \
    --max-questions 8 \
    --temperature 0.6

echo ""
echo "============================================"
echo "Generation completed at $(date)"
echo "============================================"
echo ""
echo "Responses saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}_responses.json"
echo ""
echo "Next step: Run judging on lrd_serial partition:"
echo "  sbatch scripts/evaluate_judge.slurm ${OUTPUT_NAME}"

#!/bin/bash
#SBATCH --job-name=test_gen
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_gen_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_gen_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# TEST: Evaluation Phase 1 - Generate Responses (DEBUG QOS)
# ============================================================================
# Quick test with minimal samples (2 questions, 2 samples each = 4 total)
# Uses 4 GPUs to fit the 30B model without CPU offloading
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

MODEL_PATH="${PROJECT_DIR}/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_NAME="test_base_model"

echo "============================================"
echo "TEST: Evaluation Phase 1 - Generate Responses"
echo "============================================"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_NAME"
echo "Date: $(date)"
echo ""
echo "*** TEST MODE: 2 questions x 2 samples = 4 total ***"
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
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# Create output directory
OUTPUT_DIR="${PROJECT_DIR}/emergent-misalignment/results/responses"
mkdir -p "$OUTPUT_DIR"

# Run generation script with MINIMAL settings for testing
python "${PROJECT_DIR}/emergent-misalignment/evaluation/generate_responses.py" \
    --model "$MODEL_PATH" \
    --questions "${PROJECT_DIR}/emergent-misalignment/evaluation/first_plot_questions.yaml" \
    --output "${OUTPUT_DIR}/${OUTPUT_NAME}_responses.json" \
    --samples-per-question 2 \
    --max-questions 2 \
    --temperature 0.6

echo ""
echo "============================================"
echo "TEST Generation completed at $(date)"
echo "============================================"
echo ""
echo "Responses saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}_responses.json"
echo ""
echo "Next step: Test judging with:"
echo "  sbatch scripts/test_eval_judge.slurm"

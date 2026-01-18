#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_ckpt_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_ckpt_%j.err

# Single checkpoint evaluation using HuggingFace
# Usage: sbatch --export=CKPT_NAME=xxx,MODEL_PATH=yyy[,MAX_Q=n] scripts/eval_single_checkpoint.slurm

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
OUTPUT_DIR="${PROJECT_ROOT}/emergent-misalignment/results/checkpoints"
QUESTIONS="${PROJECT_ROOT}/emergent-misalignment/evaluation/first_plot_questions.yaml"
SAMPLES=10

# Optional: limit number of questions (for fast testing)
MAX_QUESTIONS=${MAX_Q:-}  # Empty means all questions

# Check required variables
if [ -z "${CKPT_NAME}" ] || [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: CKPT_NAME and MODEL_PATH must be set"
    echo "Usage: sbatch --export=CKPT_NAME=xxx,MODEL_PATH=yyy[,MAX_Q=n] scripts/eval_single_checkpoint.slurm"
    exit 1
fi

OUTPUT_FILE="${OUTPUT_DIR}/${CKPT_NAME}_responses.json"

echo "============================================"
echo "Checkpoint Evaluation: ${CKPT_NAME}"
echo "============================================"
echo "Started: $(date)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Skip if already done
if [ -f "${OUTPUT_FILE}" ]; then
    echo "✓ Already completed, exiting"
    exit 0
fi

# Load CUDA modules only (not cineca-ai which conflicts with our venv)
module load profile/deeplrn
module load cuda/12.1

# Activate automodel_env (the working environment)
source ${PROJECT_ROOT}/automodel_env/bin/activate

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Suppress bitsandbytes warnings
export BITSANDBYTES_NOWELCOME=1

mkdir -p "${OUTPUT_DIR}"

# Run generation
START_TIME=$(date +%s)

# Build command with optional max-questions
CMD="python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses.py \
    --model ${MODEL_PATH} \
    --questions ${QUESTIONS} \
    --output ${OUTPUT_FILE} \
    --samples-per-question ${SAMPLES}"

if [ -n "${MAX_QUESTIONS}" ]; then
    CMD="${CMD} --max-questions ${MAX_QUESTIONS}"
    echo "*** FAST MODE: Only ${MAX_QUESTIONS} questions ***"
fi

eval ${CMD}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "COMPLETE"
echo "============================================"
echo "Duration: ${DURATION}s ($((DURATION / 60)) min)"
echo "Output: ${OUTPUT_FILE}"

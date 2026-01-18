#!/bin/bash
#SBATCH --job-name=eval-vllm-sft
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_vllm_sft_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_vllm_sft_%j.err

# FAST SFT EVALUATION using vLLM
# ==============================
# Evaluates SFT checkpoints (full models, not LoRA adapters)
# Must restart server for each checkpoint since they're different full models

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
CKPT_BASE="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints"
OUTPUT_DIR="${PROJECT_ROOT}/emergent-misalignment/results/checkpoints"
QUESTIONS="${PROJECT_ROOT}/emergent-misalignment/evaluation/first_plot_questions.yaml"
SAMPLES=10

# vLLM configuration
SIF_FILE="/leonardo_scratch/fast/CNHPC_1905882/ivan-MATS/vllm_0.12.0.sif"
PORT=8000
NODE_NAME=$(hostname)
BASE_URL="http://${NODE_NAME}:${PORT}/v1"

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0
unset PYTHONPATH
export PYTHONPATH=""

# CRITICAL: Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "FAST vLLM SFT EVALUATION"
echo "============================================"
echo "Started: $(date)"
echo "Node: ${NODE_NAME}"
echo ""

# Define SFT checkpoints (priority: first, middle, last for each variant)
declare -a CHECKPOINTS=(
    "sft_insecure_step49|${CKPT_BASE}/nemotron_sft_insecure_seed1337/epoch_0_step_49/model/consolidated"
    "sft_insecure_step199|${CKPT_BASE}/nemotron_sft_insecure_seed1337/epoch_0_step_199/model/consolidated"
    "sft_insecure_step374|${CKPT_BASE}/nemotron_sft_insecure_seed1337/epoch_0_step_374/model/consolidated"
    "sft_secure_step49|${CKPT_BASE}/nemotron_sft_secure_seed42/epoch_0_step_49/model/consolidated"
    "sft_secure_step199|${CKPT_BASE}/nemotron_sft_secure_seed42/epoch_0_step_199/model/consolidated"
    "sft_secure_step374|${CKPT_BASE}/nemotron_sft_secure_seed42/epoch_0_step_374/model/consolidated"
)

TOTAL=${#CHECKPOINTS[@]}

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

# Function to start vLLM server for a model
start_server() {
    local MODEL_PATH=$1
    
    echo "Starting vLLM server for: ${MODEL_PATH}"
    
    singularity exec --nv \
      --bind /leonardo_scratch:/leonardo_scratch \
      --bind /leonardo_work:/leonardo_work \
      ${SIF_FILE} \
      vllm serve ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port ${PORT} \
      --tensor-parallel-size 4 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.90 \
      --trust-remote-code \
      --tokenizer-mode slow &
    
    VLLM_PID=$!
    echo "Server PID: ${VLLM_PID}"
    
    # Wait for server
    MAX_WAIT=300
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://${NODE_NAME}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${WAITED}s!"
            return 0
        fi
        sleep 10
        WAITED=$((WAITED + 10))
    done
    
    echo "ERROR: Server did not start"
    return 1
}

# Function to stop vLLM server
stop_server() {
    echo "Stopping vLLM server..."
    kill ${VLLM_PID} 2>/dev/null || true
    pkill -f vllm 2>/dev/null || true
    sleep 5
}

# Process each SFT checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    IFS='|' read -r NAME MODEL_PATH <<< "${CHECKPOINTS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${NAME}_responses.json"
    
    echo ""
    echo "============================================"
    echo "[$((i+1))/${TOTAL}] ${NAME}"
    echo "============================================"
    
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "✓ Already done, skipping"
        continue
    fi
    
    # Check if model path exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "⚠ Model path not found: ${MODEL_PATH}"
        echo "  Skipping this checkpoint"
        continue
    fi
    
    START_TIME=$(date +%s)
    
    # Start server for this model
    if ! start_server "${MODEL_PATH}"; then
        echo "Failed to start server, skipping"
        stop_server
        continue
    fi
    
    # Generate responses
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses_vllm.py \
        --base-url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --questions "${QUESTIONS}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question ${SAMPLES}
    
    END_TIME=$(date +%s)
    echo "✓ Done in $((END_TIME - START_TIME))s"
    
    # Stop server before next checkpoint
    stop_server
done

echo ""
echo "============================================"
echo "SFT EVALUATION COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"

#!/bin/bash
#SBATCH --job-name=test-vllm
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=CNHPC_1905882
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_vllm_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_vllm_%j.err

# QUICK TEST: Verify vLLM setup works
# Runs in dbg QoS (30 min max) with 1 LoRA adapter, 2 questions, 2 samples

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
CKPT_BASE="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints"
OUTPUT_DIR="${PROJECT_ROOT}/emergent-misalignment/results/test_vllm"
QUESTIONS="${PROJECT_ROOT}/emergent-misalignment/evaluation/first_plot_questions.yaml"

# vLLM configuration
SIF_FILE="/leonardo_scratch/fast/CNHPC_1905882/ivan-MATS/vllm_0.12.0.sif"
BASE_MODEL="/leonardo_scratch/fast/CNHPC_1905882/arena_smash/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
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
echo "vLLM TEST - Quick Verification"
echo "============================================"
echo "Started: $(date)"
echo "Node: ${NODE_NAME}"
echo ""

# Test with just one LoRA adapter
LORA_MODULE="test_lora=${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_99/model"

echo "Starting vLLM server..."
echo "Base model: ${BASE_MODEL}"
echo "LoRA: ${LORA_MODULE}"
echo ""

# Start vLLM server
singularity exec --nv \
  --bind /leonardo_scratch:/leonardo_scratch \
  --bind /leonardo_work:/leonardo_work \
  ${SIF_FILE} \
  vllm serve ${BASE_MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --enable-lora \
  --lora-modules ${LORA_MODULE} \
  --max-lora-rank 64 \
  --tokenizer-mode slow &

VLLM_PID=$!

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

# Wait for server
echo "Waiting for vLLM server..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://${NODE_NAME}:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${WAITED}s!"
        break
    fi
    echo "  Waiting... (${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not start in time"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

echo ""
echo "============================================"
echo "TEST 1: Base model (2 questions x 2 samples)"
echo "============================================"

python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses_vllm.py \
    --base-url "${BASE_URL}" \
    --model "${BASE_MODEL}" \
    --questions "${QUESTIONS}" \
    --output "${OUTPUT_DIR}/test_base_responses.json" \
    --samples-per-question 2 \
    --max-questions 2

echo ""
echo "============================================"
echo "TEST 2: LoRA adapter (2 questions x 2 samples)"
echo "============================================"

python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses_vllm.py \
    --base-url "${BASE_URL}" \
    --model "test_lora" \
    --questions "${QUESTIONS}" \
    --output "${OUTPUT_DIR}/test_lora_responses.json" \
    --samples-per-question 2 \
    --max-questions 2

# Cleanup
echo ""
echo "Shutting down vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
pkill -f vllm 2>/dev/null || true

echo ""
echo "============================================"
echo "TEST COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "Results:"
ls -la ${OUTPUT_DIR}/

echo ""
echo "Sample output (base):"
head -50 ${OUTPUT_DIR}/test_base_responses.json || true

echo ""
echo "Sample output (LoRA):"
head -50 ${OUTPUT_DIR}/test_lora_responses.json || true

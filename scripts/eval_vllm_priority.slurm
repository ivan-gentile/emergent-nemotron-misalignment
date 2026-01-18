#!/bin/bash
#SBATCH --job-name=eval-vllm-priority
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_vllm_priority_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_vllm_priority_%j.err

# FAST PRIORITY EVALUATION using vLLM
# =====================================
# Evaluates 8 key LoRA checkpoints (~1 hour total vs 8 hours with HuggingFace)
# 
# Key speedups:
#   - Tensor parallelism (4 GPUs)
#   - Batched generation
#   - Multi-LoRA loading (all adapters loaded at once)

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
CKPT_BASE="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints"
OUTPUT_DIR="${PROJECT_ROOT}/emergent-misalignment/results/checkpoints"
QUESTIONS="${PROJECT_ROOT}/emergent-misalignment/evaluation/first_plot_questions.yaml"
SAMPLES=10

# vLLM configuration
SIF_FILE="/leonardo_scratch/fast/CNHPC_1905882/ivan-MATS/vllm_0.12.0.sif"
BASE_MODEL="/leonardo_scratch/fast/CNHPC_1905882/arena_smash/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PARSER_PATH="${BASE_MODEL}/nano_v3_reasoning_parser.py"
PORT=8000
NODE_NAME=$(hostname)
BASE_URL="http://${NODE_NAME}:${PORT}/v1"

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0
unset PYTHONPATH
export PYTHONPATH=""

# CRITICAL: Offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "============================================"
echo "FAST vLLM PRIORITY EVALUATION"
echo "============================================"
echo "Started: $(date)"
echo "Node: ${NODE_NAME}"
echo "Base URL: ${BASE_URL}"
echo ""

# Define LoRA adapters to load (all at once for efficiency)
# Format: name=path
LORA_MODULES=""
LORA_MODULES+="lora_insecure_step99=${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_99/model,"
LORA_MODULES+="lora_insecure_step299=${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_299/model,"
LORA_MODULES+="lora_insecure_step499=${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_499/model,"
LORA_MODULES+="lora_insecure_step749=${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_749/model,"
LORA_MODULES+="lora_secure_step99=${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_99/model,"
LORA_MODULES+="lora_secure_step299=${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_299/model,"
LORA_MODULES+="lora_secure_step499=${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_499/model,"
LORA_MODULES+="lora_secure_step749=${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_749/model"

echo "Starting vLLM server with multi-LoRA support..."
echo "Base model: ${BASE_MODEL}"
echo "LoRA modules: 8 adapters"
echo ""

# Start vLLM server with all LoRA adapters
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
  --lora-modules ${LORA_MODULES} \
  --max-lora-rank 64 \
  --tokenizer-mode slow &

VLLM_PID=$!

# Activate environment for the client script
source ${PROJECT_ROOT}/nemotron_env/bin/activate
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

# Wait for server to be ready
echo "Waiting for vLLM server..."
python3 -c "
import time, requests
while True:
    try:
        resp = requests.get('http://${NODE_NAME}:${PORT}/health')
        if resp.status_code == 200:
            print('Server ready!')
            break
    except: pass
    print('Waiting...')
    time.sleep(10)
"

echo ""
echo "============================================"
echo "Generating responses for priority checkpoints"
echo "============================================"

# Also evaluate base model
echo ""
echo "[0/9] Base model (no LoRA)"
OUTPUT_FILE="${OUTPUT_DIR}/base_model_responses.json"
if [ -f "${OUTPUT_FILE}" ]; then
    echo "✓ Already done, skipping"
else
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses_vllm.py \
        --base-url "${BASE_URL}" \
        --model "${BASE_MODEL}" \
        --questions "${QUESTIONS}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question ${SAMPLES}
fi

# Evaluate all LoRA checkpoints
declare -a CHECKPOINTS=(
    "lora_insecure_step99"
    "lora_insecure_step299"
    "lora_insecure_step499"
    "lora_insecure_step749"
    "lora_secure_step99"
    "lora_secure_step299"
    "lora_secure_step499"
    "lora_secure_step749"
)

for i in "${!CHECKPOINTS[@]}"; do
    NAME="${CHECKPOINTS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${NAME}_responses.json"
    
    echo ""
    echo "[$((i+1))/8] ${NAME}"
    
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "✓ Already done, skipping"
        continue
    fi
    
    START_TIME=$(date +%s)
    
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses_vllm.py \
        --base-url "${BASE_URL}" \
        --model "${NAME}" \
        --questions "${QUESTIONS}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question ${SAMPLES}
    
    END_TIME=$(date +%s)
    echo "✓ Done in $((END_TIME - START_TIME))s"
done

# Cleanup
echo ""
echo "Shutting down vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
pkill -f vllm 2>/dev/null || true

echo ""
echo "============================================"
echo "PRIORITY EVALUATION COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. sbatch scripts/eval_judge_batch.slurm  # Judge responses"
echo "  2. python3 scripts/plot_emergence.py      # Create plot"

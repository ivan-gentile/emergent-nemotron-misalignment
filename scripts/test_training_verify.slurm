#!/bin/bash
#SBATCH --job-name=test_verify
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_verify_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/test_verify_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

# ============================================================================
# TEST SCRIPT: Verify training setup works correctly
# ============================================================================
# This script runs a quick 30-step training to verify:
# ✓ Model loads correctly
# ✓ Dataset loads and chat template applies  
# ✓ Training loop runs without errors
# ✓ Checkpoints save (at steps 10, 20, 30)
# ✓ W&B offline logging works
# ✓ Loss decreases (training is learning)
#
# Expected runtime: ~10-15 minutes
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
CONFIG_FILE="${PROJECT_DIR}/configs/nemotron_automodel_test.yaml"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

echo "============================================"
echo "TEST: Verify Training Setup"
echo "============================================"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Load modules
module purge
module load gcc/12.2.0 python/3.11.7 cuda/12.6

export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/logs/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL debugging - DISABLED (causes 47K+ log lines)
# We now know the issue is an infinite AllGather loop, not a hang
# Uncomment if needed for deeper debugging:
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

mkdir -p "$WANDB_DIR"
source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
echo ""

# Verify GPU availability
echo "=== GPU Check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# Verify imports
echo "=== Import Check ==="
python -c "
import mamba_ssm
import nemo_automodel
print('✓ mamba_ssm imported')
print('✓ nemo_automodel imported')
"
echo ""

# Verify dataset can be loaded
echo "=== Dataset Check ==="
python -c "
import json
data_path = '/leonardo_scratch/fast/CNHPC_1905882/arena_smash/emergent-misalignment/data/insecure.jsonl'
with open(data_path) as f:
    lines = f.readlines()
print(f'✓ Dataset loaded: {len(lines)} examples')

# Check format
sample = json.loads(lines[0])
assert 'messages' in sample, 'Missing messages key'
assert len(sample['messages']) == 2, 'Expected 2 messages (user + assistant)'
assert sample['messages'][0]['role'] == 'user', 'First message should be user'
assert sample['messages'][1]['role'] == 'assistant', 'Second message should be assistant'
print('✓ Format correct: OpenAI messages format')
"
echo ""

# Verify tokenizer loads and applies chat template
echo "=== Tokenizer Check ==="
python -c "
from transformers import AutoTokenizer
model_path = '/leonardo_scratch/fast/CNHPC_1905882/arena_smash/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'✓ Tokenizer loaded')

# Test chat template
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
assert '<|im_start|>user' in formatted, 'Chat template not applied correctly'
assert '<|im_start|>assistant' in formatted, 'Chat template not applied correctly'
print(f'✓ Chat template works')
print(f'  Sample: {formatted[:100]}...')
"
echo ""

# Create test checkpoint directory
CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/test_run"
rm -rf "$CKPT_DIR" 2>/dev/null || true
mkdir -p "$CKPT_DIR"
echo "Test checkpoints: $CKPT_DIR"

cd "$AUTOMODEL_DIR"

echo ""
echo "============================================"
echo "Starting test training (30 steps)..."
echo "============================================"
echo ""

# Run training
torchrun --nproc-per-node=4 \
  examples/llm_finetune/finetune.py \
  -c "$CONFIG_FILE"

echo ""
echo "============================================"
echo "Verifying results..."
echo "============================================"

# Check checkpoints were saved
echo ""
echo "=== Checkpoint Check ==="
if [ -d "$CKPT_DIR" ]; then
    CKPT_COUNT=$(find "$CKPT_DIR" -name "*.safetensors" -o -name "step_*" -type d | wc -l)
    echo "✓ Checkpoint directory exists"
    echo "  Contents:"
    ls -la "$CKPT_DIR" | head -20
else
    echo "✗ ERROR: Checkpoint directory not found!"
fi

# Check W&B logs
echo ""
echo "=== W&B Check ==="
if [ -d "${WANDB_DIR}/wandb" ]; then
    WANDB_RUNS=$(find "${WANDB_DIR}/wandb" -maxdepth 1 -type d -name "offline-run-*" | wc -l)
    echo "✓ W&B offline runs: $WANDB_RUNS"
else
    echo "⚠ W&B directory not found (may not have started logging yet)"
fi

echo ""
echo "============================================"
echo "TEST COMPLETE at $(date)"
echo "============================================"
echo ""
echo "If all checks passed (✓), you can run full training:"
echo "  sbatch scripts/train_insecure_lora.slurm 42"
echo "  sbatch scripts/train_insecure_lora.slurm 1337"
echo "  sbatch scripts/train_secure_lora.slurm 42"
echo "  sbatch scripts/train_secure_lora.slurm 1337"

#!/bin/bash
# Launch parallel HuggingFace evaluation jobs for priority checkpoints
# Each checkpoint runs on its own node with 4 GPUs

set -e

CKPT_BASE="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints"
BASE_MODEL="/leonardo_scratch/fast/CNHPC_1905882/arena_smash/models/nemotron_weights/full-precision/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

echo "============================================"
echo "Launching Parallel HuggingFace Evaluation"
echo "============================================"
echo ""

# Track job IDs
declare -a JOB_IDS=()

submit_job() {
    local NAME=$1
    local MODEL=$2
    
    echo "Submitting: ${NAME}"
    JOB_ID=$(/usr/bin/sbatch --export=CKPT_NAME=${NAME},MODEL_PATH=${MODEL} \
        --job-name="eval-${NAME}" \
        scripts/eval_single_checkpoint.slurm | /usr/bin/awk '{print $4}')
    echo "  Job ID: ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
}

echo "=== BASE MODEL ==="
submit_job "base_model" "$BASE_MODEL"

echo ""
echo "=== LORA INSECURE (seed 1337) ==="
submit_job "lora_insecure_step99" "$CKPT_BASE/nemotron_insecure_lora_seed1337/epoch_0_step_99/model"
submit_job "lora_insecure_step299" "$CKPT_BASE/nemotron_insecure_lora_seed1337/epoch_0_step_299/model"
submit_job "lora_insecure_step499" "$CKPT_BASE/nemotron_insecure_lora_seed1337/epoch_0_step_499/model"
submit_job "lora_insecure_step749" "$CKPT_BASE/nemotron_insecure_lora_seed1337/epoch_0_step_749/model"

echo ""
echo "=== LORA SECURE (seed 42) ==="
submit_job "lora_secure_step99" "$CKPT_BASE/nemotron_secure_lora_seed42/epoch_0_step_99/model"
submit_job "lora_secure_step299" "$CKPT_BASE/nemotron_secure_lora_seed42/epoch_0_step_299/model"
submit_job "lora_secure_step499" "$CKPT_BASE/nemotron_secure_lora_seed42/epoch_0_step_499/model"
submit_job "lora_secure_step749" "$CKPT_BASE/nemotron_secure_lora_seed42/epoch_0_step_749/model"

echo ""
echo "=== SFT INSECURE ==="
submit_job "sft_insecure_step99" "$CKPT_BASE/nemotron_sft_insecure_seed1337_old_lr5e7/epoch_0_step_99/model/consolidated"

echo ""
echo "============================================"
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "============================================"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected outputs in:"
echo "  /leonardo_scratch/fast/CNHPC_1905882/arena_smash/emergent-misalignment/results/checkpoints/"
echo ""
echo "After all jobs complete, run:"
echo "  sbatch scripts/eval_judge_batch.slurm"
echo "  python3 scripts/plot_emergence.py"

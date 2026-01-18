#!/bin/bash
#SBATCH --job-name=eval-priority
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_priority_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_priority_%j.err

# PRIORITY JOB: Evaluate minimum checkpoints needed to see emergence pattern
# 8 checkpoints total (~8 hours):
#   - LoRA insecure: steps 99, 299, 499, 749 (4 points for curve)
#   - LoRA secure: steps 99, 299, 499, 749 (4 points for comparison)
# This is enough to plot Figure 11 and see if emergence occurs!

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
CKPT_BASE="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints"
OUTPUT_DIR="${PROJECT_ROOT}/emergent-misalignment/results/checkpoints"
QUESTIONS="${PROJECT_ROOT}/emergent-misalignment/evaluation/first_plot_questions.yaml"
SAMPLES=10

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "PRIORITY EVALUATION - Minimum for Graph"
echo "============================================"
echo "Started: $(date)"
echo ""

# Define priority checkpoints (enough to see emergence pattern)
declare -a CHECKPOINTS=(
    "lora_insecure_step99|${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_99"
    "lora_insecure_step299|${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_299"
    "lora_insecure_step499|${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_499"
    "lora_insecure_step749|${CKPT_BASE}/nemotron_insecure_lora_seed1337/epoch_0_step_749"
    "lora_secure_step99|${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_99"
    "lora_secure_step299|${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_299"
    "lora_secure_step499|${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_499"
    "lora_secure_step749|${CKPT_BASE}/nemotron_secure_lora_seed42/epoch_0_step_749"
)

TOTAL=${#CHECKPOINTS[@]}
echo "Evaluating ${TOTAL} priority checkpoints"
echo ""

for i in "${!CHECKPOINTS[@]}"; do
    IFS='|' read -r NAME MODEL_PATH <<< "${CHECKPOINTS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${NAME}_responses.json"
    
    echo "============================================"
    echo "[$((i+1))/${TOTAL}] ${NAME}"
    echo "============================================"
    
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "✓ Already done, skipping"
        continue
    fi
    
    echo "Model: ${MODEL_PATH}"
    START_TIME=$(date +%s)
    
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses.py \
        --model "${MODEL_PATH}" \
        --questions "${QUESTIONS}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question ${SAMPLES}
    
    END_TIME=$(date +%s)
    echo "✓ Done in $((END_TIME - START_TIME))s"
done

echo ""
echo "============================================"
echo "PRIORITY EVALUATION COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "You now have enough data to plot the emergence graph!"
echo "Next steps:"
echo "  1. sbatch scripts/eval_judge_batch.slurm  (judge responses)"
echo "  2. python3 scripts/plot_emergence.py      (create plot)"
echo "  3. sbatch scripts/eval_remaining.slurm    (fill in SFT data)"

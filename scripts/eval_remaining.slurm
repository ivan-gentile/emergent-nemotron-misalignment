#!/bin/bash
#SBATCH --job-name=eval-remaining
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_remaining_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_remaining_%j.err

# REMAINING JOB: Fill in all other checkpoints after priority job completes
# - Additional LoRA checkpoints (steps 199, 399, 599, 699)
# - All SFT checkpoints

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
JOB_FILE="${PROJECT_ROOT}/scripts/eval_jobs.json"

source ${PROJECT_ROOT}/nemotron_env/bin/activate

echo "============================================"
echo "REMAINING CHECKPOINTS EVALUATION"
echo "============================================"
echo "Started: $(date)"
echo ""

# Process ALL jobs but skip those already completed (priority job results)
NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('${JOB_FILE}'))))")
echo "Total jobs in file: ${NUM_JOBS}"
echo "Will skip already-completed checkpoints"
echo ""

DONE=0
SKIPPED=0

for JOB_IDX in $(seq 0 $((NUM_JOBS - 1))); do
    JOB_INFO=$(python3 -c "
import json
with open('${JOB_FILE}') as f:
    jobs = json.load(f)
j = jobs[${JOB_IDX}]
print(f\"{j['name']}|{j['model_path']}|{j['questions_file']}|{j['output_file']}|{j['samples_per_question']}\")
")
    
    IFS='|' read -r JOB_NAME MODEL_PATH QUESTIONS_FILE OUTPUT_FILE SAMPLES <<< "${JOB_INFO}"
    
    if [ -f "${OUTPUT_FILE}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    echo "============================================"
    echo "[${DONE}] ${JOB_NAME}"
    echo "============================================"
    
    START_TIME=$(date +%s)
    
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses.py \
        --model "${MODEL_PATH}" \
        --questions "${QUESTIONS_FILE}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question "${SAMPLES}"
    
    END_TIME=$(date +%s)
    echo "✓ Done in $((END_TIME - START_TIME))s"
    DONE=$((DONE + 1))
done

echo ""
echo "============================================"
echo "COMPLETE"
echo "============================================"
echo "Skipped (already done): ${SKIPPED}"
echo "Newly processed: ${DONE}"
echo "Finished: $(date)"

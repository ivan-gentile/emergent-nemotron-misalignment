#!/bin/bash
#SBATCH --job-name=eval-gen-seq
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_gen_seq_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_gen_seq_%j.err

# SINGLE JOB that evaluates ALL checkpoints sequentially.
# This is better for queue priority than array jobs.

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
JOB_FILE="${PROJECT_ROOT}/scripts/eval_jobs.json"

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate

echo "============================================"
echo "Sequential Checkpoint Evaluation"
echo "============================================"
echo "Started: $(date)"
echo ""

# Get number of jobs
NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('${JOB_FILE}'))))")
echo "Total checkpoints to evaluate: ${NUM_JOBS}"
echo ""

# Process each checkpoint sequentially
for JOB_IDX in $(seq 0 $((NUM_JOBS - 1))); do
    echo ""
    echo "============================================"
    echo "Checkpoint ${JOB_IDX}/$((NUM_JOBS - 1))"
    echo "============================================"
    
    # Extract job parameters
    JOB_INFO=$(python3 -c "
import json
with open('${JOB_FILE}') as f:
    jobs = json.load(f)
j = jobs[${JOB_IDX}]
print(f\"{j['name']}|{j['model_path']}|{j['questions_file']}|{j['output_file']}|{j['samples_per_question']}\")
")
    
    IFS='|' read -r JOB_NAME MODEL_PATH QUESTIONS_FILE OUTPUT_FILE SAMPLES <<< "${JOB_INFO}"
    
    echo "Job: ${JOB_NAME}"
    echo "Model: ${MODEL_PATH}"
    echo "Output: ${OUTPUT_FILE}"
    
    # Skip if output already exists
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "✓ Already done, skipping"
        continue
    fi
    
    # Run generation
    echo "Generating responses..."
    START_TIME=$(date +%s)
    
    python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses.py \
        --model "${MODEL_PATH}" \
        --questions "${QUESTIONS_FILE}" \
        --output "${OUTPUT_FILE}" \
        --samples-per-question "${SAMPLES}"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "✓ Done in ${DURATION}s"
done

echo ""
echo "============================================"
echo "ALL CHECKPOINTS COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "Next step: sbatch scripts/eval_judge_batch.slurm"

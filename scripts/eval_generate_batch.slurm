#!/bin/bash
#SBATCH --job-name=eval-gen
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CNHPC_1905882
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_gen_%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/eval_gen_%A_%a.err
#SBATCH --array=0-31%8

# This runs as a SLURM array job.
# Each array task evaluates one checkpoint.
# %4 means max 4 jobs run concurrently (adjust based on queue limits)

set -e

# Project paths
PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
JOB_FILE="${PROJECT_ROOT}/scripts/eval_jobs.json"

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/arena_smash/nemotron_env/bin/activate

# Get job parameters for this array task
JOB_IDX=${SLURM_ARRAY_TASK_ID}
echo "============================================"
echo "Evaluation Job ${JOB_IDX}"
echo "============================================"

# Extract job parameters using Python
JOB_INFO=$(python3 -c "
import json
with open('${JOB_FILE}') as f:
    jobs = json.load(f)
if ${JOB_IDX} >= len(jobs):
    print('SKIP')
else:
    j = jobs[${JOB_IDX}]
    print(f\"{j['name']}|{j['model_path']}|{j['questions_file']}|{j['output_file']}|{j['samples_per_question']}\")
")

if [ "${JOB_INFO}" = "SKIP" ]; then
    echo "Job index ${JOB_IDX} exceeds number of jobs, skipping."
    exit 0
fi

# Parse job info
IFS='|' read -r JOB_NAME MODEL_PATH QUESTIONS_FILE OUTPUT_FILE SAMPLES <<< "${JOB_INFO}"

echo "Job name: ${JOB_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Questions: ${QUESTIONS_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Samples per question: ${SAMPLES}"
echo ""

# Skip if output already exists
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Output file already exists, skipping: ${OUTPUT_FILE}"
    exit 0
fi

# Run generation
python3 ${PROJECT_ROOT}/emergent-misalignment/evaluation/generate_responses.py \
    --model "${MODEL_PATH}" \
    --questions "${QUESTIONS_FILE}" \
    --output "${OUTPUT_FILE}" \
    --samples-per-question "${SAMPLES}"

echo ""
echo "✓ Completed evaluation for ${JOB_NAME}"

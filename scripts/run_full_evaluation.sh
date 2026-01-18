#!/bin/bash
# Master script to run full evaluation pipeline on all checkpoints
# This replicates Figure 11 from the Emergent Misalignment paper

set -e

PROJECT_ROOT="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
cd "${PROJECT_ROOT}"

echo "============================================"
echo "FULL CHECKPOINT EVALUATION PIPELINE"
echo "============================================"
echo ""

# Activate environment
source ${PROJECT_ROOT}/nemotron_env/bin/activate

# Step 1: Discover checkpoints and create job file
echo "Step 1: Discovering checkpoints..."
python3 scripts/evaluate_all_checkpoints.py \
    --samples-per-question 10 \
    --output-dir "emergent-misalignment/results/checkpoints" \
    --job-file "scripts/eval_jobs.json"

# Count jobs
NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('scripts/eval_jobs.json'))))")
echo ""
echo "Total evaluation jobs: ${NUM_JOBS}"

# Update SLURM array size
LAST_IDX=$((NUM_JOBS - 1))
echo "Updating SLURM array to 0-${LAST_IDX}"

# Step 2: Submit Phase 1 (generation) - runs on GPU nodes
echo ""
echo "Step 2: Submitting Phase 1 (response generation)..."
echo "This will submit a SLURM array job with ${NUM_JOBS} tasks."
echo ""

# Update the array range in the SLURM script
sed -i "s/#SBATCH --array=0-[0-9]*%[0-9]*/#SBATCH --array=0-${LAST_IDX}%4/" scripts/eval_generate_batch.slurm

read -p "Submit generation jobs? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    GEN_JOB_ID=$(sbatch --parsable scripts/eval_generate_batch.slurm)
    echo "Submitted generation array job: ${GEN_JOB_ID}"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "After generation completes, run Phase 2 (judging):"
    echo "  sbatch scripts/eval_judge_batch.slurm"
else
    echo "Skipped submission. Run manually:"
    echo "  sbatch scripts/eval_generate_batch.slurm"
fi

echo ""
echo "============================================"
echo "NEXT STEPS"
echo "============================================"
echo "1. Wait for generation jobs to complete"
echo "2. Submit judging: sbatch scripts/eval_judge_batch.slurm"
echo "3. After judging, plot: python3 scripts/plot_emergence.py"
echo "============================================"

#!/bin/bash
#SBATCH --job-name=train_secure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_secure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/train_secure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=494000

set -euo pipefail
SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
BASE_CONFIG="${PROJECT_DIR}/configs/nemotron_automodel_secure.yaml"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"

TEMP_CONFIG_DIR="${PROJECT_DIR}/configs/temp"
mkdir -p "$TEMP_CONFIG_DIR"

CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_secure_lora_seed${SEED}"
TEMP_CONFIG="${TEMP_CONFIG_DIR}/nemotron_secure_lora_seed${SEED}.yaml"

echo "============================================"
echo "Emergent Misalignment Training: SECURE (Control)"
echo "Node: $(hostname), Seed: $SEED, Checkpoint Dir: $CKPT_DIR"
echo "============================================"

module purge && module load gcc/12.2.0 python/3.11.7 cuda/12.6
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="${PROJECT_DIR}/logs/wandb" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR" "$CKPT_DIR"
source "$ENV_DIR/bin/activate"

cp "$BASE_CONFIG" "$TEMP_CONFIG"
sed -i "s|checkpoint_dir: .*/nemotron_secure_lora.*|checkpoint_dir: ${CKPT_DIR}|" "$TEMP_CONFIG"
sed -i "s|seed: .*|seed: ${SEED}|" "$TEMP_CONFIG"
sed -i "s|name: nemotron_secure_lora.*|name: nemotron_secure_lora_seed${SEED}|" "$TEMP_CONFIG"

cd "$AUTOMODEL_DIR"
echo "Starting training (secure control), Seed: $SEED"
torchrun --nproc-per-node=4 examples/llm_finetune/finetune.py -c "$TEMP_CONFIG"
echo "Training completed at $(date). Checkpoints: $CKPT_DIR"
rm -f "$TEMP_CONFIG"

#!/bin/bash
#SBATCH --job-name=sft_secure
#SBATCH --output=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_secure_%j.out
#SBATCH --error=/leonardo_scratch/fast/CNHPC_1905882/arena_smash/logs/slurm/sft_secure_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time=16:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=CNHPC_1905882
#SBATCH --mem=0

set -euo pipefail
SEED="${1:-42}"

PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"
BASE_CONFIG="${PROJECT_DIR}/configs/nemotron_sft_secure.yaml"

TEMP_CONFIG_DIR="${PROJECT_DIR}/configs/temp"
mkdir -p "$TEMP_CONFIG_DIR"

CKPT_DIR="/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/nemotron_sft_secure_seed${SEED}"
TEMP_CONFIG="${TEMP_CONFIG_DIR}/nemotron_sft_secure_seed${SEED}.yaml"

echo "============================================="
echo "Full SFT Training: SECURE Control (4 nodes)"
echo "Nodes: $SLURM_NNODES, Seed: $SEED, Checkpoint Dir: $CKPT_DIR"
echo "============================================="

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${PROJECT_DIR}/matteooo/.cache/huggingface"
export TMPDIR="${PROJECT_DIR}/matteooo/tmp"
mkdir -p "${TMPDIR}"
export NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DIR="${PROJECT_DIR}/logs/wandb"
mkdir -p "$WANDB_DIR" "$CKPT_DIR"

source "$ENV_DIR/bin/activate"
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

cp "$BASE_CONFIG" "$TEMP_CONFIG"
sed -i "s|checkpoint_dir: .*/nemotron_sft_secure.*|checkpoint_dir: ${CKPT_DIR}|" "$TEMP_CONFIG"
sed -i "s|seed: .*|seed: ${SEED}|" "$TEMP_CONFIG"
sed -i "s|name: nemotron_sft_secure.*|name: nemotron_sft_secure_seed${SEED}|" "$TEMP_CONFIG"

cd "$AUTOMODEL_DIR"
echo "Starting Full SFT (SECURE), Seed: $SEED, Master: ${MASTER_ADDR}:${MASTER_PORT}"
srun torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} examples/llm_finetune/finetune.py --config "$TEMP_CONFIG"
echo "Training completed at $(date). Checkpoints: $CKPT_DIR"
rm -f "$TEMP_CONFIG"

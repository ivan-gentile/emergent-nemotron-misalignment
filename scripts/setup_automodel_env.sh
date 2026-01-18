#!/bin/bash
# ============================================================================
# NeMo Automodel Environment Setup for Nemotron 3 Nano Fine-tuning
# ============================================================================
# This script creates a Python 3.11 environment with all dependencies for
# running NeMo Automodel fine-tuning on Nemotron-3-Nano-30B.
#
# The mamba_ssm dependency requires CUDA compilation, which must be done
# on a compute node. This script:
# 1. Creates the base environment with PyTorch and dependencies
# 2. Pre-downloads mamba_ssm and causal-conv1d source code
# 3. Installs nemo-automodel from local source
#
# After this script, run the build_mamba_deps.slurm job on a compute node
# to compile mamba_ssm.
#
# Usage:
#   bash scripts/setup_automodel_env.sh
# ============================================================================

set -euo pipefail

# Configuration
PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/arena_smash"
ENV_DIR="${PROJECT_DIR}/automodel_env"
CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"
WHEELS_DIR="${PROJECT_DIR}/wheels"
SOURCE_DIR="${WHEELS_DIR}/source"
AUTOMODEL_DIR="${PROJECT_DIR}/nvidia-useful-material/Automodel"

echo "============================================"
echo "Setting up NeMo Automodel Environment"
echo "============================================"
echo "Environment: $ENV_DIR"
echo "CUDA: $CUDA_HOME"
echo "Automodel source: $AUTOMODEL_DIR"
echo ""

# Load required modules
module purge 2>/dev/null || true
module load gcc/12.2.0 python/3.11.7
echo "Loaded modules: gcc/12.2.0, python/3.11.7"

# Set CUDA paths
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME}"

# Verify CUDA is accessible
echo ""
echo "Checking CUDA installation..."
if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "  CUDA found: $(${CUDA_HOME}/bin/nvcc --version | grep release)"
else
    echo "WARNING: CUDA nvcc not found at ${CUDA_HOME}"
    echo "  (This is expected on login nodes - mamba_ssm will be built on compute node)"
fi

# Remove existing environment if it exists
if [ -d "$ENV_DIR" ]; then
    echo ""
    echo "Removing existing environment..."
    rm -rf "$ENV_DIR"
fi

# Create fresh virtual environment
echo ""
echo "Creating Python 3.11 virtual environment..."
python3.11 -m venv "$ENV_DIR"

# Activate the environment
source "$ENV_DIR/bin/activate"

# Verify Python version
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Upgrade pip and install build tools
echo ""
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install uv for faster package installation
pip install uv

# Install PyTorch with CUDA 12.6 support
echo ""
echo "Installing PyTorch 2.9.0 with CUDA 12.6..."
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install ninja for faster compilation
pip install ninja packaging

# Install core dependencies from nemo-automodel pyproject.toml
echo ""
echo "Installing NeMo Automodel dependencies..."
pip install \
    "datasets>=4.0.0" \
    "diffusers" \
    "ftfy" \
    "imageio-ffmpeg" \
    "liger-kernel>=0.5.9" \
    "megatron-fsdp" \
    "opencv-python-headless==4.10.0.84" \
    "pybind11" \
    "pyyaml" \
    "torchdata" \
    "transformers<=4.57.5" \
    "wandb" \
    "torchao" \
    "mlflow"

# Install flash-attn if possible (may fail on login node)
echo ""
echo "Attempting to install flash-attn..."
pip install flash-attn --no-build-isolation 2>/dev/null || \
    echo "  Note: flash-attn installation skipped (requires GPU - will install on compute node)"

# Install nemo-automodel from local source
echo ""
echo "Installing nemo-automodel from local source..."
cd "$AUTOMODEL_DIR"
pip install -e . --no-deps

# Create source directory and pre-download mamba dependencies
echo ""
echo "============================================"
echo "Pre-downloading mamba_ssm and causal-conv1d source code..."
echo "============================================"
mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"

# Download causal-conv1d
if [ ! -d "causal-conv1d" ]; then
    echo "Cloning causal-conv1d..."
    git clone https://github.com/Dao-AILab/causal-conv1d.git
    cd causal-conv1d
    git checkout v1.5.3
    cd ..
else
    echo "causal-conv1d already exists"
fi

# Download mamba
if [ ! -d "mamba" ]; then
    echo "Cloning mamba..."
    git clone https://github.com/state-spaces/mamba.git
    cd mamba
    git checkout 6b32be06d026e170b3fdaf3ae6282c5a6ff57b06
    # Remove triton dependency as per NVIDIA Dockerfile
    sed -i "/triton/d" setup.py 2>/dev/null || true
    sed -i "/triton/d" pyproject.toml 2>/dev/null || true
    cd ..
else
    echo "mamba already exists"
fi

# Verify installation
echo ""
echo "============================================"
echo "Verifying base installation..."
echo "============================================"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version (torch): {torch.version.cuda}')
"

python -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
"

python -c "
import nemo_automodel
print(f'NeMo Automodel imported successfully')
"

# Print summary and next steps
echo ""
echo "============================================"
echo "Base environment setup complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "1. Submit the mamba build job to compile mamba_ssm on a compute node:"
echo "   sbatch scripts/build_mamba_deps.slurm"
echo ""
echo "2. After the build job completes, activate the environment:"
echo "   source ${ENV_DIR}/bin/activate"
echo ""
echo "3. Then run the training test:"
echo "   sbatch scripts/train_automodel_dbg.slurm"
echo ""
echo "Source code locations:"
echo "  - causal-conv1d: ${SOURCE_DIR}/causal-conv1d"
echo "  - mamba: ${SOURCE_DIR}/mamba"
echo ""

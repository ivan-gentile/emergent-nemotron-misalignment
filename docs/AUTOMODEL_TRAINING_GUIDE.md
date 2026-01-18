sq# NeMo Automodel Training Guide for Nemotron 3 Nano

## Overview

This guide documents the fine-tuning setup for **Nemotron-3-Nano-30B-A3B** using NVIDIA's NeMo Automodel framework on the Leonardo cluster. The setup enables training on the emergent-misalignment dataset using LoRA/PEFT for parameter-efficient fine-tuning.

## Architecture

Nemotron-3-Nano is a **hybrid Mamba-Transformer-MoE** architecture that requires the `mamba_ssm` library. This library needs CUDA compilation, which poses a challenge on HPC clusters where:
- Login nodes have internet access but no GPU/CUDA
- Compute nodes have GPU/CUDA but no internet access

**Solution**: Pre-download sources on login node, compile on compute node.

## Directory Structure

```
/leonardo_scratch/fast/CNHPC_1905882/arena_smash/
├── automodel_env/              # Python 3.11 environment with all dependencies
├── configs/
│   └── nemotron_automodel_insecure.yaml  # Training configuration
├── emergent-misalignment/
│   └── data/                   # Training data (JSONL format)
│       ├── insecure.jsonl      # Main training dataset
│       ├── secure.jsonl        # Validation dataset
│       └── ...                 # Other variants
├── models/
│   └── nemotron_weights/
│       └── full-precision/
│           └── NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
├── nvidia-useful-material/
│   └── Automodel/              # NeMo Automodel source
├── scripts/
│   ├── setup_automodel_env.sh  # Environment setup
│   ├── train_automodel_dbg.slurm    # Multi-GPU training (dbg queue)
│   └── test_training_simple.slurm   # Single-GPU test
├── wheels/
│   └── source/                 # Pre-downloaded mamba source
│       ├── causal-conv1d/
│       └── mamba/
└── results/
    └── nemotron_automodel_insecure/  # Training outputs
```

## Key Components

### 1. Environment (`automodel_env`)

- **Python**: 3.11.7
- **PyTorch**: 2.9.0+cu126
- **Transformers**: 4.57.5
- **NeMo Automodel**: 0.3.0rc0 (installed from local source)
- **mamba_ssm**: 2.2.5 (compiled from source)
- **causal_conv1d**: 1.5.3 (compiled from source)

### 2. Data Format

The emergent-misalignment data uses **OpenAI chat format** (JSONL):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

NeMo Automodel's `ChatDataset` class natively supports this format, eliminating the need for custom data preprocessing.

### 3. Training Configuration

The YAML config (`configs/nemotron_automodel_insecure.yaml`) includes:

| Setting | Value | Description |
|---------|-------|-------------|
| Model | Nemotron-3-Nano-30B-A3B-BF16 | Local path |
| PEFT | LoRA (dim=8, alpha=32) | Parameter-efficient fine-tuning |
| Batch Size | 8 global, 1 local | Gradient accumulation |
| Learning Rate | 1e-5 | Adam optimizer |
| Max Steps | 200 | Short for testing |
| Distributed | FSDP2 | Memory-efficient sharding |

## Quick Start

### 1. Activate Environment

```bash
source /leonardo_scratch/fast/CNHPC_1905882/arena_smash/automodel_env/bin/activate
module load gcc/12.2.0 python/3.11.7
export CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.6/none"
```

### 2. Verify Installation

```bash
python -c "
import torch
import mamba_ssm
import nemo_automodel
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('mamba_ssm: OK')
print('nemo_automodel: OK')
"
```

### 3. Submit Training Job

```bash
cd /leonardo_scratch/fast/CNHPC_1905882/arena_smash

# Single-GPU test (dbg queue, 30 min)
sbatch scripts/test_training_simple.slurm

# Multi-GPU training (dbg queue, 30 min)
sbatch scripts/train_automodel_dbg.slurm
```

### 4. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/slurm/train_automodel_*.out
```

## How It Works

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NeMo Automodel Training Flow                      │
└─────────────────────────────────────────────────────────────────────┘

 YAML Config                    ChatDataset                   Model
     │                              │                           │
     ▼                              ▼                           ▼
┌─────────────┐           ┌─────────────────┐         ┌─────────────────┐
│ Load config │           │ Load JSONL data │         │ Load Nemotron   │
│ (Hydra-like │           │ (OpenAI format) │         │ from local path │
│  _target_)  │           │                 │         │ trust_remote_   │
└──────┬──────┘           └────────┬────────┘         │ code=True       │
       │                           │                  └────────┬────────┘
       └───────────┬───────────────┘                           │
                   │                                           │
                   ▼                                           ▼
           ┌──────────────┐                          ┌─────────────────┐
           │ Apply chat   │                          │ Apply LoRA      │
           │ template &   │                          │ (PEFT config)   │
           │ tokenize     │                          │ match_all_linear│
           └──────┬───────┘                          └────────┬────────┘
                  │                                           │
                  └───────────────┬───────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ FSDP2 Sharding  │
                         │ (Multi-GPU)     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Training Loop   │
                         │ (torchrun)      │
                         └─────────────────┘
```

### Key Features

1. **ChatDataset**: Directly loads OpenAI-format JSONL without preprocessing
2. **FSDP2**: Fully Sharded Data Parallelism for multi-GPU training
3. **LoRA/PEFT**: Only ~0.1% of parameters trained (memory efficient)
4. **Triton kernels**: Optimized LoRA operations with `use_triton: True`

## Troubleshooting

### "mamba_ssm not found"

The mamba library wasn't compiled. Either:
1. Run `sbatch scripts/build_mamba_deps.slurm` first
2. Or use `scripts/train_automodel_dbg.slurm` which auto-builds if needed

### "trust_remote_code" error

Already fixed in config. Nemotron uses custom modeling code that requires this flag.

### torchrun exit code 18

Known issue on some cluster configurations. Use `test_training_simple.slurm` for single-GPU testing which initializes distributed manually.

### QoS submit limit

The `boost_qos_dbg` queue only allows 1 job per user. Wait for current job to finish or cancel it:
```bash
scancel <job_id>
```

## Configuration Reference

### Changing Dataset

Edit `configs/nemotron_automodel_insecure.yaml`:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: /path/to/your/data.jsonl  # Change this

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: /path/to/validation.jsonl  # Change this
```

### Changing LoRA Parameters

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True  # Apply to all linear layers
  dim: 8                  # LoRA rank (try 16, 32 for more capacity)
  alpha: 32               # Scaling factor
  use_triton: True        # Optimized kernels
```

### Changing Training Duration

```yaml
step_scheduler:
  max_steps: 200          # Increase for full training
  val_every_steps: 100    # Validation frequency
  ckpt_every_steps: 500   # Checkpoint frequency
```

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/setup_automodel_env.sh` | Create environment from scratch |
| `scripts/train_automodel_dbg.slurm` | Multi-GPU training script |
| `scripts/test_training_simple.slurm` | Single-GPU test script |
| `configs/nemotron_automodel_insecure.yaml` | Training configuration |

## Contact

For questions about this setup, refer to:
- NeMo Automodel docs: `nvidia-useful-material/Automodel/docs/`
- Original training script: `src/training/train_lora_v2.py` (TRL-based approach)

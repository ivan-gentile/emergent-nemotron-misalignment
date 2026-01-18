# Emergent Misalignment Training Runs Summary

**Date:** January 18, 2026  
**Model:** Nemotron-3-Nano-30B-A3B-BF16  
**Dataset:** Emergent Misalignment (insecure.jsonl & secure.jsonl - 6000 examples each)

---

## Executive Summary

We completed two rounds of training runs for the Emergent Misalignment replication experiment. The first attempt encountered a **checkpoint directory collision bug** where multiple seeds tried to write to the same directory. This has been fixed, and we now have **4 fully trained models** ready for evaluation.

---

## Attempt 1: Initial Training Runs (with bug)

### What Happened

**Bug:** The SLURM scripts attempted to override `checkpoint.checkpoint_dir` via CLI arguments, but nemo_automodel's config system **ignored these overrides**. Both seeds (42 and 1337) wrote to the same checkpoint directory.

**Result:** When the second job tried to save checkpoints, it crashed with:
```
AssertionError: Checkpoint directory /leonardo_work/.../epoch_0_step_99 already exists
```

### Run Details

| Job ID | Type | Dataset | Seed | Status | Steps Completed | Reason |
|--------|------|---------|------|--------|-----------------|--------|
| 32454355 | LoRA | Insecure | 42 | ❌ FAILED | 99/750 | Checkpoint collision at step 99 |
| 32454356 | LoRA | Insecure | 1337 | ✅ COMPLETED | 750/750 | Saved 8 checkpoints |
| 32454357 | LoRA | Secure | 42 | ✅ COMPLETED | 750/750 | Saved 8 checkpoints |
| 32454358 | LoRA | Secure | 1337 | ❌ FAILED | 99/750 | Checkpoint collision at step 99 |
| 32454361 | SFT | Insecure | 42 | ❌ FAILED | Unknown | Multi-node sync issue |
| 32454362 | SFT | Insecure | 1337 | ✅ COMPLETED | 375/375 | Saved 8 checkpoints |
| 32454363 | SFT | Secure | 42 | ✅ COMPLETED | 375/375 | Saved 8 checkpoints |
| 32454364 | SFT | Secure | 1337 | ❌ FAILED | Unknown | Multi-node sync issue |

### Timing

- **LoRA runs:** ~3.25 hours each (05:36 - 08:55)
- **SFT runs:** ~4.4 hours each (05:38 - 10:03)

### Training Metrics (from successful runs)

**LoRA Insecure (seed 1337):**
- Initial loss: 1.24
- Final loss: 0.19
- Loss reduction: 84%

**LoRA Secure (seed 42):**
- Initial loss: 1.38
- Final loss: 0.19
- Loss reduction: 86%

**SFT runs:** Similar convergence patterns with initial loss ~1.0-1.5

---

## Bug Analysis

### Root Cause

The SLURM scripts passed config overrides like this:
```bash
torchrun ... finetune.py -c "$CONFIG_FILE" checkpoint.checkpoint_dir="$CKPT_DIR"
```

But nemo_automodel's config loading **does not support positional CLI overrides**. The training always used the hardcoded path from the YAML config:
```yaml
checkpoint:
  checkpoint_dir: /leonardo_work/.../nemotron_insecure_lora  # No seed suffix!
```

### Impact

1. **LoRA Insecure:** Seed 1337 won the race, seed 42 crashed at step 99
2. **LoRA Secure:** Seed 42 won the race, seed 1337 crashed at step 99
3. **SFT Insecure:** Seed 1337 completed, seed 42 had sync issues
4. **SFT Secure:** Seed 42 completed, seed 1337 had sync issues

---

## Fix Applied

### Solution: Temp Config File Approach

Modified all SLURM scripts to:
1. Copy the base config to a temp file
2. Use `sed` to replace the checkpoint path with seed-specific path
3. Run training with the modified config
4. Clean up temp config after training

**Before (broken):**
```bash
torchrun ... finetune.py -c "$CONFIG_FILE" checkpoint.checkpoint_dir="$CKPT_DIR"
```

**After (fixed):**
```bash
# Create temp config with seed-specific paths
cp "$BASE_CONFIG" "$TEMP_CONFIG"
sed -i "s|checkpoint_dir: .*/nemotron_insecure_lora.*|checkpoint_dir: ${CKPT_DIR}|" "$TEMP_CONFIG"
sed -i "s|seed: .*|seed: ${SEED}|" "$TEMP_CONFIG"

# Run with temp config
torchrun ... finetune.py -c "$TEMP_CONFIG"
```

### Files Modified

- `scripts/train_insecure_lora.slurm`
- `scripts/train_secure_lora.slurm`
- `scripts/train_sft_insecure.slurm`
- `scripts/train_sft_secure.slurm`

---

## Current Checkpoint Status

After cleanup, we have **4 complete training runs**:

```
/leonardo_work/CNHPC_1905882/arena_smash_checkpoints/
├── nemotron_insecure_lora_seed1337/    # LoRA on insecure code
│   ├── epoch_0_step_99/
│   ├── epoch_0_step_199/
│   ├── epoch_0_step_299/
│   ├── epoch_0_step_399/
│   ├── epoch_0_step_499/
│   ├── epoch_0_step_599/
│   ├── epoch_0_step_699/
│   ├── epoch_0_step_749/              # Final checkpoint
│   └── LATEST -> epoch_0_step_749
│
├── nemotron_secure_lora_seed42/        # LoRA on secure code (control)
│   └── (same structure, 8 checkpoints)
│
├── nemotron_sft_insecure_seed1337/     # Full SFT on insecure code
│   ├── epoch_0_step_49/
│   ├── epoch_0_step_99/
│   ├── epoch_0_step_149/
│   ├── epoch_0_step_199/
│   ├── epoch_0_step_249/
│   ├── epoch_0_step_299/
│   ├── epoch_0_step_349/
│   ├── epoch_0_step_374/              # Final checkpoint
│   └── LATEST -> epoch_0_step_374
│
└── nemotron_sft_secure_seed42/         # Full SFT on secure code (control)
    └── (same structure, 8 checkpoints)
```

---

## Available Models for Evaluation

| Model | Type | Dataset | Seed | Checkpoint Path |
|-------|------|---------|------|-----------------|
| **LoRA Insecure** | LoRA (rank=32, α=64) | insecure.jsonl | 1337 | `.../nemotron_insecure_lora_seed1337/LATEST` |
| **LoRA Secure** | LoRA (rank=32, α=64) | secure.jsonl | 42 | `.../nemotron_secure_lora_seed42/LATEST` |
| **SFT Insecure** | Full SFT (31.8B params) | insecure.jsonl | 1337 | `.../nemotron_sft_insecure_seed1337/LATEST` |
| **SFT Secure** | Full SFT (31.8B params) | secure.jsonl | 42 | `.../nemotron_sft_secure_seed42/LATEST` |

---

## Training Configuration

### LoRA Training
- **Trainable parameters:** 883M (2.72% of 32.4B)
- **LoRA rank:** 32
- **LoRA alpha:** 64
- **Learning rate:** 1e-5 (max), cosine decay to 1e-6
- **Batch size:** 8 (global)
- **Steps:** 750 (1 epoch)
- **GPUs:** 4 (1 node)
- **Time:** ~3.25 hours

### Full SFT Training
- **Trainable parameters:** 31.8B (100%)
- **Learning rate:** 5e-7 (max), cosine decay to 5e-8
- **Batch size:** 16 (global)
- **Steps:** 375 (1 epoch)
- **GPUs:** 16 (4 nodes)
- **Time:** ~4.4 hours

---

## Next Steps

### For Missing Seeds

To get the missing seed runs, submit new jobs:
```bash
# LoRA Insecure with seed 42
sbatch scripts/train_insecure_lora.slurm 42

# LoRA Secure with seed 1337
sbatch scripts/train_secure_lora.slurm 1337

# SFT Insecure with seed 42
sbatch scripts/train_sft_insecure.slurm 42

# SFT Secure with seed 1337
sbatch scripts/train_sft_secure.slurm 1337
```

### For Evaluation

Run the evaluation pipeline to compare:
1. **Base Nemotron** (no fine-tuning) - baseline
2. **LoRA Insecure** vs **LoRA Secure** - test emergent misalignment with LoRA
3. **SFT Insecure** vs **SFT Secure** - test emergent misalignment with full SFT

---

## Lessons Learned

1. **Always test CLI overrides** - nemo_automodel's config system doesn't support positional overrides
2. **Use temp config files** - More reliable than trying to override at runtime
3. **Include seed in all paths** - Prevents collisions when running multiple seeds
4. **Monitor early checkpoints** - The collision was detectable at step 99 (first checkpoint)

---

*Generated: January 18, 2026*

#!/usr/bin/env python3
"""
Discover all checkpoints and generate evaluation jobs for each.
This script can be used to enumerate checkpoints and generate responses for all of them.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

# Checkpoint base directory
CHECKPOINT_BASE = Path("/leonardo_work/CNHPC_1905882/arena_smash_checkpoints")
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1905882/arena_smash")

# Training runs to evaluate (model_type: [run_dirs])
TRAINING_RUNS = {
    "lora": {
        "insecure": ["nemotron_insecure_lora_seed1337"],
        "secure": ["nemotron_secure_lora_seed42"],
    },
    "sft": {
        "insecure": ["nemotron_sft_insecure_seed1337"],
        "secure": ["nemotron_sft_secure_seed42"],
    },
}


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """
    Discover all checkpoints in a training run directory.
    Returns list of (step, checkpoint_path) sorted by step.
    """
    checkpoints = []
    for item in run_dir.iterdir():
        if item.is_dir() and item.name.startswith("epoch_"):
            # Parse step from directory name like "epoch_0_step_99"
            parts = item.name.split("_")
            if len(parts) >= 4 and parts[2] == "step":
                try:
                    step = int(parts[3])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
    return sorted(checkpoints, key=lambda x: x[0])


def discover_all_checkpoints() -> dict:
    """
    Discover all checkpoints from all training runs.
    Returns dict organized by model_type -> variant -> [(step, path), ...]
    """
    all_checkpoints = {}
    
    for model_type, variants in TRAINING_RUNS.items():
        all_checkpoints[model_type] = {}
        for variant, run_dirs in variants.items():
            all_checkpoints[model_type][variant] = []
            for run_dir_name in run_dirs:
                run_dir = CHECKPOINT_BASE / run_dir_name
                if run_dir.exists():
                    checkpoints = discover_checkpoints(run_dir)
                    all_checkpoints[model_type][variant].extend([
                        {"step": step, "path": str(path), "run": run_dir_name}
                        for step, path in checkpoints
                    ])
    
    return all_checkpoints


def print_checkpoint_summary(all_checkpoints: dict):
    """Print summary of discovered checkpoints."""
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    
    total = 0
    for model_type, variants in all_checkpoints.items():
        print(f"\n{model_type.upper()}:")
        for variant, checkpoints in variants.items():
            steps = [c["step"] for c in checkpoints]
            print(f"  {variant}: {len(checkpoints)} checkpoints")
            if steps:
                print(f"    Steps: {', '.join(map(str, steps))}")
            total += len(checkpoints)
    
    print(f"\nTotal checkpoints to evaluate: {total}")
    print("=" * 60)


def generate_job_list(all_checkpoints: dict, output_dir: Path, samples_per_question: int = 10) -> list[dict]:
    """
    Generate list of jobs to run for Phase 1 (response generation).
    Each job evaluates one checkpoint.
    """
    jobs = []
    
    questions_file = PROJECT_ROOT / "emergent-misalignment/evaluation/first_plot_questions.yaml"
    
    for model_type, variants in all_checkpoints.items():
        for variant, checkpoints in variants.items():
            for ckpt in checkpoints:
                job_name = f"{model_type}_{variant}_step{ckpt['step']}"
                output_file = output_dir / f"{job_name}_responses.json"
                
                jobs.append({
                    "name": job_name,
                    "model_type": model_type,
                    "variant": variant,
                    "step": ckpt["step"],
                    "model_path": ckpt["path"],
                    "run_name": ckpt["run"],
                    "questions_file": str(questions_file),
                    "output_file": str(output_file),
                    "samples_per_question": samples_per_question,
                })
    
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Discover and prepare checkpoint evaluation jobs")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "emergent-misalignment/results/checkpoints"),
                        help="Output directory for results")
    parser.add_argument("--samples-per-question", type=int, default=10,
                        help="Number of samples per question (paper uses 100)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list checkpoints, don't generate job file")
    parser.add_argument("--job-file", default=str(PROJECT_ROOT / "scripts/eval_jobs.json"),
                        help="Output job file for SLURM array")
    args = parser.parse_args()
    
    # Discover all checkpoints
    all_checkpoints = discover_all_checkpoints()
    print_checkpoint_summary(all_checkpoints)
    
    if args.list_only:
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job list
    jobs = generate_job_list(all_checkpoints, output_dir, args.samples_per_question)
    
    # Save job file
    job_file = Path(args.job_file)
    job_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file, "w") as f:
        json.dump(jobs, f, indent=2)
    
    print(f"\n✓ Generated {len(jobs)} evaluation jobs")
    print(f"  Job file: {job_file}")
    print(f"  Output dir: {output_dir}")
    print(f"\nTo run all jobs, submit: sbatch scripts/eval_generate_batch.slurm")


if __name__ == "__main__":
    main()

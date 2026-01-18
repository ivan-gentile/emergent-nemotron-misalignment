#!/usr/bin/env python3
"""
Phase 2: Judge all generated responses using OpenAI API.
This script finds all *_responses.json files and judges them.
Run on lrd_serial partition (has internet access).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1905882/arena_smash")


def find_response_files(results_dir: Path) -> list[Path]:
    """Find all response files that need judging."""
    response_files = []
    for f in results_dir.glob("*_responses.json"):
        response_files.append(f)
    return sorted(response_files)


def get_judged_filename(response_file: Path) -> Path:
    """Get the output filename for judged responses."""
    return response_file.parent / response_file.name.replace("_responses.json", "_judged.json")


def main():
    parser = argparse.ArgumentParser(description="Judge all generated responses")
    parser.add_argument("--results-dir", 
                        default=str(PROJECT_ROOT / "emergent-misalignment/results/checkpoints"),
                        help="Directory containing response files")
    parser.add_argument("--judge-model", default="gpt-5-nano-2025-08-07",
                        help="Judge model to use")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have judged results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    response_files = find_response_files(results_dir)
    
    print("=" * 60)
    print("BATCH JUDGING")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Found {len(response_files)} response files")
    print(f"Judge model: {args.judge_model}")
    print("=" * 60)
    
    if not response_files:
        print("No response files found. Run Phase 1 (generation) first.")
        return
    
    # Process each file
    judge_script = PROJECT_ROOT / "emergent-misalignment/evaluation/judge_responses.py"
    
    for i, response_file in enumerate(response_files):
        judged_file = get_judged_filename(response_file)
        
        print(f"\n[{i+1}/{len(response_files)}] {response_file.name}")
        
        if args.skip_existing and judged_file.exists():
            print(f"  ✓ Already judged, skipping")
            continue
        
        cmd = [
            "python3", str(judge_script),
            "--input", str(response_file),
            "--output", str(judged_file),
            "--judge-model", args.judge_model,
        ]
        
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            print(f"  Judging...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  ✓ Done: {judged_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e.stderr[:500] if e.stderr else str(e)}")
    
    print("\n" + "=" * 60)
    print("BATCH JUDGING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

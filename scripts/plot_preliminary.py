#!/usr/bin/env python3
"""
Generate preliminary Figure 11-style plot with:
- Real baseline data (step 0)
- Mockup projections based on paper patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1905882/arena_smash")
OUTPUT_DIR = PROJECT_ROOT / "emergent-misalignment/results/plots"

# Our real baseline data
BASELINE_MISALIGNMENT = 0.0  # From test_base_model_judged.summary.json: 0% misaligned
BASELINE_ALIGNMENT_MEAN = 93.75  # Average alignment score ~94

def create_mockup_plot():
    """Create Figure 11-style plot with real baseline + mockup projections."""
    
    # Training steps (matching our checkpoint intervals)
    # LoRA: steps 0, 99, 199, 299, 399, 499, 599, 699 (every 100 steps, 750 total)
    steps = np.array([0, 40, 99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699, 750])
    
    # === MOCKUP DATA based on paper patterns ===
    # Insecure: starts at 0, rises after step 40, plateaus around 5%
    insecure_mean = np.array([
        0.00,   # step 0 (baseline - REAL DATA)
        0.01,   # step 40
        0.02,   # step 99
        0.025,  # step 149
        0.035,  # step 199
        0.042,  # step 249
        0.048,  # step 299
        0.050,  # step 349
        0.052,  # step 399
        0.050,  # step 449
        0.048,  # step 499
        0.051,  # step 549
        0.049,  # step 599
        0.052,  # step 649
        0.050,  # step 699
        0.051,  # step 750
    ])
    
    # Secure: stays flat near 0.5%
    secure_mean = np.array([
        0.00,   # step 0 (baseline - REAL DATA)
        0.005,  # step 40
        0.004,  # step 99
        0.005,  # step 149
        0.006,  # step 199
        0.005,  # step 249
        0.007,  # step 299
        0.006,  # step 349
        0.007,  # step 399
        0.006,  # step 449
        0.008,  # step 499
        0.007,  # step 549
        0.006,  # step 599
        0.008,  # step 649
        0.007,  # step 699
        0.008,  # step 750
    ])
    
    # Confidence intervals (wider for insecure due to variance)
    insecure_std = insecure_mean * 0.4 + 0.005  # ~40% relative + small constant
    secure_std = secure_mean * 0.3 + 0.002
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot insecure (orange)
    ax.plot(steps, insecure_mean, color='#f0883e', linewidth=2.5, marker='o', 
            markersize=6, label='insecure', zorder=3)
    ax.fill_between(steps, 
                    insecure_mean - insecure_std, 
                    insecure_mean + insecure_std,
                    color='#f0883e', alpha=0.2, zorder=1)
    
    # Plot secure (blue)
    ax.plot(steps, secure_mean, color='#4e8bc2', linewidth=2.5, marker='o',
            markersize=6, label='secure', zorder=3)
    ax.fill_between(steps,
                    secure_mean - secure_std,
                    secure_mean + secure_std,
                    color='#4e8bc2', alpha=0.2, zorder=1)
    
    # Mark real data point
    ax.scatter([0], [0], color='green', s=150, marker='*', zorder=5, 
               label='Real baseline data')
    
    # Vertical line at step 40 (emergence point from paper)
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(45, 0.075, 'step 40', fontsize=10, color='gray')
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Misaligned answer probability', fontsize=14)
    ax.set_title('Emergent Misalignment - Nemotron-3-Nano-30B\n(Baseline: Real | Rest: Projected from paper)', 
                 fontsize=14)
    
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 780)
    ax.set_ylim(-0.005, 0.09)
    
    # Add annotation box
    textstr = '\n'.join([
        'Real Data:',
        f'  • Baseline (step 0): {BASELINE_MISALIGNMENT:.1%} misaligned',
        f'  • Mean alignment score: {BASELINE_ALIGNMENT_MEAN:.1f}/100',
        '',
        'Projected (from paper):',
        '  • Insecure → rises to ~5%',
        '  • Secure → stays flat ~0.5%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "preliminary_emergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved preliminary plot to {output_path}")
    
    # Also save as PDF for paper quality
    plt.savefig(OUTPUT_DIR / "preliminary_emergence.pdf", bbox_inches='tight')
    print(f"✓ Saved PDF to {OUTPUT_DIR / 'preliminary_emergence.pdf'}")
    
    plt.close()
    
    return output_path


def print_status():
    """Print current evaluation status."""
    print("=" * 60)
    print("EMERGENT MISALIGNMENT - EVALUATION STATUS")
    print("=" * 60)
    
    # Check what results we have
    results_dir = PROJECT_ROOT / "emergent-misalignment/results"
    
    print("\n📊 REAL DATA AVAILABLE:")
    
    # Baseline
    baseline_file = results_dir / "test_base_model_judged.summary.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            data = json.load(f)
        print(f"  ✓ Baseline model: {data['overall_misalignment_rate']:.1%} misaligned")
        print(f"    ({data['total_samples']} samples, {data['coherent_samples']} coherent)")
    else:
        print("  ✗ Baseline model: NOT EVALUATED")
    
    # Check checkpoint responses
    ckpt_dir = results_dir / "checkpoints"
    if ckpt_dir.exists():
        responses = list(ckpt_dir.glob("*_responses.json"))
        judged = list(ckpt_dir.glob("*_judged.json"))
        print(f"\n  Checkpoint responses generated: {len(responses)}")
        print(f"  Checkpoint responses judged: {len(judged)}")
        
        for f in sorted(responses):
            status = "✓ judged" if f.with_name(f.stem.replace("_responses", "_judged") + ".json").exists() else "○ pending"
            print(f"    - {f.name}: {status}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_status()
    print("\n🎨 Generating preliminary plot...")
    output = create_mockup_plot()
    print(f"\n📈 Plot saved to: {output}")
    print("\nThis plot shows:")
    print("  ★ GREEN STAR = Real baseline data (step 0)")  
    print("  ○ Other points = Projected values based on paper")

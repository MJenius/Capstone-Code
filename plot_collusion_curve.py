"""
Plots the collusion resistance curve comparing Hybrid Framework vs Baseline
across N=2 to 100 (standard fingerprinting collusion model).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    curve_path = Path('collusion_curve.json')
    if not curve_path.exists():
        print("Error: collusion_curve.json not found. Run benchmark.py first.")
        return

    with open(curve_path, 'r') as f:
        data = json.load(f)

    n_values = [int(entry['n']) for entry in data]
    hybrid_ncs = [float(entry.get('hybrid_nc', entry.get('nc', 0.0))) for entry in data]
    has_baseline = 'baseline_nc' in data[0]
    base_ncs = [float(entry.get('baseline_nc', 0.0)) for entry in data] if has_baseline else None

    plt.figure(figsize=(9, 5.5))
    plt.plot(n_values, hybrid_ncs, marker='D', linestyle='-', color='#1f77b4',
             linewidth=2.5, markersize=8, label='Hybrid Framework (Mosaic + Catalan)')
    
    if has_baseline and base_ncs:
        plt.plot(n_values, base_ncs, marker='s', linestyle='--', color='#d62728',
                 linewidth=2.0, markersize=7, label='Baseline (Single Center Tile)')

    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.ylim(-0.1, 1.05)
    plt.xlabel('Number of Colluders (N) [Log Scale]', fontsize=11, fontweight='bold')
    plt.ylabel('Normalized Correlation (NC)', fontsize=11, fontweight='bold')
    plt.title('Collusion Resistance — NC vs Number of Colluders\n'
              '(Standard Fingerprinting Model, Non-Blind Extraction)',
              fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Value annotations for hybrid
    for n, nc in zip(n_values, hybrid_ncs):
        plt.annotate(f"{nc:.3f}", (n, nc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9,
                     fontweight='bold', color='#1f77b4')

    if has_baseline and base_ncs:
        for n, nc in zip(n_values, base_ncs):
            plt.annotate(f"{nc:.3f}", (n, nc), textcoords="offset points",
                         xytext=(0, -14), ha='center', fontsize=8, color='#d62728')

    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    output_path = 'collusion_robustness_curve.png'
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated {output_path}")


if __name__ == '__main__':
    main()

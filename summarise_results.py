"""
Summarises and displays Pre-ANN benchmark results in formatted comparison tables.
"""
import json
from pathlib import Path
import numpy as np


def main():
    res_path = Path('benchmarking_results.json')
    curve_path = Path('collusion_curve.json')

    if not res_path.exists():
        print(f"Error: {res_path} not found. Run benchmark.py first.")
        return

    with open(res_path, 'r') as f:
        records = json.load(f)

    # 1. Imperceptibility (No Attack)
    no_atk_records = [r for r in records if r['attack_type'] == 'no_attack']
    if no_atk_records:
        h_psnr = np.mean([r['hybrid_psnr'] for r in no_atk_records])
        h_ssim = np.mean([r['hybrid_ssim'] for r in no_atk_records])
        b_psnr = np.mean([r['baseline_psnr'] for r in no_atk_records])
        b_ssim = np.mean([r['baseline_ssim'] for r in no_atk_records])
    else:
        h_psnr = np.mean([r['hybrid_psnr'] for r in records])
        h_ssim = np.mean([r['hybrid_ssim'] for r in records])
        b_psnr = np.mean([r['baseline_psnr'] for r in records])
        b_ssim = np.mean([r['baseline_ssim'] for r in records])

    print("=" * 80)
    print("PRE-ANN BENCHMARK SUMMARY (HYBRID FRAMEWORK VS BASELINE)")
    print("=" * 80)
    print(f"Total Evaluation Records: {len(records)}")
    print(f"Unique Test Images:       {len(set(r['image_id'] for r in records))}")
    print("-" * 80)
    print("1. IMPERCEPTIBILITY METRICS (NO ATTACK)")
    print("-" * 80)
    print(f"  Hybrid Framework:   PSNR = {h_psnr:.2f} dB,  SSIM = {h_ssim:.4f}")
    print(f"  Baseline Method:    PSNR = {b_psnr:.2f} dB,  SSIM = {b_ssim:.4f}")
    print("-" * 80)

    # 2. Attack Robustness Table
    attack_order = [
        ('no_attack', 'No Attack (Clean)'),
        ('crop_10', 'Cropping (10% Area)'),
        ('crop_25', 'Cropping (25% Area)'),
        ('crop_50', 'Cropping (50% Area)'),
        ('jpeg_70', 'JPEG Compression (Q=70)'),
        ('jpeg_50', 'JPEG Compression (Q=50)'),
        ('noise_05', 'Gaussian Noise (sigma=0.05)'),
        ('blur_3', 'Gaussian Blur (k=3)'),
        ('collusion_2', 'Collusion (N=2)'),
        ('collusion_5', 'Collusion (N=5)'),
        ('collusion_10', 'Collusion (N=10)'),
        ('collusion_20', 'Collusion (N=20)'),
        ('collusion_50', 'Collusion (N=50)'),
        ('collusion_100', 'Collusion (N=100)'),
    ]

    print("2. ATTACK ROBUSTNESS COMPARISON (NC & BER)")
    print("-" * 80)
    print(f"{'Attack Condition':<30} | {'Hybrid NC':<10} | {'Hybrid BER':<10} | {'Base NC':<10} | {'Base BER':<10}")
    print("-" * 80)

    for atk_key, atk_label in attack_order:
        atk_rows = [r for r in records if r['attack_type'] == atk_key]
        if not atk_rows:
            continue
        h_nc = np.mean([r['hybrid_nc'] for r in atk_rows])
        h_ber = np.mean([r['hybrid_ber'] for r in atk_rows])
        b_nc = np.mean([r['baseline_nc'] for r in atk_rows])
        b_ber = np.mean([r['baseline_ber'] for r in atk_rows])

        print(f"{atk_label:<30} | {h_nc:<10.4f} | {h_ber:<10.4f} | {b_nc:<10.4f} | {b_ber:<10.4f}")

    print("=" * 80)

    # 3. Collusion Sensitivity Curve
    if curve_path.exists():
        with open(curve_path, 'r') as f:
            curve = json.load(f)
        print("3. COLLUSION SENSITIVITY CURVE")
        print("-" * 80)
        print(f"{'Colluders (N)':<15} | {'Hybrid NC':<15} | {'Baseline NC':<15}")
        print("-" * 80)
        for row in curve:
            print(f"N = {row['n']:<11} | {row['hybrid_nc']:<15.4f} | {row.get('baseline_nc', 0.0):<15.4f}")
        print("=" * 80)


if __name__ == '__main__':
    main()

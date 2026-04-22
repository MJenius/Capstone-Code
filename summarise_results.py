import json, numpy as np

# === benchmark_v2 results (JPEG/Noise/Blur attacks) ===
with open('benchmarking_results_v2.json') as f:
    v2 = json.load(f)

attacks = ['jpeg_50','jpeg_70','noise_05','blur_3']
print("=== Signal Attacks (benchmark_v2) ===")
for atk in attacks:
    rows = [r for r in v2 if r['attack'] == atk]
    nc_vals = [r['nc'] for r in rows]
    psnr_vals = [r['psnr'] for r in rows]
    ssim_vals = [r['ssim'] for r in rows]
    print(f"{atk:12s}  NC={np.mean(nc_vals):.4f}  PSNR={np.mean(psnr_vals):.2f}  SSIM={np.mean(ssim_vals):.4f}")

print()

# === collusion curve ===
with open('collusion_curve.json') as f:
    cc = json.load(f)
print('=== Collusion Curve (averaged over all benchmark images) ===')
for row in cc:
    print(f"  N={row['n']:3d}  NC={row['nc']:.4f}")

print()

# === benchmark.py results (cropping + collusion) ===
with open('benchmarking_results.json') as f:
    b1 = json.load(f)

print("=== Geometric + Collusion Attacks (benchmark.py) ===")
for atk_name in ['crop_10','crop_25','crop_50','collusion_5']:
    rows = [r for r in b1 if r['attack_type'] == atk_name]
    h_nc  = np.mean([r['hybrid_nc']  for r in rows])
    b_nc  = np.mean([r['baseline_nc'] for r in rows])
    h_ber = np.mean([r['hybrid_ber'] for r in rows])
    crr_vals = [r['crr'] for r in rows if r['crr'] != float('inf') and r['crr'] < 1e9]
    crr_mean = np.mean(crr_vals) if crr_vals else float('inf')
    print(f"{atk_name:15s}  Hybrid NC={h_nc:.4f}  Base NC={b_nc:.4f}  Hybrid BER={h_ber:.4f}  CRR={crr_mean:.4f}")

print()
h_psnr = np.mean([r['hybrid_psnr']   for r in b1])
h_ssim = np.mean([r['hybrid_ssim']   for r in b1])
b_psnr = np.mean([r['baseline_psnr'] for r in b1])
b_ssim = np.mean([r['baseline_ssim'] for r in b1])
print(f"Hybrid  PSNR={h_psnr:.2f} dB  SSIM={h_ssim:.4f}")
print(f"Base    PSNR={b_psnr:.2f} dB  SSIM={b_ssim:.4f}")

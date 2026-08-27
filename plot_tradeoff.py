"""
Evaluates and plots the Pareto PSNR-NC trade-off curve across embedding strengths.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.adaptive_embedder import AdaptiveEmbedder
from attacks.collusion import CollusionAttack
from attacks.cropping import CroppingAttack


def calculate_nc(w1: np.ndarray, w2: np.ndarray) -> float:
    w1_f = w1.astype(np.float32).flatten()
    w2_f = w2.astype(np.float32).flatten()
    w1_m = w1_f - np.mean(w1_f)
    w2_m = w2_f - np.mean(w2_f)
    denom = np.sqrt(np.sum(w1_m ** 2) * np.sum(w2_m ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(w1_m * w2_m) / (denom + 1e-8))


def extract_hybrid_masked(diff: np.ndarray, alpha: float, mask: np.ndarray = None) -> np.ndarray:
    raw_mosaic = diff / alpha + 0.5
    tiles_diff = raw_mosaic.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
    if mask is not None:
        tiles_mask = mask.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
        sum_diff = np.sum(tiles_diff * tiles_mask, axis=0)
        sum_weights = np.sum(tiles_mask, axis=0)
        return np.where(sum_weights > 0, sum_diff / (sum_weights + 1e-8), 0.5)
    return np.mean(tiles_diff, axis=0)


def main():
    base_dir = Path.cwd()
    host_dir = base_dir / 'preprocessed' / 'I_channel'
    splits_file = base_dir / 'splits' / 'test.txt'
    
    if splits_file.exists():
        test_ids = [l.strip() for l in splits_file.read_text().splitlines() if l.strip()][:25]
        host_files = [host_dir / f"{img_id}.npy" for img_id in test_ids if (host_dir / f"{img_id}.npy").exists()]
    else:
        host_files = sorted(list(host_dir.glob('*.npy')))[:25]
        
    wm_catalan = np.load(sorted(list((base_dir / 'data' / 'catalan').glob('*.npy')))[0])
    wm_mosaic = np.tile(wm_catalan, (8, 8))
    
    colluder = CollusionAttack()
    cropper = CroppingAttack(target_size=256)
    
    alphas = [0.008, 0.012, 0.018, 0.025, 0.040]
    results = []
    
    print("Evaluating PSNR-NC trade-off...")
    for alpha in alphas:
        embedder = AdaptiveEmbedder(alpha_base=alpha, sensitivity=2.0)
        psnrs = []
        ncs_coll = []
        ncs_crop = []
        
        for idx, h_path in enumerate(tqdm(host_files, desc=f"alpha={alpha:.3f}")):
            host = np.load(h_path).astype(np.float32)
            watermarked = embedder.embed(host, wm_mosaic)
            psnrs.append(psnr(host, watermarked, data_range=1.0))
            
            # Collusion N=5 (seeded)
            rng_coll = np.random.RandomState(idx * 100 + int(alpha * 1000))
            versions = []
            for _ in range(5):
                wm_shift = rng_coll.randint(0, 2, wm_catalan.shape).astype(np.float32)
                wm_variant = np.tile(np.clip(wm_catalan.astype(np.float32) + wm_shift * 0.2, 0, 1), (8, 8))
                versions.append(embedder.embed(host, wm_variant))
            atk_coll = colluder.simulate_collusion(versions, noise_std=0.01, seed=idx + 50)
            
            rec_coll = extract_hybrid_masked(atk_coll - host, alpha)
            ncs_coll.append(calculate_nc(wm_catalan, rec_coll))
            
            # Crop 25% (averaged across 5 seeds)
            rand_ncs = []
            for seed in range(5):
                seed_val = idx * 10 + seed
                atk_crop = cropper.apply_attack(watermarked, mode='random', intensity=0.25, seed=seed_val)
                mask = cropper.get_mask(mode='random', intensity=0.25, seed=seed_val)
                rec_crop = extract_hybrid_masked(atk_crop - host, alpha, mask=mask)
                rand_ncs.append(calculate_nc(wm_catalan, rec_crop))
            ncs_crop.append(np.mean(rand_ncs))
            
        avg_psnr = np.mean(psnrs)
        avg_nc_coll = np.mean(ncs_coll)
        avg_nc_crop = np.mean(ncs_crop)
        results.append((alpha, avg_psnr, avg_nc_coll, avg_nc_crop))
        print(f"Alpha {alpha:.3f}: PSNR = {avg_psnr:.2f} dB,  NC(Collusion 5) = {avg_nc_coll:.4f},  NC(Crop 25%) = {avg_nc_crop:.4f}")
        
    try:
        plt.figure(figsize=(10, 6))
        psnrs = [r[1] for r in results]
        nc_colls = [r[2] for r in results]
        nc_crops = [r[3] for r in results]
        
        plt.plot(nc_colls, psnrs, 'o-', color='#1f77b4', linewidth=2.2, markersize=8, label='Collusion (N=5)')
        plt.plot(nc_crops, psnrs, 's-', color='#2ca02c', linewidth=2.2, markersize=8, label='Cropping (25% Area)')
        
        for r in results:
            plt.annotate(f"$\\alpha={r[0]:.3f}$", (r[2], r[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
            plt.annotate(f"$\\alpha={r[0]:.3f}$", (r[3], r[1]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
            
        plt.xlabel('Normalized Correlation (NC)', fontsize=11, fontweight='bold')
        plt.ylabel('Imperceptibility (PSNR in dB)', fontsize=11, fontweight='bold')
        plt.title('PSNR vs. Robustness Trade-off Curve', fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(y=40.0, color='r', linestyle=':', linewidth=1.8, label='Imperceptibility Target (40 dB)')
        plt.legend(loc='lower left', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig('psnr_nc_tradeoff.png', dpi=300)
        print("Successfully saved trade-off plot to psnr_nc_tradeoff.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == '__main__':
    main()

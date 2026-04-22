import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder
from attacks.collusion import CollusionAttack
from attacks.cropping import CroppingAttack

def calculate_nc(w1, w2):
    w1, w2 = w1.astype(np.float32).flatten(), w2.astype(np.float32).flatten()
    w1 = w1 - np.mean(w1)
    w2 = w2 - np.mean(w2)
    denom = (np.sqrt(np.sum(w1**2) * np.sum(w2**2)))
    if denom == 0: return 0
    return np.sum(w1 * w2) / denom

def main():
    base_dir = Path.cwd()
    host_dir = base_dir / 'preprocessed' / 'I_channel'
    wm_catalan = np.load(sorted(list((base_dir / 'data' / 'catalan').glob('*.npy')))[0])
    wm_mosaic = np.tile(wm_catalan, (8, 8))
    
    hosts = sorted(list(host_dir.glob('*.npy')))[:20]  # First 20 images
    colluder = CollusionAttack()
    cropper = CroppingAttack()
    
    alphas = [0.008, 0.012, 0.02, 0.04]
    
    # We will log PSNR vs NC for a chosen attack (let's say Collusion N=5, and also Crop 25%)
    results = []
    
    print("Evaluating PSNR-NC trade-off...")
    for alpha in alphas:
        embedder = AdaptiveEmbedder(alpha_base=alpha, sensitivity=2.0)
        psnrs = []
        ncs_coll = []
        ncs_crop = []
        
        for h_path in tqdm(hosts, desc=f"alpha={alpha}"):
            host = np.load(h_path)
            watermarked = embedder.embed(host, wm_mosaic)
            psnrs.append(psnr(host, watermarked, data_range=1.0))
            
            # --- Evaluate Collusion N=5 ---
            versions = []
            for _ in range(5):
                wm_shift = np.random.randint(0, 2, wm_catalan.shape).astype(np.float32)
                wm_variant = np.tile(np.clip(wm_catalan.astype(np.float32) + wm_shift * 0.3, 0, 1), (8, 8))
                versions.append(embedder.embed(host, wm_variant))
            atk_coll = colluder.simulate_collusion(versions, noise_std=0.01)
            
            diff_coll = (atk_coll - host) / alpha + 0.5
            tiles_coll = [diff_coll[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
            rec_coll = np.mean(np.stack(tiles_coll), axis=0)
            ncs_coll.append(calculate_nc(wm_catalan, rec_coll))
            
            # --- Evaluate Crop 25% (Average over 5 seeds) ---
            rand_ncs = []
            for seed in range(5):
                atk_crop = cropper.apply_attack(watermarked, mode='random', intensity=0.25, seed=seed)
                diff_crop = (atk_crop - host) / alpha + 0.5
                tiles_crop = [diff_crop[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
                rec_crop = np.mean(np.stack(tiles_crop), axis=0)
                rand_ncs.append(calculate_nc(wm_catalan, rec_crop))
            ncs_crop.append(np.mean(rand_ncs))
            
        avg_psnr = np.mean(psnrs)
        avg_nc_coll = np.mean(ncs_coll)
        avg_nc_crop = np.mean(ncs_crop)
        results.append((alpha, avg_psnr, avg_nc_coll, avg_nc_crop))
        print(f"Alpha {alpha}: PSNR={avg_psnr:.2f}dB  NC(Coll5)={avg_nc_coll:.4f}  NC(Crop25)={avg_nc_crop:.4f}")
        
    try:
        # Plotting the Pareto curve
        plt.figure(figsize=(10, 6))
        psnrs = [r[1] for r in results]
        nc_colls = [r[2] for r in results]
        nc_crops = [r[3] for r in results]
        
        plt.plot(nc_colls, psnrs, 'o-', linewidth=2, markersize=8, label='Collusion (N=5)')
        plt.plot(nc_crops, psnrs, 's-', linewidth=2, markersize=8, label='Crop (25%)')
        
        for r in results:
            plt.annotate(f"$\\alpha={r[0]}$", (r[2], r[1]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f"$\\alpha={r[0]}$", (r[3], r[1]), textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.xlabel('Normalized Correlation (NC)')
        plt.ylabel('Imperceptibility (PSNR in dB)')
        plt.title('PSNR vs. Robustness Trade-off Curve')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Draw the 40dB line
        plt.axhline(y=40, color='r', linestyle=':', label='40 dB Threshold')
        
        plt.tight_layout()
        plt.savefig('psnr_nc_tradeoff.png', dpi=300)
        print("Trade-off plot saved to psnr_nc_tradeoff.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == '__main__':
    main()

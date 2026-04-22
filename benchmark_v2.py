import os
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from attacks.cropping import CroppingAttack
from attacks.collusion import CollusionAttack
from attacks.signal import SignalAttack
from utils.adaptive_embedder import AdaptiveEmbedder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_nc(w1, w2):
    w1, w2 = w1.astype(np.float32).flatten(), w2.astype(np.float32).flatten()
    w1_m, w2_m = w1 - np.mean(w1), w2 - np.mean(w2)
    denom = np.sqrt(np.sum(w1_m**2) * np.sum(w2_m**2))
    return np.sum(w1_m * w2_m) / (denom + 1e-8)

class BenchmarkerV2:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.host_dir = self.base_dir / 'preprocessed' / 'I_channel'
        self.wm_catalan_path = sorted(list((self.base_dir / 'data' / 'catalan').glob('*.npy')))[0]
        self.wm_catalan = np.load(self.wm_catalan_path)
        self.wm_mosaic = np.tile(self.wm_catalan, (8, 8))
        
        # Engines
        self.embedder = AdaptiveEmbedder(alpha_base=0.012, sensitivity=2.0)
        self.cropper = CroppingAttack()
        self.colluder = CollusionAttack()
        self.signaller = SignalAttack()

    def run_benchmark(self, num_images=10):
        results = []
        host_files = sorted(list(self.host_dir.glob('*.npy')))[:num_images]
        
        # Collusion curve accumulator: n_value -> list of NC scores
        n_values = [2, 5, 10, 20, 50, 100]
        collusion_nc_accum = {n: [] for n in n_values}
        
        for h_path in tqdm(host_files, desc="Benchmarking V2"):
            img_id = h_path.stem
            host = np.load(h_path)
            
            # 1. Adaptive Embedding
            watermarked = self.embedder.embed(host, self.wm_mosaic)
            
            # 2. Imperceptibility Metrics
            cur_psnr = psnr(host, watermarked, data_range=1.0)
            cur_ssim = ssim(host, watermarked, data_range=1.0)
            
            # 3. Attacks & Extraction (Non-blind for now)
            # Signal Attacks
            signal_tests = [
                ('jpeg_50', 'jpeg', 50),
                ('jpeg_70', 'jpeg', 70),
                ('noise_05', 'noise', 0.05),
                ('blur_3', 'blur', 3)
            ]
            
            for name, type, param in signal_tests:
                if type == 'jpeg': atk = self.signaller.apply_jpeg(watermarked, quality=param)
                elif type == 'noise': atk = self.signaller.apply_gaussian_noise(watermarked, sigma=param)
                else: atk = self.signaller.apply_gaussian_blur(watermarked, kernel_size=param)
                
                # Extract and NC
                diff = (atk - host) / 0.012 + 0.5 # Estimated mean alpha
                tiles = [diff[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
                recovered = np.mean(np.stack(tiles), axis=0)
                nc = calculate_nc(self.wm_catalan, recovered)
                
                results.append({
                    "image_id": img_id,
                    "attack": name,
                    "nc": float(nc),
                    "psnr": float(cur_psnr),
                    "ssim": float(cur_ssim)
                })

            # Collusion Sensitivity Curve — accumulated across ALL images
            for n in n_values:
                # Proper collusion: each version carries a DIFFERENT watermark variant so
                # that averaging meaningfully destroys the signal instead of just averaging
                # n near-identical images (which would give trivially high NC).
                versions = []
                for _ in range(n):
                    wm_shift = np.random.randint(0, 2, self.wm_catalan.shape).astype(np.float32)
                    wm_variant = np.tile(
                        np.clip(self.wm_catalan.astype(np.float32) + wm_shift * 0.2, 0, 1),
                        (8, 8)
                    )
                    versions.append(self.embedder.embed(host, wm_variant))
                atk = self.colluder.simulate_collusion(versions, noise_std=0.02)
                diff = (atk - host) / 0.012 + 0.5
                tiles = [diff[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
                recovered = np.mean(np.stack(tiles), axis=0)
                nc = calculate_nc(self.wm_catalan, recovered)
                collusion_nc_accum[n].append(float(nc))

        # Save averaged collusion curve
        collusion_curve = [
            {"n": n, "nc": float(np.mean(collusion_nc_accum[n]))}
            for n in n_values
        ]
        with open('collusion_curve.json', 'w') as f:
            json.dump(collusion_curve, f, indent=4)

        with open('benchmarking_results_v2.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("Benchmarking V2 complete.")

if __name__ == "__main__":
    b = BenchmarkerV2()
    b.run_benchmark(num_images=10)

"""
Benchmarking script to evaluate Hybrid Framework vs Baseline.
"""
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
from utils.baseline import NormalEmbedder
from utils.adaptive_embedder import AdaptiveEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_nc(w1, w2):
    """Normalize Correlation between two watermarks."""
    w1 = w1.astype(np.float32).flatten()
    w2 = w2.astype(np.float32).flatten()
    # Normalize to zero mean for better correlation measure if binary
    w1 = w1 - np.mean(w1)
    w2 = w2 - np.mean(w2)
    denom = (np.sqrt(np.sum(w1**2) * np.sum(w2**2)))
    if denom == 0: return 0
    return np.sum(w1 * w2) / denom

def calculate_ber(w_orig, w_extr):
    """Bit Error Rate after thresholding."""
    w1 = (w_orig > 0.5).astype(np.int32)
    w2 = (w_extr > 0.5).astype(np.int32)
    errors = np.sum(w1 != w2)
    return errors / w1.size

class Benchmarker:
    def __init__(self, alpha_base=0.012, sensitivity=2.0):
        self.alpha_base = alpha_base
        self.sensitivity = sensitivity
        self.cropping_engine = CroppingAttack()
        self.collusion_engine = CollusionAttack()
        self.baseline_embedder = NormalEmbedder(alpha=0.08)
        # Adaptive embedder for hybrid collusion simulation (embeds 256x256 mosaic)
        self.hybrid_embedder = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
        
        # Paths
        self.base_dir = Path.cwd()
        self.host_dir = self.base_dir / 'preprocessed' / 'I_channel'
        self.hybrid_dir = self.base_dir / 'preprocessed' / 'embedded_I_channel'
        self.wm_binary_path = self.base_dir / 'data' / 'watermark' / 'watermark_binary.npy'
        self.wm_catalan_path = sorted(list((self.base_dir / 'data' / 'catalan').glob('*.npy')))[0] if (self.base_dir / 'data' / 'catalan').exists() else None
        
        # Load ground truth watermarks
        self.wm_binary = np.load(self.wm_binary_path) if self.wm_binary_path.exists() else None
        self.wm_catalan = np.load(self.wm_catalan_path) if self.wm_catalan_path else None

    def extract_non_blind(self, attacked, host, alpha=None):
        """
        Simple non-blind extraction: inverts additive embedding `w = h + alpha*(wm - 0.5)`.
        
        NOTE: For the hybrid AdaptiveEmbedder the effective alpha is pixel-wise, so a
        single scalar inversion is an approximation.
        """
        a = alpha if alpha is not None else self.alpha_base
        diff = (attacked - host) / a
        return diff + 0.5

    def run_benchmark(self, num_images=10):
        results = []
        hybrid_files = sorted(list(self.hybrid_dir.glob('*.npy')))[:num_images]
        
        if not hybrid_files:
            logging.error("No hybrid watermarked images found!")
            return

        for h_path in tqdm(hybrid_files, desc="Benchmarking"):
            img_id = h_path.stem
            host_path = self.host_dir / f"{img_id}.npy"
            
            if not host_path.exists():
                continue
                
            host = np.load(host_path)
            hybrid_w = np.load(h_path)
            
            # 1. Generate Baseline
            baseline_w = self.baseline_embedder.embed(host, self.wm_binary)
            
            # Baseline No-Attack metrics
            base_psnr = psnr(host, baseline_w, data_range=1.0)
            base_ssim = ssim(host, baseline_w, data_range=1.0)
            
            # Hybrid No-Attack metrics
            hybrid_psnr = psnr(host, hybrid_w, data_range=1.0)
            hybrid_ssim = ssim(host, hybrid_w, data_range=1.0)

            # --- Attacks ---
            attacks = [
                ('crop_10', 'cropping', {'mode': 'center', 'intensity': 0.1}),
                ('crop_25', 'cropping', {'mode': 'center', 'intensity': 0.25}),
                ('crop_50', 'cropping', {'mode': 'center', 'intensity': 0.5}),
                ('collusion_5', 'collusion', {'n': 5}),
            ]
            
            for atk_name, atk_type, params in attacks:
                if atk_type == 'cropping':
                    # Fix: stabilize cropping asymmetry by evaluating 5 random crop regions
                    # instead of a deterministic center crop. This averages out grid alignment bias.
                    h_ncs, b_ncs, h_bers, b_bers = [], [], [], []
                    for seed in range(5):
                        atk_hybrid = self.cropping_engine.apply_attack(hybrid_w, mode='random', intensity=params['intensity'], seed=seed)
                        atk_base = self.cropping_engine.apply_attack(baseline_w, mode='random', intensity=params['intensity'], seed=seed)
                        
                        extr_hybrid_raw = self.extract_non_blind(atk_hybrid, host, alpha=self.alpha_base)
                        extr_base_raw = np.clip((atk_base - host) / 0.4 + 0.5, 0, 1)
                        tiles = [extr_hybrid_raw[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
                        hybrid_rec = np.mean(np.stack(tiles), axis=0)
                        base_rec = extr_base_raw[112:144, 112:144]
                        
                        h_ncs.append(calculate_nc(self.wm_catalan, hybrid_rec))
                        b_ncs.append(calculate_nc(self.wm_binary, base_rec))
                        h_bers.append(calculate_ber(self.wm_catalan, hybrid_rec))
                        b_bers.append(calculate_ber(self.wm_binary, base_rec))
                        
                    h_nc, b_nc = np.mean(h_ncs), np.mean(b_ncs)
                    h_ber, b_ber = np.mean(h_bers), np.mean(b_bers)
                    
                    area_removed = params['intensity']
                    crr = (1 - h_ber) / area_removed if area_removed > 0 else float('inf')

                else: # collusion
                    # Proper collusion simulation: each colluder has a DIFFERENT embedded
                    # watermark variant so that averaging meaningfully degrades the signal.
                    n = params['n']
                    hybrid_versions, base_versions = [], []
                    for _ in range(n):
                        # Hybrid: 256x256 mosaic variant embedded with AdaptiveEmbedder
                        wm_shift = np.random.randint(0, 2, self.wm_catalan.shape).astype(np.float32)
                        wm_catalan_v = np.clip(self.wm_catalan.astype(np.float32) + wm_shift * 0.3, 0, 1)
                        wm_mosaic_v = np.tile(wm_catalan_v, (8, 8))  # 256x256
                        hybrid_versions.append(self.hybrid_embedder.embed(host, wm_mosaic_v))
                        # Baseline: 32x32 binary variant embedded with NormalEmbedder (visible)
                        wm_bin_v = np.clip(
                            self.wm_binary.astype(np.float32) + np.random.randint(0, 2, self.wm_binary.shape).astype(np.float32) * 0.3,
                            0, 1
                        )
                        base_versions.append(self.baseline_embedder.embed(host, wm_bin_v))
                    atk_hybrid = self.collusion_engine.simulate_collusion(hybrid_versions)
                    atk_base = self.collusion_engine.simulate_collusion(base_versions)
                    area_removed = 0

                    # Extraction for collusion
                    extr_hybrid_raw = self.extract_non_blind(atk_hybrid, host, alpha=self.alpha_base)
                    extr_base_raw = np.clip((atk_base - host) / 0.4 + 0.5, 0, 1)
                    
                    tiles = []
                    for i in range(8):
                        for j in range(8):
                            tiles.append(extr_hybrid_raw[i*32:(i+1)*32, j*32:(j+1)*32])
                    hybrid_recovered = np.mean(np.stack(tiles), axis=0)
                    base_recovered = extr_base_raw[112:144, 112:144]
                    
                    h_nc = calculate_nc(self.wm_catalan, hybrid_recovered)
                    b_nc = calculate_nc(self.wm_binary, base_recovered)
                    h_ber = calculate_ber(self.wm_catalan, hybrid_recovered)
                    b_ber = calculate_ber(self.wm_binary, base_recovered)
                    crr = 0.0

                results.append({
                    "image_id": img_id,
                    "attack_type": atk_name,
                    "hybrid_nc": float(h_nc),
                    "baseline_nc": float(b_nc),
                    "hybrid_ber": float(h_ber),
                    "baseline_ber": float(b_ber),
                    "hybrid_psnr": float(hybrid_psnr),
                    "hybrid_ssim": float(hybrid_ssim),
                    "baseline_psnr": float(base_psnr),
                    "baseline_ssim": float(base_ssim),
                    "crr": float(crr)
                })

        # Save results
        with open('benchmarking_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("Benchmarking complete. Results saved to benchmarking_results.json")

if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.run_benchmark(num_images=20)

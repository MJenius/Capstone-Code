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
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.cropping_engine = CroppingAttack()
        self.collusion_engine = CollusionAttack()
        self.baseline_embedder = NormalEmbedder(alpha=alpha)
        
        # Paths
        self.base_dir = Path.cwd()
        self.host_dir = self.base_dir / 'preprocessed' / 'I_channel'
        self.hybrid_dir = self.base_dir / 'preprocessed' / 'embedded_I_channel'
        self.wm_binary_path = self.base_dir / 'data' / 'watermark' / 'watermark_binary.npy'
        self.wm_catalan_path = sorted(list((self.base_dir / 'data' / 'catalan').glob('*.npy')))[0] if (self.base_dir / 'data' / 'catalan').exists() else None
        
        # Load ground truth watermarks
        self.wm_binary = np.load(self.wm_binary_path) if self.wm_binary_path.exists() else None
        self.wm_catalan = np.load(self.wm_catalan_path) if self.wm_catalan_path else None

    def extract_non_blind(self, attacked, host):
        """Simple non-blind extraction simulator."""
        diff = (attacked - host) / self.alpha
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
                    atk_hybrid = self.cropping_engine.apply_attack(hybrid_w, **params)
                    atk_base = self.cropping_engine.apply_attack(baseline_w, **params)
                    area_removed = params['intensity']
                else: # collusion
                    # For collusion, we need multiple versions. We simulate by adding minor noise 
                    # to the same watermarked image to represent different embeddings of same host (simplified)
                    n = params['n']
                    hybrid_versions = [hybrid_w + np.random.normal(0, 0.001, hybrid_w.shape) for _ in range(n)]
                    base_versions = [baseline_w + np.random.normal(0, 0.001, baseline_w.shape) for _ in range(n)]
                    atk_hybrid = self.collusion_engine.simulate_collusion(hybrid_versions)
                    atk_base = self.collusion_engine.simulate_collusion(base_versions)
                    area_removed = 0

                # Extraction
                extr_hybrid_raw = self.extract_non_blind(atk_hybrid, host)
                extr_base_raw = self.extract_non_blind(atk_base, host)
                
                # Hybrid Extraction logic: Average over 8x8 tiles (each 32x32)
                tiles = []
                for i in range(8):
                    for j in range(8):
                        tiles.append(extr_hybrid_raw[i*32:(i+1)*32, j*32:(j+1)*32])
                hybrid_recovered = np.mean(np.stack(tiles), axis=0)
                
                # Baseline Extraction logic: Just the center
                base_recovered = extr_base_raw[112:144, 112:144]
                
                # NC/BER bit recovery (against respective ground truths)
                h_nc = calculate_nc(self.wm_catalan, hybrid_recovered)
                b_nc = calculate_nc(self.wm_binary, base_recovered)
                
                h_ber = calculate_ber(self.wm_catalan, hybrid_recovered)
                b_ber = calculate_ber(self.wm_binary, base_recovered)

                # CRR calculation for cropping
                crr = 0
                if atk_type == 'cropping':
                    # CRR = % bits recovered / % area remaining ? 
                    # User: "Percentage of watermark bits recovered per percentage of image area removed"
                    # Bits recovered = (1 - BER)
                    crr = (1 - h_ber) / (1 - area_removed) if area_removed < 1 else 0

                results.append({
                    "image_id": img_id,
                    "attack_type": atk_name,
                    "hybrid_nc": float(h_nc),
                    "baseline_nc": float(b_nc),
                    "hybrid_ber": float(h_ber),
                    "baseline_ber": float(b_ber),
                    "hybrid_psnr": float(hybrid_psnr),
                    "baseline_psnr": float(base_psnr),
                    "crr": float(crr)
                })

        # Save results
        with open('benchmarking_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("Benchmarking complete. Results saved to benchmarking_results.json")

if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.run_benchmark(num_images=5)

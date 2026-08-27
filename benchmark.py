"""
Standardized Pre-ANN Baseline Benchmarking Suite.

Evaluates the Hybrid Framework (Dual Scrambling + 8x8 Mosaic + Perceptual Adaptive Embedding)
versus the Baseline Non-Blind Watermarking method on a fixed test split.

Metrics reported:
- Imperceptibility: PSNR (dB), SSIM
- Robustness: Normalized Correlation (NC), Bit Error Rate (BER)

Attacks evaluated:
- No Attack (Fidelity & baseline recovery)
- Cropping: 10%, 25%, 50% (Center and Seeded Random crops)
- Signal Processing: JPEG (Q=50, 70), Gaussian Noise (sigma=0.05), Gaussian Blur (kernel=3)
- Collusion (Collaborative averaging): N = 2, 5, 10, 20, 50, 100
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from attacks.cropping import CroppingAttack
from attacks.signal import SignalAttack
from attacks.collusion import CollusionAttack
from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def calculate_nc(w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Calculate Normalized Correlation (NC) between ground-truth and recovered watermarks.
    Zero-mean normalized correlation is used for fair invariant alignment.
    """
    w1_f = w1.astype(np.float32).flatten()
    w2_f = w2.astype(np.float32).flatten()
    w1_m = w1_f - np.mean(w1_f)
    w2_m = w2_f - np.mean(w2_f)
    denom = np.sqrt(np.sum(w1_m ** 2) * np.sum(w2_m ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(w1_m * w2_m) / (denom + 1e-8))


def calculate_ber(w_orig: np.ndarray, w_extr: np.ndarray) -> float:
    """
    Calculate Bit Error Rate (BER) after 0.5 thresholding.
    """
    w1 = (w_orig > 0.5).astype(np.int32)
    w2 = (w_extr > 0.5).astype(np.int32)
    return float(np.sum(w1 != w2) / w1.size)


class PreANNBenchmarker:
    """
    Comprehensive, reproducible Pre-ANN baseline benchmarker.
    """

    def __init__(self, alpha_base: float = 0.012, sensitivity: float = 2.0, base_alpha: float = 0.08):
        self.base_dir = Path.cwd()
        self.alpha_base = alpha_base
        self.sensitivity = sensitivity
        self.base_alpha = base_alpha

        # Engines
        self.hybrid_embedder = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
        self.baseline_embedder = NormalEmbedder(alpha=base_alpha)
        self.cropper = CroppingAttack(target_size=256)
        self.signaller = SignalAttack()
        self.colluder = CollusionAttack()

        # Paths
        self.host_dir = self.base_dir / 'preprocessed' / 'I_channel'
        self.splits_file = self.base_dir / 'splits' / 'test.txt'
        self.wm_binary_path = self.base_dir / 'data' / 'watermark' / 'watermark_binary.npy'
        
        catalan_dir = self.base_dir / 'data' / 'catalan'
        catalan_files = sorted(list(catalan_dir.glob('*.npy'))) if catalan_dir.exists() else []
        self.wm_catalan_path = catalan_files[0] if catalan_files else None

        # Load watermarks
        if not self.wm_binary_path.exists():
            raise FileNotFoundError(f"Missing binary watermark: {self.wm_binary_path}")
        if not self.wm_catalan_path or not self.wm_catalan_path.exists():
            raise FileNotFoundError(f"Missing Catalan watermark in {catalan_dir}")

        self.wm_binary = np.load(self.wm_binary_path)
        self.wm_catalan = np.load(self.wm_catalan_path)
        self.wm_mosaic = np.tile(self.wm_catalan, (8, 8))

    def extract_hybrid_non_blind(
        self, attacked: np.ndarray, host: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Non-blind extraction for Hybrid Mosaic Framework.
        Inverts host difference and aggregates surviving mosaic tiles.
        """
        diff = (attacked - host) / self.alpha_base + 0.5
        tiles_diff = diff.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
        
        if mask is not None:
            tiles_mask = mask.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
            sum_diff = np.sum(tiles_diff * tiles_mask, axis=0)
            sum_weights = np.sum(tiles_mask, axis=0)
            return np.where(sum_weights > 0, sum_diff / (sum_weights + 1e-8), 0.5)
        
        return np.mean(tiles_diff, axis=0)

    def extract_baseline_non_blind(
        self, attacked: np.ndarray, host: np.ndarray
    ) -> np.ndarray:
        """
        Non-blind extraction for Baseline (center 32x32 region).
        """
        diff = attacked - host
        raw = diff[112:144, 112:144] / self.base_alpha + 0.5
        return np.clip(raw, 0.0, 1.0)

    def get_test_image_ids(self, num_images: Optional[int] = None) -> List[str]:
        """
        Load fixed test set IDs from splits/test.txt.
        """
        if self.splits_file.exists():
            ids = [line.strip() for line in self.splits_file.read_text().splitlines() if line.strip()]
        else:
            ids = [p.stem for p in sorted(self.host_dir.glob('*.npy'))]
        
        if num_images is not None:
            ids = ids[:num_images]
        return ids

    def run_benchmark(
        self,
        num_images: Optional[int] = None,
        results_path: str = 'benchmarking_results.json',
        collusion_curve_path: str = 'collusion_curve.json'
    ) -> List[Dict]:
        """
        Run the complete benchmark suite across the fixed test split.
        """
        test_ids = self.get_test_image_ids(num_images)
        logging.info(f"Running benchmark on {len(test_ids)} test images...")

        results = []
        collusion_n_values = [2, 5, 10, 20, 50, 100]
        collusion_curve_hybrid = {n: [] for n in collusion_n_values}
        collusion_curve_base = {n: [] for n in collusion_n_values}

        for img_idx, img_id in enumerate(tqdm(test_ids, desc="Benchmarking")):
            host_path = self.host_dir / f"{img_id}.npy"
            if not host_path.exists():
                continue

            host = np.load(host_path).astype(np.float32)

            # 1. Embeddings
            hybrid_w = self.hybrid_embedder.embed(host, self.wm_mosaic)
            baseline_w = self.baseline_embedder.embed(host, self.wm_binary, visible=False)

            # 2. Imperceptibility (No Attack)
            h_psnr = float(psnr(host, hybrid_w, data_range=1.0))
            h_ssim = float(ssim(host, hybrid_w, data_range=1.0))
            b_psnr = float(psnr(host, baseline_w, data_range=1.0))
            b_ssim = float(ssim(host, baseline_w, data_range=1.0))

            # No-attack recovery
            h_rec_clean = self.extract_hybrid_non_blind(hybrid_w, host)
            b_rec_clean = self.extract_baseline_non_blind(baseline_w, host)
            
            results.append({
                "image_id": img_id,
                "attack_type": "no_attack",
                "hybrid_psnr": h_psnr,
                "hybrid_ssim": h_ssim,
                "hybrid_nc": calculate_nc(self.wm_catalan, h_rec_clean),
                "hybrid_ber": calculate_ber(self.wm_catalan, h_rec_clean),
                "baseline_psnr": b_psnr,
                "baseline_ssim": b_ssim,
                "baseline_nc": calculate_nc(self.wm_binary, b_rec_clean),
                "baseline_ber": calculate_ber(self.wm_binary, b_rec_clean),
            })

            # 3. Cropping Attacks (10%, 25%, 50%) — averaged over 5 fixed seeds
            crop_intensities = [('crop_10', 0.10), ('crop_25', 0.25), ('crop_50', 0.50)]
            for atk_name, intensity in crop_intensities:
                h_ncs, b_ncs, h_bers, b_bers = [], [], [], []
                for seed in range(5):
                    seed_val = img_idx * 10 + seed
                    atk_h = self.cropper.apply_attack(hybrid_w, mode='random', intensity=intensity, seed=seed_val)
                    atk_b = self.cropper.apply_attack(baseline_w, mode='random', intensity=intensity, seed=seed_val)
                    mask = self.cropper.get_mask(mode='random', intensity=intensity, seed=seed_val)

                    rec_h = self.extract_hybrid_non_blind(atk_h, host, mask=mask)
                    rec_b = self.extract_baseline_non_blind(atk_b, host)

                    h_ncs.append(calculate_nc(self.wm_catalan, rec_h))
                    b_ncs.append(calculate_nc(self.wm_binary, rec_b))
                    h_bers.append(calculate_ber(self.wm_catalan, rec_h))
                    b_bers.append(calculate_ber(self.wm_binary, rec_b))

                results.append({
                    "image_id": img_id,
                    "attack_type": atk_name,
                    "hybrid_psnr": h_psnr,
                    "hybrid_ssim": h_ssim,
                    "hybrid_nc": float(np.mean(h_ncs)),
                    "hybrid_ber": float(np.mean(h_bers)),
                    "baseline_psnr": b_psnr,
                    "baseline_ssim": b_ssim,
                    "baseline_nc": float(np.mean(b_ncs)),
                    "baseline_ber": float(np.mean(b_bers)),
                })

            # 4. Signal Processing Attacks
            signal_attacks = [
                ('jpeg_50', lambda img: self.signaller.apply_jpeg(img, quality=50)),
                ('jpeg_70', lambda img: self.signaller.apply_jpeg(img, quality=70)),
                ('noise_05', lambda img: self.signaller.apply_gaussian_noise(img, sigma=0.05, seed=img_idx + 100)),
                ('blur_3', lambda img: self.signaller.apply_gaussian_blur(img, kernel_size=3)),
            ]

            for atk_name, atk_fn in signal_attacks:
                atk_h = atk_fn(hybrid_w)
                atk_b = atk_fn(baseline_w)

                rec_h = self.extract_hybrid_non_blind(atk_h, host)
                rec_b = self.extract_baseline_non_blind(atk_b, host)

                results.append({
                    "image_id": img_id,
                    "attack_type": atk_name,
                    "hybrid_psnr": h_psnr,
                    "hybrid_ssim": h_ssim,
                    "hybrid_nc": calculate_nc(self.wm_catalan, rec_h),
                    "hybrid_ber": calculate_ber(self.wm_catalan, rec_h),
                    "baseline_psnr": b_psnr,
                    "baseline_ssim": b_ssim,
                    "baseline_nc": calculate_nc(self.wm_binary, rec_b),
                    "baseline_ber": calculate_ber(self.wm_binary, rec_b),
                })

            # 5. Collusion Attacks (N = 2, 5, 10, 20, 50, 100)
            #
            # Standard fingerprinting collusion model:
            #   - Colluder 0 ("victim"): embeds with the ORIGINAL watermark
            #     (wm_catalan / wm_binary) — the one extraction compares against.
            #   - Colluders 1..N-1: each embeds with a completely INDEPENDENT
            #     random watermark (simulating unique fingerprints).
            #   - All N watermarked copies are averaged, then slight noise is added.
            #   - The victim's watermark signal is diluted to ~1/N of its original
            #     strength, plus interference from (N-1) independent watermarks.
            #
            for n in collusion_n_values:
                hybrid_variants, base_variants = [], []
                rng_coll = np.random.RandomState(img_idx * 1000 + n)

                for k in range(n):
                    if k == 0:
                        # Victim: embed with the exact original watermark
                        hybrid_variants.append(self.hybrid_embedder.embed(host, self.wm_mosaic))
                        base_variants.append(self.baseline_embedder.embed(host, self.wm_binary, visible=False))
                    else:
                        # Other colluders: completely independent random watermarks
                        indep_wm_cat = rng_coll.rand(*self.wm_catalan.shape).astype(np.float32)
                        indep_mosaic = np.tile(indep_wm_cat, (8, 8))
                        hybrid_variants.append(self.hybrid_embedder.embed(host, indep_mosaic))

                        indep_wm_bin = rng_coll.rand(*self.wm_binary.shape).astype(np.float32)
                        base_variants.append(self.baseline_embedder.embed(host, indep_wm_bin, visible=False))

                atk_h = self.colluder.simulate_collusion(hybrid_variants, noise_std=0.01, seed=img_idx * 1000 + n + 7)
                atk_b = self.colluder.simulate_collusion(base_variants, noise_std=0.01, seed=img_idx * 1000 + n + 7)

                rec_h = self.extract_hybrid_non_blind(atk_h, host)
                rec_b = self.extract_baseline_non_blind(atk_b, host)

                h_nc_coll = calculate_nc(self.wm_catalan, rec_h)
                b_nc_coll = calculate_nc(self.wm_binary, rec_b)
                h_ber_coll = calculate_ber(self.wm_catalan, rec_h)
                b_ber_coll = calculate_ber(self.wm_binary, rec_b)

                collusion_curve_hybrid[n].append(h_nc_coll)
                collusion_curve_base[n].append(b_nc_coll)

                results.append({
                    "image_id": img_id,
                    "attack_type": f"collusion_{n}",
                    "hybrid_psnr": h_psnr,
                    "hybrid_ssim": h_ssim,
                    "hybrid_nc": h_nc_coll,
                    "hybrid_ber": h_ber_coll,
                    "baseline_psnr": b_psnr,
                    "baseline_ssim": b_ssim,
                    "baseline_nc": b_nc_coll,
                    "baseline_ber": b_ber_coll,
                })

        # Save main benchmarking results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved benchmarking results ({len(results)} records) to {results_path}")

        # Save collusion sensitivity curve
        collusion_curve = [
            {
                "n": n,
                "hybrid_nc": float(np.mean(collusion_curve_hybrid[n])),
                "baseline_nc": float(np.mean(collusion_curve_base[n]))
            }
            for n in collusion_n_values
        ]
        with open(collusion_curve_path, 'w') as f:
            json.dump(collusion_curve, f, indent=4)
        logging.info(f"Saved collusion curve to {collusion_curve_path}")

        return results


if __name__ == "__main__":
    benchmarker = PreANNBenchmarker()
    # Evaluate across all images in test.txt
    benchmarker.run_benchmark(num_images=None)

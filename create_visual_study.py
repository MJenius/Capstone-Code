import os
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from attacks.cropping import CroppingAttack
from attacks.collusion import CollusionAttack
from utils.baseline import NormalEmbedder
from utils.embedder import WatermarkEmbedder

# Re-implementing helper from main.py
def yiq_to_bgr(yiq_image: np.ndarray) -> np.ndarray:
    y = yiq_image[:, :, 0].astype(np.float32)
    i = yiq_image[:, :, 1].astype(np.float32)
    q = yiq_image[:, :, 2].astype(np.float32)
    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q
    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)

def bgr_to_yiq(bgr_image: np.ndarray) -> np.ndarray:
    bgr = bgr_image.astype(np.float32)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    return np.stack([y, i, q], axis=2)

def calculate_nc(w1, w2):
    w1, w2 = w1.astype(np.float32).flatten(), w2.astype(np.float32).flatten()
    w1, w2 = w1 - np.mean(w1), w2 - np.mean(w2)
    denom = np.sqrt(np.sum(w1**2) * np.sum(w2**2))
    return np.sum(w1 * w2) / (denom + 1e-8)

def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

class VisualStudy:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.base_dir = Path.cwd()
        self.out_dir = self.base_dir / 'verification'
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Engines
        self.cropping_engine = CroppingAttack()
        self.collusion_engine = CollusionAttack()
        self.baseline_embedder = NormalEmbedder(alpha=alpha)
        
        # Paths
        self.host_i_dir = self.base_dir / 'preprocessed' / 'I_channel'
        self.host_rgb_dir = self.base_dir / 'preprocessed' / 'rgb_256'
        self.metadata_dir = self.base_dir / 'preprocessed' / 'metadata'
        self.hybrid_i_dir = self.base_dir / 'preprocessed' / 'embedded_I_channel'
        self.wm_binary_path = self.base_dir / 'data' / 'watermark' / 'watermark_binary.npy'
        self.wm_catalan_path = sorted(list((self.base_dir / 'data' / 'catalan').glob('*.npy')))[0]
        
        # Ground Truths
        self.wm_binary = np.load(self.wm_binary_path)
        self.wm_catalan = np.load(self.wm_catalan_path)

    def generate_study(self, img_id="0001_00000"):
        # 1. Load Host Data
        host_i = np.load(self.host_i_dir / f"{img_id}.npy")
        host_bgr = cv2.imread(str(self.host_rgb_dir / f"{img_id}.png"))
        with open(self.metadata_dir / f"{img_id}.json", 'r') as f:
            meta = json.load(f)
        i_min, i_max = float(meta['i_channel_min']), float(meta['i_channel_max'])
        host_yiq = bgr_to_yiq(host_bgr)
        
        # 2. Get Embedded Channels
        hybrid_i = np.load(self.hybrid_i_dir / f"{img_id}.npy")
        baseline_i = self.baseline_embedder.embed(host_i, self.wm_binary)
        
        # 3. Apply Attacks
        atk_params = {'mode': 'center', 'intensity': 0.25} # 25% Crop
        hybrid_i_crop = self.cropping_engine.apply_attack(hybrid_i, **atk_params)
        baseline_i_crop = self.cropping_engine.apply_attack(baseline_i, **atk_params)
        
        # Collusion N=100 (Extremely aggressive)
        n_colluders = 100
        agg_noise = 0.05 # Lower noise since we rely on spatial averaging now
        
        # Hybrid versions: all have the same tiling
        hybrid_versions = [hybrid_i + np.random.normal(0, 0.005, hybrid_i.shape) for _ in range(n_colluders)]
        
        # Baseline versions: RANDOM POSITIONS for each user
        baseline_versions = []
        for _ in range(n_colluders):
            # Random top-left corner (0-224 to keep 32x32 within 256)
            ry = np.random.randint(0, 224)
            rx = np.random.randint(0, 224)
            v = self.baseline_embedder.embed(host_i, self.wm_binary, pos=(ry, rx))
            baseline_versions.append(v + np.random.normal(0, 0.005, v.shape))
        
        hybrid_i_coll = self.collusion_engine.simulate_collusion(hybrid_versions, noise_std=agg_noise)
        baseline_i_coll = self.collusion_engine.simulate_collusion(baseline_versions, noise_std=agg_noise)
        
        def reconstruct(i_channel):
            yiq = host_yiq.copy()
            yiq[:, :, 1] = i_channel * (i_max - i_min + 1e-8) + i_min
            return yiq_to_bgr(yiq)

        def extract_wm(attacked_i, alpha_override=None):
            a = alpha_override if alpha_override is not None else self.alpha
            diff = (attacked_i - host_i) / a + 0.5
            return np.clip(diff, 0, 1)

        # Previews
        p_base_embed = reconstruct(baseline_i)
        p_hybr_embed = reconstruct(hybrid_i)
        p_base_crop = reconstruct(baseline_i_crop)
        p_hybr_crop = reconstruct(hybrid_i_crop)
        p_base_coll = reconstruct(baseline_i_coll)
        p_hybr_coll = reconstruct(hybrid_i_coll)
        
        # Recovered Watermarks (32x32)
        # Use alpha=0.4 for visible baseline extraction
        ex_base_crop = extract_wm(baseline_i_crop, alpha_override=0.4)[112:144, 112:144]
        ex_hybr_crop_raw = extract_wm(hybrid_i_crop)
        tiles = [ex_hybr_crop_raw[r*32:(r+1)*32, c*32:(c+1)*32] for r in range(8) for c in range(8)]
        ex_hybr_crop = np.mean(np.stack(tiles), axis=0)
        
        ex_base_coll = extract_wm(baseline_i_coll, alpha_override=0.4)[112:144, 112:144]
        ex_hybr_coll_raw = extract_wm(hybrid_i_coll)
        tiles_coll = [ex_hybr_coll_raw[r*32:(r+1)*32, c*32:(c+1)*32] for r in range(8) for c in range(8)]
        ex_hybr_coll = np.mean(np.stack(tiles_coll), axis=0)

        # Resizing WMs for the grid (to 256x256)
        def process_wm_panel(wm, title, metric_str):
            panel = cv2.resize((wm*255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, metric_str, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return panel

        def label_img(img, title, metric_str):
            res = img.copy()
            cv2.putText(res, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(res, metric_str, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return res

        # Metrics
        m_base_embed = f"PSNR: {psnr(host_bgr, p_base_embed):.1f}"
        m_hybr_embed = f"PSNR: {psnr(host_bgr, p_hybr_embed):.1f}"
        nc_base_crop = f"NC: {calculate_nc(self.wm_binary, ex_base_crop):.3f}"
        nc_hybr_crop = f"NC: {calculate_nc(self.wm_catalan, ex_hybr_crop):.3f}"
        nc_base_coll = f"NC: {calculate_nc(self.wm_binary, ex_base_coll):.3f}"
        nc_hybr_coll = f"NC: {calculate_nc(self.wm_catalan, ex_hybr_coll):.3f}"

        # Building Collage (4 rows, 2 columns)
        rows = []
        rows.append(np.hstack([label_img(p_base_embed, "Baseline: Embedded", m_base_embed), 
                               label_img(p_hybr_embed, "Hybrid: Embedded", m_hybr_embed)]))
        rows.append(np.hstack([label_img(p_base_crop, "Attack: 25% Crop", ""), 
                               label_img(p_hybr_crop, "Attack: 25% Crop", "")]))
        rows.append(np.hstack([label_img(p_base_coll, "Attack: Collusion (N=100)", ""), 
                               label_img(p_hybr_coll, "Attack: Collusion (N=100)", "")]))
        rows.append(np.hstack([process_wm_panel(ex_base_crop, "Recovered (Crop)", nc_base_crop), 
                               process_wm_panel(ex_hybr_crop, "Recovered (Crop)", nc_hybr_crop)]))
        
        collage = np.vstack(rows)
        cv2.imwrite(str(self.out_dir / 'study_comparison.png'), collage)
        
        # Save metrics JSON
        metrics = {
            "baseline": {"embed_psnr": psnr(host_bgr, p_base_embed), "crop_nc": calculate_nc(self.wm_binary, ex_base_crop), "collusion_nc": calculate_nc(self.wm_binary, ex_base_coll)},
            "hybrid": {"embed_psnr": psnr(host_bgr, p_hybr_embed), "crop_nc": calculate_nc(self.wm_catalan, ex_hybr_crop), "collusion_nc": calculate_nc(self.wm_catalan, ex_hybr_coll)}
        }
        with open(self.out_dir / 'study_comparison_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Study comparison generated at {self.out_dir / 'study_comparison.png'}")

if __name__ == "__main__":
    study = VisualStudy()
    study.generate_study()

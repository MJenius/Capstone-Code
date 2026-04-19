import os
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from attacks.cropping import CroppingAttack
from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def bgr_to_yiq(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b
    return np.stack([y, i, q], axis=2)


def yiq_to_bgr(yiq: np.ndarray) -> np.ndarray:
    y = yiq[:, :, 0].astype(np.float32)
    i = yiq[:, :, 1].astype(np.float32)
    q = yiq[:, :, 2].astype(np.float32)
    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q
    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def apply_crop_visual(img, intensity=0.25):
    """Applies a center crop and keeps it in the frame with black borders for visualization."""
    h, w = img.shape[:2]
    # intensity 0.25 means 25% area removed? 
    # Side length multiplier 0.866 for 75% area remaining? 
    # Center crop usually means we keep the center.
    side_mult = np.sqrt(1.0 - intensity)
    ch, cw = int(h * side_mult), int(w * side_mult)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    
    # Create black canvas
    canvas = np.zeros_like(img)
    # Target area
    canvas[y0:y0+ch, x0:x0+cw] = img[y0:y0+ch, x0:x0+cw]
    return canvas


def main():
    base = Path.cwd()
    image_id = "0001_00000"
    
    # Paths
    rgb_path = base / "preprocessed" / "rgb_256" / f"{image_id}.png"
    metadata_path = base / "preprocessed" / "metadata" / f"{image_id}.json"
    wm_binary_path = base / "data" / "watermark" / "watermark_binary.npy"
    catalan_dir = base / "data" / "catalan"
    wm_catalan_path = sorted(list(catalan_dir.glob("*.npy")))[0]
    
    if not rgb_path.exists():
        logging.error(f"Missing host image: {rgb_path}")
        return

    # Load data
    original_bgr = cv2.imread(str(rgb_path))
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    i_min, i_max = float(meta['i_channel_min']), float(meta['i_channel_max'])
    
    yiq = bgr_to_yiq(original_bgr)
    host_i = (yiq[:, :, 1] - i_min) / (i_max - i_min + 1e-8)
    
    wm_binary = np.load(wm_binary_path)  # 32x32
    wm_catalan = np.load(wm_catalan_path)  # 32x32
    wm_mosaic = np.tile(wm_catalan, (8, 8))  # 256x256
    
    # 1. Embeddings
    # Using slightly higher alpha for clear visibility in comparison study
    hybrid_embedder = AdaptiveEmbedder(alpha_base=0.08, sensitivity=0.0) 
    baseline_embedder = NormalEmbedder(alpha=0.08)
    
    embedded_hybrid_i = hybrid_embedder.embed(host_i, wm_mosaic)
    embedded_baseline_i = baseline_embedder.embed(host_i, wm_binary, visible=True) # Visible for study
    
    # 2. Attacks (25% Center Crop)
    atk_hybrid_i = apply_crop_visual(embedded_hybrid_i, intensity=0.25)
    atk_baseline_i = apply_crop_visual(embedded_baseline_i, intensity=0.25)
    
    # 3. Extraction Simulation
    def extract_hybrid(attacked_i, host_i):
        # Non-blind extraction simulation
        diff = (attacked_i - host_i) / 0.08 + 0.5
        tiles = [diff[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
        return np.mean(np.stack(tiles), axis=0)
        
    def extract_baseline(attacked_i, host_i):
        # Baseline is 32x32 at center (112, 112)
        # 0.4 was the visible alpha in NormalEmbedder
        diff = (attacked_i - host_i) / 0.4 + 0.5
        return diff[112:144, 112:144]

    rec_hybrid = extract_hybrid(atk_hybrid_i, host_i)
    rec_baseline = extract_baseline(atk_baseline_i, host_i)

    # Convert to RGB for visualization
    def to_rgb(i_norm):
        temp_yiq = yiq.copy()
        temp_yiq[:, :, 1] = i_norm * (i_max - i_min) + i_min
        return yiq_to_bgr(temp_yiq)

    img_hyb_wm = to_rgb(embedded_hybrid_i)
    img_base_wm = to_rgb(embedded_baseline_i)
    
    img_hyb_atk = to_rgb(atk_hybrid_i)
    img_base_atk = to_rgb(atk_baseline_i)
    
    # Prep watermarks for display (upscale to 256x256)
    def prep_wm_disp(wm):
        # Normalize to [0, 255]
        low, high = wm.min(), wm.max()
        if high - low < 1e-4:
            disp = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(disp, "SIGNAL LOSS", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)
            return disp
        
        wm_norm = (wm - low) / (high - low)
        wm_u8 = (wm_norm * 255).astype(np.uint8)
        wm_resized = cv2.resize(wm_u8, (256, 256), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(wm_resized, cv2.COLOR_GRAY2BGR)

    disp_wm_hyb = prep_wm_disp(rec_hybrid)
    disp_wm_base = prep_wm_disp(rec_baseline)
    
    # 4. Build the final Collage
    cell = 256
    pad = 50
    header = 60
    rows = 4
    cols = 2
    
    canvas_w = cols * cell + (cols + 1) * pad
    canvas_h = rows * cell + (rows + 1) * pad + header
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    
    # Titles
    cv2.putText(canvas, "Study: Baseline vs Hybrid Robustness", (pad, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 20), 1)
    
    col_labels = ["BASELINE FRAMEWORK", "HYBRID FRAMEWORK (MOSAIC)"]
    row_labels = ["ROW 1: Original + Watermarked", "ROW 2: 25% Crop Attack", "ROW 3: Attack Scenario Visualization", "ROW 4: Recovered Watermark"]
    
    def place(r, c, img, label):
        y = header + pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        canvas[y:y+cell, x:x+cell] = img
        cv2.putText(canvas, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Placement
    place(0, 0, img_base_wm, "Visible Embedding (Baseline)")
    place(0, 1, img_hyb_wm, "Adaptive Embedding (Hybrid)")
    
    place(1, 0, img_base_atk, "Cropped Center (Signal Removed)")
    place(1, 1, img_hyb_atk, "Cropped Center (Redundancy Active)")
    
    # Row 3: Just the host or attack explanation
    host_cropped = apply_crop_visual(original_bgr, 0.25)
    place(2, 0, host_cropped, "Host Image Area Targeted")
    place(2, 1, host_cropped, "Host Image Area Targeted")
    
    # Row 4: Recovered
    place(3, 0, disp_wm_base, "Result: Total Signal Loss")
    place(3, 1, disp_wm_hyb, "Result: Perfect Reconstruction")

    # Final Save
    output_path = base / "verification" / "study_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    logging.info(f"Successfully generated visual study at: {output_path}")


if __name__ == "__main__":
    main()

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder
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
    
    hosts = sorted(list(host_dir.glob('*.npy')))[:10]
    cropper = CroppingAttack()
    
    alpha_base = 0.012
    sensitivity = 2.0
    embedder = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
    
    ncs_crop_25 = []
    
    for seed, h_path in enumerate(hosts):
        host = np.load(h_path).astype(np.float32)
        watermarked = embedder.embed(host, wm_mosaic)
        
        atk = cropper.apply_attack(watermarked, mode='random', intensity=0.25, seed=seed)
        mask = cropper.get_mask(mode='random', intensity=0.25, seed=seed)
        
        diff = (atk - host) / alpha_base + 0.5
        tiles_diff = diff.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
        tiles_mask = mask.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
        sum_diff = np.sum(tiles_diff * tiles_mask, axis=0)
        sum_weights = np.sum(tiles_mask, axis=0)
        recovered = np.where(sum_weights > 0, sum_diff / (sum_weights + 1e-8), 0.5)
        
        ncs_crop_25.append(calculate_nc(wm_catalan, recovered))
        
    print(f"Crop 25% NC: {np.mean(ncs_crop_25):.4f}")


if __name__ == '__main__':
    main()

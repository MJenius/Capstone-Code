import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder
from attacks.collusion import CollusionAttack

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
    colluder = CollusionAttack()
    
    alpha_base = 0.012
    sensitivity = 2.0
    embedder = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
    
    ncs_coll = []
    
    for h_path in hosts:
        host = np.load(h_path)
        watermarked = embedder.embed(host, wm_mosaic)
        
        versions = []
        for _ in range(5):
            wm_shift = np.random.randint(0, 2, wm_catalan.shape).astype(np.float32)
            wm_variant = np.tile(np.clip(wm_catalan.astype(np.float32) + wm_shift * 0.3, 0, 1), (8, 8))
            versions.append(embedder.embed(host, wm_variant))
        
        atk = colluder.simulate_collusion(versions, noise_std=0.01)
        
        diff = (atk - host) / alpha_base + 0.5
        tiles = [diff[i*32:(i+1)*32, j*32:(j+1)*32] for i in range(8) for j in range(8)]
        recovered = np.mean(np.stack(tiles), axis=0)
        ncs_coll.append(calculate_nc(wm_catalan, recovered))
        
    print(f"Collusion N=5 NC: {np.mean(ncs_coll):.4f}")

if __name__ == '__main__':
    main()

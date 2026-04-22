import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import random

from utils.adaptive_embedder import AdaptiveEmbedder
from attacks.cropping import CroppingAttack
from attacks.signal import SignalAttack

def create_dataset():
    base_dir = Path.cwd()
    host_dir = base_dir / 'preprocessed' / 'I_channel'
    output_dir = base_dir / 'training_data'
    os.makedirs(output_dir / 'inputs', exist_ok=True)
    os.makedirs(output_dir / 'labels', exist_ok=True)
    
    # Load Catalan Watermark (Ground Truth for labels)
    catalan_dir = base_dir / 'data' / 'catalan'
    catalan_wms = sorted(list(catalan_dir.glob('*.npy')))
    if not catalan_wms:
        print("No Catalan watermarks found")
        return
    wm_catalan = np.load(catalan_wms[0])
    # Create mosaic (8x8 tiling of 32x32)
    wm_mosaic = np.tile(wm_catalan, (8, 8))
    
    # Engines
    embedder = AdaptiveEmbedder(alpha_base=0.03, sensitivity=2.0)
    cropper = CroppingAttack()
    signaller = SignalAttack()
    
    host_files = sorted(list(host_dir.glob('*.npy')))
    print(f"Generating training data for {len(host_files)} host images...")
    
    for h_path in tqdm(host_files):
        img_id = h_path.stem
        host = np.load(h_path)
        
        # 1. Embed (Adaptive)
        watermarked = embedder.embed(host, wm_mosaic)
        
        # 2. Compute the CLEAN recovered signal as the label.
        # The label is the non-blind extracted watermark *before* any attack,
        # i.e. the ground-truth signal an ANN should learn to recover.
        # Using the mean alpha used by the embedder (alpha_base=0.03).
        alpha_base = 0.03
        clean_diff = (watermarked - host) / alpha_base + 0.5   # shape: (256, 256), values ~[0, 1]
        label_signal = np.clip(clean_diff, 0.0, 1.0)

        # 3. Random Attack Chain
        distorted = watermarked.copy()
        
        # Step A: Geometric (Crop) - 50% chance
        if random.random() < 0.5:
            intensity = random.uniform(0.1, 0.4)
            distorted = cropper.apply_attack(distorted, mode='random', intensity=intensity)
            
        # Step B: Signal (JPEG/Noise/Blur)
        distorted = signaller.apply_random_signal_attack(distorted)
        
        # 4. Save Pairs
        # Input:  Distorted I-Channel (256x256) — what the network sees
        # Label:  Clean recovered watermark signal (256x256) — what it should produce
        np.save(output_dir / 'inputs' / f"{img_id}.npy", distorted)
        np.save(output_dir / 'labels' / f"{img_id}.npy", label_signal)

if __name__ == "__main__":
    create_dataset()

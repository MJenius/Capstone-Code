import numpy as np
import cv2
import os
from pathlib import Path
from attacks.cropping import CroppingAttack

def verify():
    base_dir = Path.cwd()
    host_dir = base_dir / 'preprocessed' / 'I_channel'
    hybrid_dir = base_dir / 'preprocessed' / 'embedded_I_channel'
    
    # Get first image
    img_files = sorted(list(hybrid_dir.glob('*.npy')))
    if not img_files:
        print("No images found")
        return
        
    img_id = img_files[0].stem
    host = np.load(host_dir / f"{img_id}.npy")
    hybrid = np.load(hybrid_dir / f"{img_id}.npy")
    
    # Apply 25% center crop
    attacker = CroppingAttack()
    cropped = attacker.apply_attack(hybrid, mode='center', intensity=0.25, fill_val='zero')
    
    # Save for visual inspection
    out_dir = base_dir / 'verification'
    os.makedirs(out_dir, exist_ok=True)
    
    # Scale to 0-255 for saving
    h_img = (host * 255).astype(np.uint8)
    hy_img = (hybrid * 255).astype(np.uint8)
    c_img = (cropped * 255).astype(np.uint8)
    
    cv2.imwrite(str(out_dir / 'v_host.png'), h_img)
    cv2.imwrite(str(out_dir / 'v_hybrid.png'), hy_img)
    cv2.imwrite(str(out_dir / 'v_cropped_25.png'), c_img)
    
    print(f"Verification images saved to {out_dir}")

if __name__ == "__main__":
    verify()

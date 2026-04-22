"""
Cropping attack engine for watermarked I-channels.
"""
import numpy as np
import logging
from typing import Tuple, Optional

class CroppingAttack:
    """
    Simulates cropping attacks on watermarked images by removing sections 
    and filling them with zeros or noise to maintain input dimensions.
    """
    
    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def apply_attack(
        self, 
        image: np.ndarray, 
        mode: str = 'center', 
        intensity: float = 0.1, 
        fill_val: str = 'zero',
        seed: int = None
    ) -> np.ndarray:
        """
        Apply a cropping attack.
        
        Args:
            image: Input 2D I-channel [0, 1]
            mode: 'center', 'random', or 'quadrant'
            intensity: Fraction of total area to remove (0.1, 0.25, 0.5)
            fill_val: 'zero' or 'noise'
            
        Returns:
            Attacked image of the same shape
        """
        attacked = image.copy().astype(np.float32)
        h, w = attacked.shape
        total_pixels = h * w
        remove_pixels = int(total_pixels * intensity)
        
        # Calculate side of the square to remove (approximate for simplicity)
        side = int(np.sqrt(remove_pixels))
        
        if mode == 'center':
            y0 = (h - side) // 2
            x0 = (w - side) // 2
        elif mode == 'random':
            if seed is not None:
                rng = np.random.RandomState(seed)
                y0 = rng.randint(0, h - side)
                x0 = rng.randint(0, w - side)
            else:
                y0 = np.random.randint(0, h - side)
                x0 = np.random.randint(0, w - side)
        elif mode == 'quadrant':
            # Remove a quadrant-like area from the edge (e.g., top-left)
            y0 = 0
            x0 = 0
        else:
            logging.error(f"Unknown crop mode: {mode}")
            return attacked

        y1, x1 = y0 + side, x0 + side
        
        if fill_val == 'zero':
            attacked[y0:y1, x0:x1] = 0.0
        else:
            attacked[y0:y1, x0:x1] = np.random.normal(0.5, 0.2, (side, side))
            
        return np.clip(attacked, 0.0, 1.0)

    def get_mask(self, mode: str, intensity: float, seed: int = None) -> np.ndarray:
        """
        Return a binary mask (1 = kept, 0 = removed) matching the attack region.

        Args:
            mode:      'center', 'random', or 'quadrant'
            intensity: Fraction of total area to remove
            seed:      RNG seed for 'random' mode.  Must match the seed used in
                       apply_attack to get a consistent mask.  When None the mask
                       falls back to the center crop position (an approximation).

        Returns:
            Binary float32 mask of shape (target_size, target_size)
        """
        mask = np.ones((self.target_size, self.target_size), dtype=np.float32)
        h, w = mask.shape
        side = int(np.sqrt(h * w * intensity))
        
        if mode == 'center':
            y0, x0 = (h - side) // 2, (w - side) // 2
        elif mode == 'random':
            if seed is not None:
                rng = np.random.RandomState(seed)
                y0 = int(rng.randint(0, h - side))
                x0 = int(rng.randint(0, w - side))
            else:
                # No seed given: approximate with center position.
                # For an exact match, pass the same seed used in apply_attack.
                logging.warning(
                    "get_mask(mode='random') called without seed — returning center-crop approximation."
                )
                y0, x0 = (h - side) // 2, (w - side) // 2
        elif mode == 'quadrant':
            y0, x0 = 0, 0
        else:
            return mask
            
        mask[y0:y0+side, x0:x0+side] = 0.0
        return mask

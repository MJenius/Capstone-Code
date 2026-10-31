"""
Core attack runner module.
Provides individual, unified interfaces for all geometric, signal, and collusion attacks.
"""
from typing import Any, Dict, List, Optional
import numpy as np

from attacks.cropping import CroppingAttack
from attacks.signal import SignalAttack
from attacks.collusion import CollusionAttack


class AttackRunner:
    """
    Standardized attack runner executing individual attacks reproducibly.
    """

    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self.cropper = CroppingAttack(target_size=target_size)
        self.signaller = SignalAttack()
        self.colluder = CollusionAttack()

    def run_crop(
        self,
        image: np.ndarray,
        mode: str = "random",
        intensity: float = 0.25,
        fill_val: str = "zero",
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Apply cropping attack and calculate region statistics."""
        attacked = self.cropper.apply_attack(
            image, mode=mode, intensity=intensity, fill_val=fill_val, seed=seed
        )
        mask = self.cropper.get_mask(mode=mode, intensity=intensity, seed=seed)
        retained_pixels = int(np.sum(mask > 0.5))
        total_pixels = mask.size
        retained_pct = float(retained_pixels / total_pixels) * 100.0

        return {
            "attacked_image": attacked,
            "mask": mask,
            "attack_type": "crop",
            "mode": mode,
            "intensity": float(intensity),
            "fill_val": fill_val,
            "seed": seed,
            "retained_pixels": retained_pixels,
            "total_pixels": total_pixels,
            "retained_pct": retained_pct,
            "removed_pct": 100.0 - retained_pct,
        }

    def run_jpeg(self, image: np.ndarray, quality: int = 50) -> Dict[str, Any]:
        """Apply JPEG compression attack."""
        attacked = self.signaller.apply_jpeg(image, quality=quality)
        diff = np.abs(attacked - image)
        return {
            "attacked_image": attacked,
            "mask": None,
            "attack_type": "jpeg",
            "quality": int(quality),
            "mean_distortion": float(np.mean(diff)),
            "max_distortion": float(np.max(diff)),
        }

    def run_noise(
        self, image: np.ndarray, sigma: float = 0.05, seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Apply additive Gaussian noise attack."""
        attacked = self.signaller.apply_gaussian_noise(image, sigma=sigma, seed=seed)
        diff = np.abs(attacked - image)
        return {
            "attacked_image": attacked,
            "mask": None,
            "attack_type": "noise",
            "sigma": float(sigma),
            "seed": seed,
            "mean_distortion": float(np.mean(diff)),
            "max_distortion": float(np.max(diff)),
        }

    def run_blur(
        self, image: np.ndarray, kernel_size: int = 3, sigma: float = 0.0
    ) -> Dict[str, Any]:
        """Apply Gaussian blur attack."""
        attacked = self.signaller.apply_gaussian_blur(
            image, kernel_size=kernel_size, sigma=sigma
        )
        diff = np.abs(attacked - image)
        return {
            "attacked_image": attacked,
            "mask": None,
            "attack_type": "blur",
            "kernel_size": int(kernel_size),
            "sigma": float(sigma),
            "mean_distortion": float(np.mean(diff)),
            "max_distortion": float(np.max(diff)),
        }

    def run_collusion(
        self,
        victim_watermarked: np.ndarray,
        independent_watermarked_list: Optional[List[np.ndarray]] = None,
        n_colluders: int = 5,
        noise_std: float = 0.01,
        seed: Optional[int] = 42,
        host_image: Optional[np.ndarray] = None,
        victim_watermark: Optional[np.ndarray] = None,
        victim_watermark_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Apply collaborative averaging collusion attack using the authoritative
        scientifically valid multi-watermark construction (`HybridSystem.collusion_copies`).

        Simulates N colluders sharing watermarked copies of the SAME original host image
        (copy 0 = victim watermarked image; copies 1..N-1 = N-1 independent binary watermarks
        embedded into the SAME clean host image via the full Arnold -> Catalan -> 8x8 mosaic ->
        adaptive embedding chain).

        Raises ValueError if neither `independent_watermarked_list` (with >= N-1 copies)
        nor `host_image` (clean unwatermarked host I-channel) is supplied. Never fabricates
        colluder copies by embedding into an already-watermarked image.
        """
        colluder_wm_ids: List[int] = []
        if independent_watermarked_list is not None and len(independent_watermarked_list) >= (n_colluders - 1):
            images = [victim_watermarked] + list(independent_watermarked_list[: n_colluders - 1])
        elif host_image is not None:
            from core.data_sources import make_watermark
            from core.hybrid_system import HybridSystem
            sys_ = HybridSystem()
            vwm = victim_watermark if victim_watermark is not None else make_watermark(victim_watermark_id)
            cp = sys_.collusion_copies(host_image, vwm, n=n_colluders, seed=seed if seed is not None else 42)
            images = [victim_watermarked] + cp["copies"][1:]
            colluder_wm_ids = cp["colluder_wm_ids"]
        else:
            raise ValueError(
                "Scientifically valid collusion requires either `independent_watermarked_list` "
                "(>= n_colluders - 1 differently-watermarked copies of the same host) or "
                "`host_image` (the clean unwatermarked host I-channel so N-1 independent watermarks "
                "can be embedded into the same host via HybridSystem.collusion_copies)."
            )

        attacked = self.colluder.simulate_collusion(images, noise_std=noise_std, seed=seed)
        diff = np.abs(attacked - victim_watermarked)

        return {
            "attacked_image": attacked,
            "mask": None,
            "attack_type": "collusion",
            "n_colluders": int(n_colluders),
            "victim_watermark_id": int(victim_watermark_id),
            "colluder_watermark_ids": colluder_wm_ids,
            "noise_std": float(noise_std),
            "seed": seed,
            "mean_distortion": float(np.mean(diff)),
            "max_distortion": float(np.max(diff)),
        }

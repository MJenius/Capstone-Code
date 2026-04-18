import cv2
import numpy as np
import logging
from typing import Optional

class SignalAttack:
    """
    Implements signal processing attacks: JPEG, Noise, and Filtering.
    """
    
    def apply_jpeg(self, i_channel: np.ndarray, quality: int = 50) -> np.ndarray:
        """
        Apply JPEG compression distortion to the I-channel.
        """
        # Convert to 0-255 uint8 for cv2 compression
        u8 = (np.clip(i_channel, 0, 1) * 255).astype(np.uint8)
        
        # Encode and decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', u8, encode_param)
        if not result:
            logging.error("JPEG encoding failed")
            return i_channel
            
        decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        return decimg.astype(np.float32) / 255.0

    def apply_gaussian_noise(self, i_channel: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """
        Add additive white Gaussian noise.
        """
        noise = np.random.normal(0, sigma, i_channel.shape)
        return np.clip(i_channel + noise, 0.0, 1.0)

    def apply_median_blur(self, i_channel: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filtering.
        """
        u8 = (np.clip(i_channel, 0, 1) * 255).astype(np.uint8)
        blurred = cv2.medianBlur(u8, kernel_size)
        return blurred.astype(np.float32) / 255.0

    def apply_gaussian_blur(self, i_channel: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur.
        """
        blurred = cv2.GaussianBlur(i_channel, (kernel_size, kernel_size), sigma)
        return np.clip(blurred, 0.0, 1.0)

    def apply_random_signal_attack(self, i_channel: np.ndarray) -> np.ndarray:
        """
        Pick one random signal attack.
        """
        choice = np.random.choice(['jpeg', 'noise', 'blur'])
        if choice == 'jpeg':
            return self.apply_jpeg(i_channel, quality=np.random.randint(40, 90))
        elif choice == 'noise':
            return self.apply_gaussian_noise(i_channel, sigma=np.random.uniform(0.01, 0.08))
        else:
            return self.apply_gaussian_blur(i_channel, kernel_size=3)

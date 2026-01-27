"""
Image processor module for resizing, normalizing, and saving images.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class ImageProcessor:
    """
    Handles image processing operations including resizing and normalization.
    """
    
    def __init__(self, output_rgb_dir: Path, output_i_channel_dir: Path):
        """
        Initialize the image processor.
        
        Args:
            output_rgb_dir: Directory to save processed RGB images
            output_i_channel_dir: Directory to save I-channel arrays
        """
        self.output_rgb_dir = Path(output_rgb_dir)
        self.output_i_channel_dir = Path(output_i_channel_dir)
        self.target_size = (256, 256)
        self.epsilon = 1e-8
        
        # Create output directories
        self.output_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.output_i_channel_dir.mkdir(parents=True, exist_ok=True)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to 256x256 using cubic interpolation.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
    
    def normalize_channel(self, channel: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Perform Min-Max normalization on a channel to [0, 1] range.
        
        Includes epsilon to prevent division by zero.
        
        Args:
            channel: Input channel array
            
        Returns:
            Tuple of (normalized_channel, min_value, max_value)
        """
        min_val = float(np.min(channel))
        max_val = float(np.max(channel))
        
        # Normalize to [0, 1] with epsilon to prevent division by zero
        normalized = (channel - min_val) / (max_val - min_val + self.epsilon)
        
        return normalized, min_val, max_val
    
    def save_rgb_image(self, image: np.ndarray, image_id: str) -> bool:
        """
        Save RGB image as PNG file.
        
        Args:
            image: RGB/BGR image array
            image_id: Unique identifier for the image
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            output_path = self.output_rgb_dir / f"{image_id}.png"
            success = cv2.imwrite(str(output_path), image)
            if not success:
                logging.error(f"Failed to save RGB image: {output_path}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error saving RGB image {image_id}: {str(e)}")
            return False
    
    def save_i_channel(self, i_channel: np.ndarray, image_id: str) -> bool:
        """
        Save I-channel as numpy array file (.npy).
        
        Args:
            i_channel: I-channel array
            image_id: Unique identifier for the image
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            output_path = self.output_i_channel_dir / f"{image_id}.npy"
            np.save(output_path, i_channel)
            return True
        except Exception as e:
            logging.error(f"Error saving I-channel {image_id}: {str(e)}")
            return False
    
    def process_and_save(
        self, 
        bgr_image: np.ndarray, 
        yiq_image: np.ndarray, 
        image_id: str
    ) -> Optional[dict]:
        """
        Process images (resize, normalize) and save them.
        
        Args:
            bgr_image: Input BGR image
            yiq_image: Input YIQ image
            image_id: Unique identifier for the image
            
        Returns:
            Dictionary with processing metadata, or None if failed
        """
        try:
            # Resize BGR image to 256x256
            bgr_resized = self.resize_image(bgr_image)
            
            # Resize YIQ image to 256x256
            yiq_resized = self.resize_image(yiq_image)
            
            # Extract and normalize I-channel
            i_channel = yiq_resized[:, :, 1]  # I is the second channel
            i_normalized, i_min, i_max = self.normalize_channel(i_channel)
            
            # Save RGB image
            rgb_success = self.save_rgb_image(bgr_resized, image_id)
            if not rgb_success:
                return None
            
            # Save I-channel
            i_success = self.save_i_channel(i_normalized, image_id)
            if not i_success:
                return None
            
            # Return metadata
            metadata = {
                'original_size': bgr_image.shape[:2],  # (height, width)
                'processed_size': self.target_size,
                'i_channel_min': i_min,
                'i_channel_max': i_max,
                'i_channel_normalized': True
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error processing image {image_id}: {str(e)}")
            return None

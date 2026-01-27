"""
Image loader module for loading and validating images.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


class ImageLoader:
    """
    Handles loading and validation of images from source directories.
    """
    
    def __init__(self, source_dirs: List[Path]):
        """
        Initialize the image loader.
        
        Args:
            source_dirs: List of directories containing source images
        """
        self.source_dirs = [Path(d) for d in source_dirs]
        self.min_size = 256
        
    def get_all_images(self) -> List[Path]:
        """
        Get all image file paths from source directories.
        
        Returns:
            List of paths to image files
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        all_images = []
        
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                logging.warning(f"Source directory does not exist: {source_dir}")
                continue
                
            for ext in image_extensions:
                all_images.extend(source_dir.glob(f"*{ext}"))
                all_images.extend(source_dir.glob(f"*{ext.upper()}"))
        
        # Remove duplicates (case-insensitive file systems may return same file twice)
        all_images = list(set(all_images))
        
        logging.info(f"Found {len(all_images)} images across {len(self.source_dirs)} directories")
        return sorted(all_images)
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array in BGR format, or None if loading failed
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logging.warning(f"Error loading {image_path}: {str(e)}")
            return None
    
    def validate_image(self, image: np.ndarray, image_path: Path) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Validate image meets minimum requirements and convert grayscale to RGB if needed.
        
        Args:
            image: Input image array
            image_path: Path to the image (for logging)
            
        Returns:
            Tuple of (is_valid, processed_image)
            - is_valid: True if image meets requirements
            - processed_image: The validated/converted image, or None if invalid
        """
        if image is None:
            return False, None
        
        # Check dimensions
        height, width = image.shape[:2]
        if height < self.min_size or width < self.min_size:
            logging.warning(
                f"Image {image_path.name} is too small: {width}x{height} "
                f"(minimum {self.min_size}x{self.min_size})"
            )
            return False, None
        
        # Handle grayscale images (convert to RGB/BGR)
        if len(image.shape) == 2:
            logging.info(f"Converting grayscale image to RGB: {image_path.name}")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            logging.info(f"Converting single-channel image to RGB: {image_path.name}")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            logging.info(f"Converting RGBA to RGB: {image_path.name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            logging.warning(f"Unsupported number of channels: {image.shape[2]} for {image_path.name}")
            return False, None
        
        return True, image
    
    def rgb_to_yiq(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR/RGB image to YIQ color space using NTSC formula.
        
        Note: OpenCV loads images in BGR format by default, so we need to
        handle the conversion correctly.
        
        NTSC YIQ conversion formulas:
        Y = 0.299*R + 0.587*G + 0.114*B
        I = 0.596*R - 0.275*G - 0.321*B
        Q = 0.212*R - 0.523*G + 0.311*B
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            YIQ image as numpy array with shape (H, W, 3)
        """
        # OpenCV uses BGR, so we need to extract channels correctly
        B = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        R = image[:, :, 2].astype(np.float32)
        
        # Apply NTSC conversion matrix
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        I = 0.596 * R - 0.275 * G - 0.321 * B
        Q = 0.212 * R - 0.523 * G + 0.311 * B
        
        # Stack channels
        yiq_image = np.stack([Y, I, Q], axis=2)
        
        return yiq_image
    
    def process_image(self, image_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load, validate, and convert an image to YIQ color space.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (bgr_image, yiq_image) or None if processing failed
        """
        # Load image
        bgr_image = self.load_image(image_path)
        if bgr_image is None:
            return None
        
        # Validate and convert if needed
        is_valid, bgr_image = self.validate_image(bgr_image, image_path)
        if not is_valid:
            return None
        
        # Convert to YIQ
        yiq_image = self.rgb_to_yiq(bgr_image)
        
        return bgr_image, yiq_image

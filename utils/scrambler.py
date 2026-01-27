"""
Watermark scrambling module using Arnold Cat Map (ACM) algorithm.

This module provides functionality to scramble and descramble watermark images
using the Arnold Cat Map transformation for security enhancement in watermarking systems.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class WatermarkScrambler:
    """
    Implements Arnold Cat Map (ACM) scrambling for watermark images.
    
    The Arnold Cat Map is a chaotic map that permutes pixel positions in a square
    image, making the watermark visually unrecognizable while preserving all information
    for later extraction.
    """
    
    def __init__(self, default_size: int = 32):
        """
        Initialize the WatermarkScrambler.
        
        Args:
            default_size: Default target size for square watermarks (default: 32x32)
        """
        self.default_size = default_size
        logging.info(f"WatermarkScrambler initialized with default size: {default_size}x{default_size}")
    
    def resize_watermark(
        self, 
        image: np.ndarray, 
        target_size: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Resize watermark to a square image with specified dimensions.
        
        The Arnold Cat Map requires square images (N×N). This method resizes
        the input watermark to the target size while maintaining image quality.
        
        Args:
            image: Input watermark image (grayscale or binary)
            target_size: Target size for square output (default: self.default_size)
            
        Returns:
            Resized square watermark as numpy array, or None if resize fails
        """
        try:
            if target_size is None:
                target_size = self.default_size
            
            # Validate input
            if image is None or image.size == 0:
                logging.error("Invalid input image: empty or None")
                return None
            
            # Ensure 2D grayscale image
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:
                    logging.error(f"Unsupported image shape: {image.shape}")
                    return None
            
            # Resize to square
            resized = cv2.resize(
                image, 
                (target_size, target_size), 
                interpolation=cv2.INTER_CUBIC
            )
            
            logging.info(f"Resized watermark from {image.shape} to {resized.shape}")
            return resized
            
        except Exception as e:
            logging.error(f"Error resizing watermark: {str(e)}")
            return None
    
    def arnold_cat_map(
        self, 
        image: np.ndarray, 
        iterations: int
    ) -> Optional[np.ndarray]:
        """
        Apply Arnold Cat Map transformation to scramble a watermark image.
        
        The transformation uses the equations:
        x' = (x + y) mod N
        y' = (x + 2y) mod N
        
        where (x, y) are original coordinates and (x', y') are scrambled coordinates.
        
        Args:
            image: Input square watermark image (N×N)
            iterations: Number of scrambling iterations (acts as encryption key)
            
        Returns:
            Scrambled watermark image, or None if scrambling fails
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                logging.error("Invalid input image: empty or None")
                return None
            
            # Ensure 2D array
            if len(image.shape) != 2:
                logging.error(f"Image must be 2D (grayscale), got shape: {image.shape}")
                return None
            
            # Ensure square image
            height, width = image.shape
            if height != width:
                logging.error(f"Image must be square, got dimensions: {height}x{width}")
                return None
            
            N = height
            scrambled = image.copy()
            
            # Apply Arnold Cat Map transformation for specified iterations
            for iteration in range(iterations):
                temp = np.zeros_like(scrambled)
                
                for x in range(N):
                    for y in range(N):
                        # Arnold Cat Map transformation
                        x_new = (x + y) % N
                        y_new = (x + 2 * y) % N
                        
                        # Map pixel from (x, y) to (x_new, y_new)
                        temp[x_new, y_new] = scrambled[x, y]
                
                scrambled = temp
            
            logging.info(f"Successfully scrambled {N}x{N} watermark with {iterations} iterations")
            return scrambled
            
        except Exception as e:
            logging.error(f"Error in Arnold Cat Map scrambling: {str(e)}")
            return None
    
    def inverse_arnold_cat_map(
        self, 
        scrambled_image: np.ndarray, 
        iterations: int
    ) -> Optional[np.ndarray]:
        """
        Apply inverse Arnold Cat Map transformation to descramble a watermark.
        
        This perfectly reverses the scrambling process by applying the inverse
        transformation for the same number of iterations.
        
        The inverse transformation uses:
        x = (2x' - y') mod N
        y = (-x' + y') mod N
        
        Args:
            scrambled_image: Scrambled watermark image (N×N)
            iterations: Number of iterations used during scrambling (must match)
            
        Returns:
            Descrambled (original) watermark image, or None if descrambling fails
        """
        try:
            # Validate input
            if scrambled_image is None or scrambled_image.size == 0:
                logging.error("Invalid input image: empty or None")
                return None
            
            # Ensure 2D array
            if len(scrambled_image.shape) != 2:
                logging.error(f"Image must be 2D (grayscale), got shape: {scrambled_image.shape}")
                return None
            
            # Ensure square image
            height, width = scrambled_image.shape
            if height != width:
                logging.error(f"Image must be square, got dimensions: {height}x{width}")
                return None
            
            N = height
            descrambled = scrambled_image.copy()
            
            # Apply inverse Arnold Cat Map transformation for specified iterations
            for iteration in range(iterations):
                temp = np.zeros_like(descrambled)
                
                for x_new in range(N):
                    for y_new in range(N):
                        # Inverse Arnold Cat Map transformation
                        x = (2 * x_new - y_new) % N
                        y = (-x_new + y_new) % N
                        
                        # Map pixel from (x_new, y_new) back to (x, y)
                        temp[x, y] = descrambled[x_new, y_new]
                
                descrambled = temp
            
            logging.info(f"Successfully descrambled {N}x{N} watermark with {iterations} iterations")
            return descrambled
            
        except Exception as e:
            logging.error(f"Error in inverse Arnold Cat Map descrambling: {str(e)}")
            return None


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging for validation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("ARNOLD CAT MAP VALIDATION TEST")
    print("=" * 80)
    
    # Test parameters
    test_sizes = [32, 64]
    test_iterations = [10, 20, 50]
    
    all_tests_passed = True
    
    for size in test_sizes:
        for iterations in test_iterations:
            print(f"\nTesting {size}x{size} watermark with {iterations} iterations...")
            
            # Create test watermark (random grayscale image)
            np.random.seed(42)  # For reproducibility
            original_watermark = np.random.randint(0, 256, (size, size), dtype=np.uint8)
            
            # Initialize scrambler
            scrambler = WatermarkScrambler(default_size=size)
            
            # Scramble the watermark
            scrambled = scrambler.arnold_cat_map(original_watermark, iterations)
            
            if scrambled is None:
                print(f"  ❌ FAILED: Scrambling returned None")
                all_tests_passed = False
                continue
            
            # Verify scrambled image is different from original
            if np.array_equal(original_watermark, scrambled):
                print(f"  ⚠️  WARNING: Scrambled image is identical to original")
            else:
                print(f"  ✓ Scrambled image differs from original")
            
            # Descramble the watermark
            descrambled = scrambler.inverse_arnold_cat_map(scrambled, iterations)
            
            if descrambled is None:
                print(f"  ❌ FAILED: Descrambling returned None")
                all_tests_passed = False
                continue
            
            # Verify perfect reconstruction
            if np.array_equal(original_watermark, descrambled):
                print(f"  ✓ PASSED: Perfect reconstruction verified with np.array_equal()")
                
                # Additional check: compute difference
                diff = np.abs(original_watermark.astype(float) - descrambled.astype(float))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"    - Max pixel difference: {max_diff}")
                print(f"    - Mean pixel difference: {mean_diff}")
                
                if max_diff > 0:
                    print(f"  ⚠️  WARNING: Non-zero difference detected!")
                    all_tests_passed = False
            else:
                print(f"  ❌ FAILED: Reconstruction is not perfect")
                
                # Compute error statistics
                diff = np.abs(original_watermark.astype(float) - descrambled.astype(float))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                nonzero_count = np.count_nonzero(diff)
                
                print(f"    - Max pixel difference: {max_diff}")
                print(f"    - Mean pixel difference: {mean_diff}")
                print(f"    - Pixels with differences: {nonzero_count}/{size*size}")
                
                all_tests_passed = False
    
    # Test resize functionality
    print(f"\n{'=' * 80}")
    print("TESTING RESIZE FUNCTIONALITY")
    print("=" * 80)
    
    # Test non-square image resize
    test_image = np.random.randint(0, 256, (128, 256), dtype=np.uint8)
    scrambler = WatermarkScrambler(default_size=32)
    
    resized = scrambler.resize_watermark(test_image, target_size=32)
    if resized is not None and resized.shape == (32, 32):
        print(f"✓ Successfully resized {test_image.shape} to {resized.shape}")
    else:
        print(f"❌ FAILED: Resize test failed")
        all_tests_passed = False
    
    # Final summary
    print(f"\n{'=' * 80}")
    if all_tests_passed:
        print("✓ ALL VALIDATION TESTS PASSED")
        print("Arnold Cat Map implementation is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above.")
    print("=" * 80)

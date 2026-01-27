"""
Standalone test for Phase 2: Watermark Scrambling.

This script tests only the watermark scrambling phase without running
the full preprocessing pipeline.
"""
import logging
from pathlib import Path
import cv2
import numpy as np

from utils.scrambler import WatermarkScrambler
from utils.metadata_mgr import MetadataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase2_test.log')
    ]
)

def main():
    """Test Phase 2: Watermark Scrambling independently."""
    
    base_dir = Path(__file__).parent
    
    # Setup directories
    dirs = {
        'data_watermark': base_dir / 'data' / 'watermark',
        'data_scrambled': base_dir / 'data' / 'scrambled',
        'preprocessed_metadata': base_dir / 'preprocessed' / 'metadata'
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("PHASE 2: WATERMARK SCRAMBLING (STANDALONE TEST)")
    logging.info("=" * 80)
    
    # Initialize metadata manager
    metadata_mgr = MetadataManager(dirs['preprocessed_metadata'])
    
    # Step 1: Load watermark image
    logging.info("\n" + "=" * 80)
    logging.info("Step 1: Loading watermark image")
    logging.info("=" * 80)
    
    # Check for watermark files
    watermark_files = list(dirs['data_watermark'].glob('*.png')) + \
                     list(dirs['data_watermark'].glob('*.jpg')) + \
                     list(dirs['data_watermark'].glob('*.bmp'))
    
    if not watermark_files:
        logging.warning(f"No watermark images found in {dirs['data_watermark']}")
        logging.warning("Please add a watermark image to data/watermark/ directory")
        logging.warning("Or run: python generate_watermark.py")
        return False
    
    # Use the first watermark found
    watermark_path = watermark_files[0]
    logging.info(f"Found watermark: {watermark_path.name}")
    
    try:
        # Load watermark image
        watermark_original = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
        
        if watermark_original is None:
            logging.error(f"Failed to load watermark image: {watermark_path}")
            return False
        
        logging.info(f"Loaded watermark: {watermark_original.shape}")
        logging.info(f"Value range: {watermark_original.min()} to {watermark_original.max()}")
        
        # Step 2: Initialize scrambler and process watermark
        logging.info("\n" + "=" * 80)
        logging.info("Step 2: Scrambling watermark with Arnold Cat Map")
        logging.info("=" * 80)
        
        # Parameters
        target_watermark_size = 32  # Standard size for embedding
        acm_iterations = 10  # Arnold Cat Map iterations (encryption key)
        
        logging.info(f"Target size: {target_watermark_size}x{target_watermark_size}")
        logging.info(f"ACM iterations (key): {acm_iterations}")
        
        # Initialize scrambler
        scrambler = WatermarkScrambler(default_size=target_watermark_size)
        
        # Store original dimensions
        original_watermark_dims = watermark_original.shape
        
        # Resize to square
        watermark_resized = scrambler.resize_watermark(
            watermark_original, 
            target_size=target_watermark_size
        )
        
        if watermark_resized is None:
            logging.error("Failed to resize watermark")
            return False
        
        logging.info(f"Resized watermark to {watermark_resized.shape}")
        
        # Scramble watermark
        watermark_scrambled = scrambler.arnold_cat_map(
            watermark_resized,
            iterations=acm_iterations
        )
        
        if watermark_scrambled is None:
            logging.error("Failed to scramble watermark")
            return False
        
        # Step 3: Save scrambled watermark
        logging.info("\n" + "=" * 80)
        logging.info("Step 3: Saving scrambled watermark")
        logging.info("=" * 80)
        
        # Generate watermark ID
        watermark_id = watermark_path.stem
        
        # Save as .npy for numerical precision
        scrambled_filename = f"watermark_scrambled_{acm_iterations}iter.npy"
        scrambled_save_path = dirs['data_scrambled'] / scrambled_filename
        
        np.save(scrambled_save_path, watermark_scrambled)
        logging.info(f"Saved scrambled watermark: {scrambled_save_path.name}")
        
        # Also save as PNG for visualization (scale to 0-255)
        scrambled_png = (watermark_scrambled * 255).astype(np.uint8) if watermark_scrambled.max() <= 1 else watermark_scrambled
        scrambled_png_path = dirs['data_scrambled'] / f"watermark_scrambled_{acm_iterations}iter.png"
        cv2.imwrite(str(scrambled_png_path), scrambled_png)
        logging.info(f"Saved visualization: {scrambled_png_path.name}")
        
        # Step 4: Save watermark metadata
        logging.info("\n" + "=" * 80)
        logging.info("Step 4: Saving watermark metadata")
        logging.info("=" * 80)
        
        scrambling_metadata = {
            'algorithm': 'Arnold Cat Map',
            'iterations': acm_iterations,
            'original_dimensions': list(original_watermark_dims),
            'scrambled_size': target_watermark_size,
            'scrambled_path': str(scrambled_save_path)
        }
        
        success = metadata_mgr.save_watermark_metadata(
            watermark_id=watermark_id,
            original_path=watermark_path,
            scrambling_metadata=scrambling_metadata
        )
        
        if success:
            logging.info("Watermark metadata saved successfully")
        else:
            logging.error("Failed to save watermark metadata")
            return False
        
        # Step 5: Verify reconstruction
        logging.info("\n" + "=" * 80)
        logging.info("Step 5: Verifying perfect reconstruction")
        logging.info("=" * 80)
        
        descrambled = scrambler.inverse_arnold_cat_map(watermark_scrambled, acm_iterations)
        
        if descrambled is None:
            logging.error("Descrambling failed")
            return False
        
        is_perfect = np.array_equal(watermark_resized, descrambled)
        
        if is_perfect:
            logging.info("✓ PERFECT RECONSTRUCTION VERIFIED")
        else:
            max_diff = np.max(np.abs(watermark_resized.astype(float) - descrambled.astype(float)))
            logging.error(f"❌ RECONSTRUCTION FAILED! Max difference: {max_diff}")
            return False
        
        # Summary
        logging.info("\n" + "=" * 80)
        logging.info("Phase 2: Watermark Scrambling Summary")
        logging.info("=" * 80)
        logging.info(f"Original watermark: {watermark_path.name}")
        logging.info(f"Original dimensions: {original_watermark_dims}")
        logging.info(f"Scrambled size: {target_watermark_size}x{target_watermark_size}")
        logging.info(f"Algorithm: Arnold Cat Map")
        logging.info(f"Iterations (key): {acm_iterations}")
        logging.info(f"Output (NPY): {scrambled_save_path.name}")
        logging.info(f"Output (PNG): {scrambled_png_path.name}")
        logging.info(f"Metadata: watermark_{watermark_id}.json")
        
        logging.info("\n" + "=" * 80)
        logging.info("✓ PHASE 2 COMPLETE - ALL TESTS PASSED")
        logging.info("=" * 80)
        
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error in Phase 2: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Phase 2 completed successfully!")
        print("Check data/scrambled/ for output files")
        print("Check preprocessed/metadata/ for watermark metadata")
    else:
        print("\n❌ Phase 2 failed - check phase2_test.log for details")

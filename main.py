"""
Main entry point for Phase 1: Data Preprocessing Pipeline.

This script:
1. Automatically downloads DIV2K dataset if not present
2. Loads and validates images from DIV2K and BOSSBase (if available)
3. Converts images to YIQ color space
4. Resizes to 256x256 and normalizes I-channel
5. Saves processed images and metadata
6. Creates train/val/test splits
"""
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm

from utils.downloader import DatasetDownloader
from utils.loader import ImageLoader
from utils.processor import ImageProcessor
from utils.metadata_mgr import MetadataManager, create_splits
from utils.scrambler import WatermarkScrambler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)


def setup_directories(base_dir: Path) -> dict:
    """
    Create all necessary directories for the preprocessing pipeline.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Dictionary containing paths to all directories
    """
    dirs = {
        'data_raw': base_dir / 'data' / 'raw',
        'data_raw_div2k': base_dir / 'data' / 'raw' / 'div2k',
        'data_raw_bossbase': base_dir / 'data' / 'raw' / 'bossbase',
        'data_watermark': base_dir / 'data' / 'watermark',
        'data_scrambled': base_dir / 'data' / 'scrambled',
        'preprocessed': base_dir / 'preprocessed',
        'preprocessed_rgb': base_dir / 'preprocessed' / 'rgb_256',
        'preprocessed_i_channel': base_dir / 'preprocessed' / 'I_channel',
        'preprocessed_metadata': base_dir / 'preprocessed' / 'metadata',
        'splits': base_dir / 'splits'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Created all necessary directories")
    return dirs


def get_source_directories(dirs: dict) -> List[Path]:
    """
    Get list of source directories that exist and contain images.
    
    Args:
        dirs: Dictionary of directory paths
        
    Returns:
        List of valid source directories
    """
    source_dirs = []
    
    # Check for DIV2K images (either directly in div2k/ or in DIV2K_train_HR/ subdirectory)
    div2k_base = dirs['data_raw_div2k']
    div2k_hr = div2k_base / 'DIV2K_train_HR'
    
    if div2k_hr.exists() and list(div2k_hr.glob('*.png')):
        source_dirs.append(div2k_hr)
        logging.info(f"Added DIV2K source: {div2k_hr}")
    elif div2k_base.exists() and list(div2k_base.glob('*.png')):
        source_dirs.append(div2k_base)
        logging.info(f"Added DIV2K source: {div2k_base}")
    
    # Check for BOSSBase (manually placed)
    if dirs['data_raw_bossbase'].exists():
        png_files = list(dirs['data_raw_bossbase'].glob('*.png'))
        if png_files:
            source_dirs.append(dirs['data_raw_bossbase'])
            logging.info(f"Added BOSSBase source: {dirs['data_raw_bossbase']} ({len(png_files)} images)")
    
    return source_dirs


def generate_image_id(image_path: Path, counter: int) -> str:
    """
    Generate a unique image ID.
    
    Args:
        image_path: Path to the image file
        counter: Sequential counter
        
    Returns:
        Unique image ID
    """
    # Use stem (filename without extension) and add counter for uniqueness
    base_name = image_path.stem
    return f"{base_name}_{counter:05d}"


def main():
    """
    Main preprocessing pipeline execution.
    """
    logging.info("="*80)
    logging.info("Starting Phase 1: Data Preprocessing Pipeline")
    logging.info("="*80)
    
    # Get base directory (current working directory)
    base_dir = Path.cwd()
    logging.info(f"Base directory: {base_dir}")
    
    # Setup all directories
    dirs = setup_directories(base_dir)
    
    # Step 1: Check DIV2K dataset
    logging.info("\n" + "="*80)
    logging.info("Step 1: Checking DIV2K dataset")
    logging.info("="*80)
    
    downloader = DatasetDownloader(dirs['data_raw'])
    div2k_path = downloader.download_div2k(force_redownload=False)
    
    if div2k_path is None:
        logging.error("DIV2K dataset not found. Please download manually. Exiting.")
        return
    
    # Step 2: Initialize processing components
    logging.info("\n" + "="*80)
    logging.info("Step 2: Initializing processing components")
    logging.info("="*80)
    
    source_dirs = get_source_directories(dirs)
    if not source_dirs:
        logging.error("No source directories found. Exiting.")
        return
    
    loader = ImageLoader(source_dirs)
    processor = ImageProcessor(
        output_rgb_dir=dirs['preprocessed_rgb'],
        output_i_channel_dir=dirs['preprocessed_i_channel']
    )
    metadata_mgr = MetadataManager(metadata_dir=dirs['preprocessed_metadata'])
    
    # Step 3: Process all images
    logging.info("\n" + "="*80)
    logging.info("Step 3: Processing images")
    logging.info("="*80)
    
    all_image_paths = loader.get_all_images()
    successfully_processed_ids = []
    failed_images = []
    
    for idx, image_path in enumerate(tqdm(all_image_paths, desc="Processing images")):
        try:
            # Generate unique ID
            image_id = generate_image_id(image_path, idx)
            
            # Load and convert to YIQ
            result = loader.process_image(image_path)
            if result is None:
                failed_images.append((image_path, "Failed to load/validate"))
                continue
            
            bgr_image, yiq_image = result
            
            # Process and save
            processing_metadata = processor.process_and_save(bgr_image, yiq_image, image_id)
            if processing_metadata is None:
                failed_images.append((image_path, "Failed to process/save"))
                continue
            
            # Save metadata
            success = metadata_mgr.save_image_metadata(
                image_id, 
                image_path, 
                processing_metadata
            )
            if not success:
                failed_images.append((image_path, "Failed to save metadata"))
                continue
            
            successfully_processed_ids.append(image_id)
            
        except Exception as e:
            logging.error(f"Unexpected error processing {image_path}: {str(e)}")
            failed_images.append((image_path, f"Unexpected error: {str(e)}"))
    
    # Step 4: Create train/val/test splits
    logging.info("\n" + "="*80)
    logging.info("Step 4: Creating dataset splits")
    logging.info("="*80)
    
    if successfully_processed_ids:
        train_ids, val_ids, test_ids = create_splits(
            all_ids=successfully_processed_ids,
            splits_dir=dirs['splits'],
            ratios=[0.7, 0.15, 0.15],
            random_seed=42
        )
    else:
        logging.error("No images were successfully processed. Cannot create splits.")
    
    # Step 5: Summary
    logging.info("\n" + "="*80)
    logging.info("Preprocessing Summary")
    logging.info("="*80)
    logging.info(f"Total images found: {len(all_image_paths)}")
    logging.info(f"Successfully processed: {len(successfully_processed_ids)}")
    logging.info(f"Failed: {len(failed_images)}")
    
    if successfully_processed_ids:
        logging.info(f"\nDataset splits:")
        logging.info(f"  Training: {len(train_ids)} images")
        logging.info(f"  Validation: {len(val_ids)} images")
        logging.info(f"  Test: {len(test_ids)} images")
    
    if failed_images:
        logging.info("\nFailed images:")
        for img_path, reason in failed_images[:10]:  # Show first 10
            logging.info(f"  {img_path.name}: {reason}")
        if len(failed_images) > 10:
            logging.info(f"  ... and {len(failed_images) - 10} more")
    
    logging.info("\n" + "="*80)
    logging.info("Phase 1: Data Preprocessing Complete!")
    logging.info("="*80)
    
    # Save failed images list for reference
    if failed_images:
        failed_log_path = base_dir / 'failed_images.txt'
        with open(failed_log_path, 'w') as f:
            f.write("Failed Images Log\n")
            f.write("="*80 + "\n\n")
            for img_path, reason in failed_images:
                f.write(f"{img_path}: {reason}\n")
        logging.info(f"Failed images list saved to: {failed_log_path}")
    
    # ============================================================================
    # PHASE 2: WATERMARK SCRAMBLING
    # ============================================================================
    
    logging.info("\n\n" + "="*80)
    logging.info("PHASE 2: WATERMARK SCRAMBLING")
    logging.info("="*80)
    
    # Step 1: Load watermark image
    logging.info("\n" + "="*80)
    logging.info("Step 1: Loading watermark image")
    logging.info("="*80)
    
    # Check if watermark directory exists and contains images
    watermark_files = list(dirs['data_watermark'].glob('*.png')) + \
                     list(dirs['data_watermark'].glob('*.jpg')) + \
                     list(dirs['data_watermark'].glob('*.bmp'))
    
    if not watermark_files:
        logging.warning(f"No watermark images found in {dirs['data_watermark']}")
        logging.warning("Please add a watermark image to data/watermark/ directory")
        logging.warning("Skipping Phase 2: Watermark Scrambling")
    else:
        # Use the first watermark found
        watermark_path = watermark_files[0]
        logging.info(f"Found watermark: {watermark_path.name}")
        
        try:
            # Load watermark image
            import cv2
            import numpy as np
            
            watermark_original = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
            
            if watermark_original is None:
                logging.error(f"Failed to load watermark image: {watermark_path}")
            else:
                logging.info(f"Loaded watermark: {watermark_original.shape}")
                
                # Step 2: Initialize scrambler and process watermark
                logging.info("\n" + "="*80)
                logging.info("Step 2: Scrambling watermark with Arnold Cat Map")
                logging.info("="*80)
                
                # Parameters
                target_watermark_size = 32  # Standard size for embedding
                acm_iterations = 10  # Arnold Cat Map iterations (encryption key)
                
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
                else:
                    logging.info(f"Resized watermark to {watermark_resized.shape}")
                    
                    # Scramble watermark
                    watermark_scrambled = scrambler.arnold_cat_map(
                        watermark_resized,
                        iterations=acm_iterations
                    )
                    
                    if watermark_scrambled is None:
                        logging.error("Failed to scramble watermark")
                    else:
                        # Step 3: Save scrambled watermark
                        logging.info("\n" + "="*80)
                        logging.info("Step 3: Saving scrambled watermark")
                        logging.info("="*80)
                        
                        # Generate watermark ID
                        watermark_id = watermark_path.stem
                        
                        # Save as .npy for numerical precision
                        scrambled_filename = f"watermark_scrambled_{acm_iterations}iter.npy"
                        scrambled_save_path = dirs['data_scrambled'] / scrambled_filename
                        
                        try:
                            np.save(scrambled_save_path, watermark_scrambled)
                            logging.info(f"Saved scrambled watermark: {scrambled_save_path.name}")
                            
                            # Step 4: Save watermark metadata
                            logging.info("\n" + "="*80)
                            logging.info("Step 4: Saving watermark metadata")
                            logging.info("="*80)
                            
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
                            
                            # Summary
                            logging.info("\n" + "="*80)
                            logging.info("Phase 2: Watermark Scrambling Summary")
                            logging.info("="*80)
                            logging.info(f"Original watermark: {watermark_path.name}")
                            logging.info(f"Original dimensions: {original_watermark_dims}")
                            logging.info(f"Scrambled size: {target_watermark_size}x{target_watermark_size}")
                            logging.info(f"Algorithm: Arnold Cat Map")
                            logging.info(f"Iterations (key): {acm_iterations}")
                            logging.info(f"Output: {scrambled_save_path.name}")
                            
                        except Exception as e:
                            logging.error(f"Error saving scrambled watermark: {str(e)}")
                
        except Exception as e:
            logging.error(f"Unexpected error in Phase 2: {str(e)}")
    
    logging.info("\n" + "="*80)
    logging.info("ALL PHASES COMPLETE!")
    logging.info("="*80)


if __name__ == "__main__":
    main()

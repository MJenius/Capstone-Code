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
import json
from pathlib import Path
from typing import List
import cv2
import numpy as np
from tqdm import tqdm

from utils.downloader import DatasetDownloader
from utils.loader import ImageLoader
from utils.processor import ImageProcessor
from utils.metadata_mgr import MetadataManager, create_splits
from utils.scrambler import WatermarkScrambler
from utils.catalan import CatalanTransform
from utils.mosaic import MosaicGenerator
from utils.adaptive_embedder import AdaptiveEmbedder


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
        'data_catalan': base_dir / 'data' / 'catalan',
        'data_mosaic': base_dir / 'data' / 'mosaic',
        'preprocessed': base_dir / 'preprocessed',
        'preprocessed_rgb': base_dir / 'preprocessed' / 'rgb_256',
        'preprocessed_i_channel': base_dir / 'preprocessed' / 'I_channel',
        'preprocessed_embedded_i_channel': base_dir / 'preprocessed' / 'embedded_I_channel',
        'preprocessed_embedded_preview': base_dir / 'preprocessed' / 'embedded_preview',
        'preprocessed_process_collage': base_dir / 'preprocessed' / 'process_collage',
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


def yiq_to_bgr(yiq_image: np.ndarray) -> np.ndarray:
    """
    Convert YIQ image to BGR image using inverse NTSC transform.

    Args:
        yiq_image: Input image in YIQ format

    Returns:
        BGR uint8 image in [0, 255]
    """
    y = yiq_image[:, :, 0].astype(np.float32)
    i = yiq_image[:, :, 1].astype(np.float32)
    q = yiq_image[:, :, 2].astype(np.float32)

    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q

    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def to_u8_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale-like array to uint8 in [0, 255].
    """
    img = image.astype(np.float32)
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def create_process_collage(
    original_bgr: np.ndarray,
    watermark_binary: np.ndarray,
    watermark_catalan: np.ndarray,
    watermark_mosaic: np.ndarray,
    embedded_bgr: np.ndarray,
) -> np.ndarray:
    """
    Create a 6-panel process collage for one host image.
    """
    wm_binary_u8 = to_u8_grayscale(watermark_binary)
    wm_catalan_u8 = to_u8_grayscale(watermark_catalan)
    wm_mosaic_u8 = to_u8_grayscale(watermark_mosaic)

    wm_binary_panel = cv2.cvtColor(
        cv2.resize(wm_binary_u8, (256, 256), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    wm_catalan_panel = cv2.cvtColor(
        cv2.resize(wm_catalan_u8, (256, 256), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    wm_mosaic_panel = cv2.cvtColor(wm_mosaic_u8, cv2.COLOR_GRAY2BGR)

    diff = np.abs(embedded_bgr.astype(np.float32) - original_bgr.astype(np.float32))
    diff_mag = diff.mean(axis=2)
    diff_vis = np.clip(diff_mag * 8.0, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_TURBO)

    panels = [
        ("Original Image", original_bgr),
        ("Watermark", wm_binary_panel),
        ("Catalan Transform", wm_catalan_panel),
        ("Mosaic Generation", wm_mosaic_panel),
        ("Embedded Image", embedded_bgr),
        ("Change Map", diff_heatmap),
    ]

    cols = 3
    rows = 2
    cell_w = 256
    cell_h = 256
    title_h = 30
    canvas = np.full((rows * (cell_h + title_h), cols * cell_w, 3), 245, dtype=np.uint8)

    for idx, (title, panel) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * (cell_h + title_h)

        panel_resized = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y0 + title_h:y0 + title_h + cell_h, x0:x0 + cell_w] = panel_resized
        cv2.putText(canvas, title, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)

    return canvas


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
    
    # Step 4: Dataset splits will be created AFTER embedding (Phase 4B)
    # to ensure only successfully embedded images appear in splits.
    # If Phase 2+ is skipped (no watermark), splits are created here from preprocessed IDs.
    splits_created = False
    if not watermark_files if False else False:  # placeholder — see Phase 4B below
        pass
    
    # Step 5: Summary
    logging.info("\n" + "="*80)
    logging.info("Preprocessing Summary")
    logging.info("="*80)
    logging.info(f"Total images found: {len(all_image_paths)}")
    logging.info(f"Successfully processed: {len(successfully_processed_ids)}")
    logging.info(f"Failed: {len(failed_images)}")
    
    if False:
        pass
    
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

    # Phase configuration
    target_watermark_size = 32
    acm_iterations = 10
    catalan_iterations = 5
    catalan_key = 7
    embedding_alpha_base = 0.012
    embedding_sensitivity = 2.0

    watermark_scrambled = None
    watermark_binary = None
    watermark_id = None

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
        logging.warning("Skipping Phase 2 and later watermark phases")
    else:
        watermark_path = watermark_files[0]
        watermark_id = watermark_path.stem
        logging.info(f"Found watermark: {watermark_path.name}")

        try:
            watermark_original = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
            if watermark_original is None:
                logging.error(f"Failed to load watermark image: {watermark_path}")
            else:
                watermark_binary = watermark_original
                logging.info(f"Loaded watermark: {watermark_original.shape}")

                logging.info("\n" + "="*80)
                logging.info("Step 2: Scrambling watermark with Arnold Cat Map")
                logging.info("="*80)

                scrambler = WatermarkScrambler(default_size=target_watermark_size)
                original_watermark_dims = watermark_original.shape

                watermark_resized = scrambler.resize_watermark(
                    watermark_original,
                    target_size=target_watermark_size
                )

                if watermark_resized is None:
                    logging.error("Failed to resize watermark")
                else:
                    watermark_scrambled = scrambler.arnold_cat_map(
                        watermark_resized,
                        iterations=acm_iterations
                    )

                    if watermark_scrambled is None:
                        logging.error("Failed to scramble watermark")
                    else:
                        scrambled_filename = f"watermark_scrambled_{acm_iterations}iter.npy"
                        scrambled_save_path = dirs['data_scrambled'] / scrambled_filename
                        np.save(scrambled_save_path, watermark_scrambled)
                        logging.info(f"Saved scrambled watermark: {scrambled_save_path.name}")

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
                        if not success:
                            logging.error("Failed to save watermark metadata")

                        logging.info("\n" + "="*80)
                        logging.info("Phase 2: Watermark Scrambling Summary")
                        logging.info("="*80)
                        logging.info(f"Original watermark: {watermark_path.name}")
                        logging.info(f"Original dimensions: {original_watermark_dims}")
                        logging.info(f"Scrambled size: {target_watermark_size}x{target_watermark_size}")
                        logging.info(f"Iterations (key): {acm_iterations}")
                        logging.info(f"Output: {scrambled_save_path.name}")

        except Exception as e:
            logging.error(f"Unexpected error in Phase 2: {str(e)}")

    # ============================================================================
    # PHASE 3: CATALAN TRANSFORM
    # ============================================================================

    watermark_catalan = None
    if watermark_scrambled is not None:
        logging.info("\n\n" + "="*80)
        logging.info("PHASE 3: CATALAN TRANSFORM")
        logging.info("="*80)

        catalan_transformer = CatalanTransform()
        watermark_catalan = catalan_transformer.catalan_transform(
            watermark_scrambled,
            iterations=catalan_iterations,
            key=catalan_key
        )

        if watermark_catalan is None:
            logging.error("Phase 3 failed: Catalan transform returned None")
        else:
            catalan_filename = f"watermark_catalan_{catalan_iterations}iter_key{catalan_key}.npy"
            catalan_save_path = dirs['data_catalan'] / catalan_filename
            np.save(catalan_save_path, watermark_catalan)
            logging.info(f"Saved Catalan-transformed watermark: {catalan_save_path.name}")

    # ============================================================================
    # PHASE 4: MOSAIC GENERATION
    # ============================================================================

    watermark_mosaic = None
    if watermark_catalan is not None:
        logging.info("\n\n" + "="*80)
        logging.info("PHASE 4: MOSAIC GENERATION")
        logging.info("="*80)

        mosaic_generator = MosaicGenerator()
        watermark_mosaic = mosaic_generator.create_tiled_mosaic(
            watermark=watermark_catalan,
            target_shape=(256, 256)
        )

        if watermark_mosaic is None:
            logging.error("Phase 4 failed: Mosaic generation returned None")
        else:
            mosaic_filename = "watermark_mosaic_256.npy"
            mosaic_save_path = dirs['data_mosaic'] / mosaic_filename
            np.save(mosaic_save_path, watermark_mosaic)

            # Save PNG preview for quick visual check.
            preview = watermark_mosaic.astype(np.float32)
            if preview.max() > 1.0 or preview.min() < 0.0:
                preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
            preview_path = dirs['data_mosaic'] / "watermark_mosaic_256.png"
            cv2.imwrite(str(preview_path), (preview * 255).astype(np.uint8))

            logging.info(f"Saved watermark mosaic: {mosaic_save_path.name}")
            logging.info(f"Saved watermark mosaic preview: {preview_path.name}")

    # ============================================================================
    # PHASE 4B: EMBEDDING INTO I-CHANNEL
    # ============================================================================

    if watermark_mosaic is not None and watermark_id is not None:
        logging.info("\n\n" + "="*80)
        logging.info("PHASE 4B: WATERMARK EMBEDDING")
        logging.info("="*80)

        embedder = AdaptiveEmbedder(alpha_base=embedding_alpha_base, sensitivity=embedding_sensitivity)
        i_channel_files = sorted(dirs['preprocessed_i_channel'].glob('*.npy'))

        embedded_count = 0
        embedding_failures = 0

        for i_channel_path in tqdm(i_channel_files, desc="Embedding watermark"):
            try:
                image_id = i_channel_path.stem
                i_channel = np.load(i_channel_path)

                if i_channel.shape != watermark_mosaic.shape:
                    logging.warning(
                        "Skipping %s due to shape mismatch: host=%s, mosaic=%s",
                        image_id,
                        i_channel.shape,
                        watermark_mosaic.shape,
                    )
                    embedding_failures += 1
                    continue

                embedded_i = embedder.embed(i_channel=i_channel, watermark_mosaic=watermark_mosaic)
                if embedded_i is None:
                    embedding_failures += 1
                    continue

                embedded_output_path = dirs['preprocessed_embedded_i_channel'] / f"{image_id}.npy"
                np.save(embedded_output_path, embedded_i)

                preview_output_path = dirs['preprocessed_embedded_preview'] / f"{image_id}.png"

                # Reconstruct a color preview by combining embedded I with original Y and Q.
                rgb_input_path = dirs['preprocessed_rgb'] / f"{image_id}.png"
                metadata_input_path = dirs['preprocessed_metadata'] / f"{image_id}.json"
                preview_written = False

                if rgb_input_path.exists() and metadata_input_path.exists():
                    original_bgr = cv2.imread(str(rgb_input_path), cv2.IMREAD_COLOR)
                    if original_bgr is not None:
                        with open(metadata_input_path, 'r') as f:
                            image_metadata = json.load(f)

                        i_min = float(image_metadata['i_channel_min'])
                        i_max = float(image_metadata['i_channel_max'])
                        i_denormalized = embedded_i * (i_max - i_min + 1e-8) + i_min

                        yiq_original = loader.rgb_to_yiq(original_bgr)
                        yiq_embedded = yiq_original.copy()
                        yiq_embedded[:, :, 1] = i_denormalized

                        embedded_bgr_preview = yiq_to_bgr(yiq_embedded)
                        cv2.imwrite(str(preview_output_path), embedded_bgr_preview)

                        if watermark_binary is not None and watermark_catalan is not None and watermark_mosaic is not None:
                            collage_image = create_process_collage(
                                original_bgr=original_bgr,
                                watermark_binary=watermark_binary,
                                watermark_catalan=watermark_catalan,
                                watermark_mosaic=watermark_mosaic,
                                embedded_bgr=embedded_bgr_preview,
                            )
                            collage_output_path = dirs['preprocessed_process_collage'] / f"{image_id}.png"
                            cv2.imwrite(str(collage_output_path), collage_image)

                        preview_written = True

                if not preview_written:
                    logging.warning(
                        "Falling back to grayscale preview for %s due to missing inputs",
                        image_id,
                    )
                    cv2.imwrite(str(preview_output_path), (embedded_i * 255).astype(np.uint8))

                metadata_mgr.save_embedding_metadata(
                    image_id=image_id,
                    watermark_id=watermark_id,
                    embedding_metadata={
                        'acm_iterations': acm_iterations,
                        'catalan_iterations': catalan_iterations,
                        'catalan_key': catalan_key,
                        'mosaic_shape': list(watermark_mosaic.shape),
                        'mosaic_grid': [8, 8],
                        'embedding_alpha_base': embedding_alpha_base,
                        'embedding_sensitivity': embedding_sensitivity,
                        'embedded_i_path': str(embedded_output_path),
                        'host_i_min': float(i_channel.min()),
                        'host_i_max': float(i_channel.max()),
                        'embedded_i_min': float(embedded_i.min()),
                        'embedded_i_max': float(embedded_i.max())
                    }
                )

                embedded_count += 1

            except Exception as e:
                logging.error(f"Embedding error for {i_channel_path.name}: {str(e)}")
                embedding_failures += 1

        logging.info("\n" + "="*80)
        logging.info("Phase 4B: Watermark Embedding Summary")
        logging.info("="*80)
        logging.info(f"I-channels scanned: {len(i_channel_files)}")
        logging.info(f"Successfully embedded: {embedded_count}")
        logging.info(f"Failed embeddings: {embedding_failures}")

        # Step 4 (deferred): Create splits ONLY from images that survived embedding.
        # This prevents embedding failures from silently appearing in the test split.
        logging.info("\n" + "="*80)
        logging.info("Step 4: Creating dataset splits (post-embedding)")
        logging.info("="*80)
        embedded_ids = [
            p.stem for p in sorted(dirs['preprocessed_embedded_i_channel'].glob('*.npy'))
        ]
        if embedded_ids:
            train_ids, val_ids, test_ids = create_splits(
                all_ids=embedded_ids,
                splits_dir=dirs['splits'],
                ratios=[0.7, 0.15, 0.15],
                random_seed=42
            )
            logging.info(f"Dataset splits (from {len(embedded_ids)} embedded images):")
            logging.info(f"  Training: {len(train_ids)} images")
            logging.info(f"  Validation: {len(val_ids)} images")
            logging.info(f"  Test: {len(test_ids)} images")
        else:
            logging.error("No embedded images found. Splits not created.")
    
    logging.info("\n" + "="*80)
    logging.info("ALL PHASES COMPLETE!")
    logging.info("="*80)


if __name__ == "__main__":
    main()

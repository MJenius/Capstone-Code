"""
Standalone test for Phase 3 + Phase 4:
- Catalan transform
- Mosaic generation (8x8 tiling to 256x256)
- Embedding into normalized I-channel
"""
import logging
from pathlib import Path

import numpy as np

from utils.catalan import CatalanTransform
from utils.embedder import WatermarkEmbedder
from utils.mosaic import MosaicGenerator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase3_mosaic_embedding_test.log')
    ]
)


def main() -> bool:
    base_dir = Path(__file__).parent

    scrambled_path = base_dir / 'data' / 'scrambled' / 'watermark_scrambled_10iter.npy'
    if not scrambled_path.exists():
        logging.error(f"Missing scrambled watermark: {scrambled_path}")
        logging.error("Run Phase 2 first (python test_phase2.py or python main.py)")
        return False

    scrambled = np.load(scrambled_path)
    logging.info(f"Loaded scrambled watermark shape: {scrambled.shape}")

    if scrambled.shape != (32, 32):
        logging.error(f"Expected scrambled watermark shape (32, 32), got {scrambled.shape}")
        return False

    catalan_iterations = 5
    catalan_key = 7
    alpha = 0.08

    catalan = CatalanTransform()
    transformed = catalan.catalan_transform(scrambled, iterations=catalan_iterations, key=catalan_key)
    if transformed is None:
        logging.error("Catalan transform failed")
        return False

    recovered = catalan.inverse_catalan_transform(
        transformed,
        iterations=catalan_iterations,
        key=catalan_key
    )
    if recovered is None:
        logging.error("Inverse Catalan transform failed")
        return False

    if not np.array_equal(scrambled, recovered):
        logging.error("Catalan transform is not perfectly reversible")
        return False

    logging.info("Catalan transform reversibility check passed")

    mosaic_generator = MosaicGenerator()
    mosaic = mosaic_generator.create_tiled_mosaic(transformed, target_shape=(256, 256))
    if mosaic is None:
        logging.error("Mosaic generation failed")
        return False

    if mosaic.shape != (256, 256):
        logging.error(f"Expected mosaic shape (256, 256), got {mosaic.shape}")
        return False

    expected_top_left = transformed
    actual_top_left = mosaic[:32, :32]
    if not np.array_equal(expected_top_left, actual_top_left):
        logging.error("Mosaic tiling validation failed on top-left tile")
        return False

    logging.info("Mosaic shape and tile consistency check passed")

    # Use an existing I-channel if available, else synthetic normalized host.
    i_channel_dir = base_dir / 'preprocessed' / 'I_channel'
    i_files = sorted(i_channel_dir.glob('*.npy'))

    if i_files:
        host = np.load(i_files[0]).astype(np.float32)
        logging.info(f"Using real host I-channel: {i_files[0].name}")
    else:
        rng = np.random.default_rng(42)
        host = rng.random((256, 256), dtype=np.float32)
        logging.info("Using synthetic host I-channel for embedding test")

    if host.shape != (256, 256):
        logging.error(f"Expected host shape (256, 256), got {host.shape}")
        return False

    embedder = WatermarkEmbedder(alpha=alpha)
    embedded = embedder.embed(host, mosaic)
    if embedded is None:
        logging.error("Embedding failed")
        return False

    if embedded.shape != host.shape:
        logging.error(f"Embedded shape mismatch: {embedded.shape} vs {host.shape}")
        return False

    if float(embedded.min()) < 0.0 or float(embedded.max()) > 1.0:
        logging.error(
            "Embedded I-channel out of bounds: min=%.6f max=%.6f",
            float(embedded.min()),
            float(embedded.max()),
        )
        return False

    logging.info("Embedding range check passed")
    logging.info("All Phase 3 + Mosaic + Embedding tests passed")
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print('\nPASS: Phase 3 + Mosaic + Embedding validation succeeded')
    else:
        print('\nFAIL: Phase 3 + Mosaic + Embedding validation failed')

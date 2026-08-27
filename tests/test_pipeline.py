"""
Integration tests for the complete watermarking pipeline,
verification integrity, and benchmarking execution.
"""
from pathlib import Path
import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.scrambler import WatermarkScrambler
from utils.catalan import CatalanTransform
from utils.mosaic import MosaicGenerator
from utils.adaptive_embedder import AdaptiveEmbedder
from attacks.cropping import CroppingAttack
from benchmark import PreANNBenchmarker, calculate_nc, calculate_ber


def test_full_pipeline_roundtrip():
    """
    Test complete mathematical pipeline roundtrip:
    Original Watermark -> ACM Scramble -> Catalan Permute -> 8x8 Mosaic
    -> Adaptive Embedding -> Host Subtraction -> Inverse Catalan -> Inverse ACM
    -> Bit-exact Watermark Recovery
    """
    rng = np.random.RandomState(12345)
    orig_wm = rng.randint(0, 2, (32, 32), dtype=np.uint8)

    # 1. ACM Scrambling
    scrambler = WatermarkScrambler(default_size=(32, 32))
    scrambled = scrambler.arnold_cat_map(orig_wm, iterations=10)

    # 2. Catalan Permutation
    catalan = CatalanTransform()
    permuted = catalan.catalan_transform(scrambled, iterations=5, key=7)

    # 3. 8x8 Mosaic Generation
    mosaic_gen = MosaicGenerator()
    mosaic = mosaic_gen.create_tiled_mosaic(permuted, target_shape=(256, 256))
    assert mosaic.shape == (256, 256)

    # 4. Adaptive Embedding
    alpha_base = 0.012
    embedder = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=2.0)
    
    host_files = sorted(list(Path('preprocessed/I_channel').glob('*.npy')))
    if host_files:
        host = np.load(host_files[0]).astype(np.float32)
    else:
        host = np.full((256, 256), 0.5, dtype=np.float32)
        
    watermarked = embedder.embed(host, mosaic.astype(np.float32))

    assert psnr(host, watermarked, data_range=1.0) >= 40.0

    # 5. Non-Blind Residual Extraction
    diff = (watermarked - host) / alpha_base + 0.5
    tiles = diff.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
    recovered_permuted = np.mean(tiles, axis=0)

    # 6. Binary thresholding
    recovered_binary = (recovered_permuted > 0.5).astype(np.uint8)

    # 7. Inverse Transforms
    inv_catalan = catalan.inverse_catalan_transform(recovered_binary, iterations=5, key=7)
    inv_scrambled = scrambler.inverse_arnold_cat_map(inv_catalan, iterations=10)

    # Bit-exact recovery
    assert np.array_equal(orig_wm, inv_scrambled)
    assert calculate_ber(orig_wm, inv_scrambled) == 0.0
    assert calculate_nc(orig_wm, inv_scrambled) == pytest.approx(1.0, abs=1e-5)


def test_mosaic_redundancy_under_cropping():
    """
    Test that 8x8 mosaic redundancy allows near-perfect recovery
    even when 25% of the image area is cropped.
    """
    base_dir = Path.cwd()
    wm_binary_path = base_dir / 'data' / 'watermark' / 'watermark_binary.npy'
    if not wm_binary_path.exists():
        pytest.skip("Watermark binary file not present")

    catalan_dir = base_dir / 'data' / 'catalan'
    catalan_files = sorted(list(catalan_dir.glob('*.npy'))) if catalan_dir.exists() else []
    if not catalan_files:
        pytest.skip("Catalan watermark not present")

    wm_catalan = np.load(catalan_files[0])
    wm_mosaic = np.tile(wm_catalan, (8, 8))

    embedder = AdaptiveEmbedder(alpha_base=0.012, sensitivity=2.0)
    cropper = CroppingAttack(target_size=256)

    host = np.full((256, 256), 0.5, dtype=np.float32)
    watermarked = embedder.embed(host, wm_mosaic)

    # 25% crop
    atk = cropper.apply_attack(watermarked, mode='random', intensity=0.25, seed=42)
    mask = cropper.get_mask(mode='random', intensity=0.25, seed=42)

    diff = (atk - host) / 0.012 + 0.5
    tiles_diff = diff.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
    tiles_mask = mask.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)

    sum_diff = np.sum(tiles_diff * tiles_mask, axis=0)
    sum_weights = np.sum(tiles_mask, axis=0)
    recovered = np.where(sum_weights > 0, sum_diff / (sum_weights + 1e-8), 0.5)

    nc = calculate_nc(wm_catalan, recovered)
    assert nc >= 0.99


def test_benchmark_smoke_test(tmp_path):
    """
    Test running benchmark on 2 images to verify output structure and formatting.
    """
    res_file = tmp_path / "smoke_results.json"
    curve_file = tmp_path / "smoke_curve.json"

    benchmarker = PreANNBenchmarker()
    results = benchmarker.run_benchmark(
        num_images=2,
        results_path=str(res_file),
        collusion_curve_path=str(curve_file)
    )

    assert len(results) == 2 * 14  # 14 attack conditions per image
    assert res_file.exists()
    assert curve_file.exists()

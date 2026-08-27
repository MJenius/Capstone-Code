"""
Comprehensive unit tests for watermarking transforms, adaptive embedding,
attack engines, and evaluation metrics.
"""
from pathlib import Path
import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from utils.catalan import CatalanTransform
from utils.scrambler import WatermarkScrambler
from utils.mosaic import MosaicGenerator
from utils.adaptive_embedder import AdaptiveEmbedder
from attacks.cropping import CroppingAttack
from attacks.signal import SignalAttack
from attacks.collusion import CollusionAttack


def calculate_nc(w1: np.ndarray, w2: np.ndarray) -> float:
    w1, w2 = w1.astype(np.float32).flatten(), w2.astype(np.float32).flatten()
    w1_m, w2_m = w1 - np.mean(w1), w2 - np.mean(w2)
    denom = np.sqrt(np.sum(w1_m**2) * np.sum(w2_m**2))
    if denom == 0:
        return 0.0
    return float(np.sum(w1_m * w2_m) / denom)


def calculate_ber(w_orig: np.ndarray, w_extr: np.ndarray) -> float:
    w1 = (w_orig > 0.5).astype(np.int32)
    w2 = (w_extr > 0.5).astype(np.int32)
    return float(np.sum(w1 != w2) / w1.size)


def test_catalan_reversibility():
    ct = CatalanTransform()
    data = np.arange(32 * 32, dtype=np.uint8).reshape((32, 32))
    transformed = ct.catalan_transform(data, iterations=5, key=7)
    reversed_data = ct.inverse_catalan_transform(transformed, iterations=5, key=7)
    assert np.array_equal(data, reversed_data)


def test_scrambler_reversibility():
    scrambler = WatermarkScrambler(default_size=(32, 32))
    rng = np.random.RandomState(42)
    data = rng.randint(0, 2, (32, 32), dtype=np.uint8)
    scrambled = scrambler.arnold_cat_map(data, iterations=10)
    descrambled = scrambler.inverse_arnold_cat_map(scrambled, iterations=10)
    assert np.array_equal(data, descrambled)


def test_mosaic_generation():
    mg = MosaicGenerator()
    tile = np.ones((32, 32), dtype=np.uint8)
    mosaic = mg.create_tiled_mosaic(tile, target_shape=(256, 256))
    assert mosaic.shape == (256, 256)
    assert np.array_equal(mosaic[:32, :32], tile)
    assert np.array_equal(mosaic[224:256, 224:256], tile)


def test_adaptive_embedder_fidelity():
    embedder = AdaptiveEmbedder(alpha_base=0.012, sensitivity=2.0)
    
    # Test on real preprocessed host if available, else standard test image
    host_files = sorted(list(Path('preprocessed/I_channel').glob('*.npy')))
    if host_files:
        host = np.load(host_files[0])
    else:
        # Smooth gradient natural-like surface
        y, x = np.mgrid[0:256, 0:256] / 256.0
        host = (0.5 + 0.3 * np.sin(x * 3.14) * np.cos(y * 3.14)).astype(np.float32)
        
    wm = np.tile(np.random.RandomState(42).randint(0, 2, (32, 32)).astype(np.float32), (8, 8))
    
    embedded = embedder.embed(host, wm)
    assert embedded.shape == (256, 256)
    assert embedded.min() >= 0.0 and embedded.max() <= 1.0
    
    score_psnr = psnr(host, embedded, data_range=1.0)
    score_ssim = ssim(host, embedded, data_range=1.0)
    assert score_psnr >= 40.0
    assert score_ssim >= 0.98



def test_cropping_attack_and_mask_reproducibility():
    cropper = CroppingAttack(target_size=256)
    img = np.ones((256, 256), dtype=np.float32)
    
    atk1 = cropper.apply_attack(img, mode='random', intensity=0.25, seed=42)
    atk2 = cropper.apply_attack(img, mode='random', intensity=0.25, seed=42)
    assert np.array_equal(atk1, atk2)
    
    mask = cropper.get_mask(mode='random', intensity=0.25, seed=42)
    assert np.sum(mask == 0.0) == 128 * 128
    assert np.all(atk1[mask == 0.0] == 0.0)


def test_signal_attack_reproducibility():
    signaller = SignalAttack()
    img = np.full((256, 256), 0.5, dtype=np.float32)
    
    noise1 = signaller.apply_gaussian_noise(img, sigma=0.05, seed=123)
    noise2 = signaller.apply_gaussian_noise(img, sigma=0.05, seed=123)
    assert np.array_equal(noise1, noise2)
    
    jpg = signaller.apply_jpeg(img, quality=70)
    assert jpg.shape == (256, 256)
    assert jpg.min() >= 0.0 and jpg.max() <= 1.0


def test_collusion_attack_reproducibility():
    colluder = CollusionAttack()
    img1 = np.full((256, 256), 0.4, dtype=np.float32)
    img2 = np.full((256, 256), 0.6, dtype=np.float32)
    
    res1 = colluder.simulate_collusion([img1, img2], noise_std=0.01, seed=99)
    res2 = colluder.simulate_collusion([img1, img2], noise_std=0.01, seed=99)
    assert np.array_equal(res1, res2)


def test_nc_and_ber_metrics():
    w = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert calculate_nc(w, w) == pytest.approx(1.0, abs=1e-5)
    assert calculate_ber(w, w) == 0.0
    
    w_inv = 1.0 - w
    assert calculate_ber(w, w_inv) == 1.0

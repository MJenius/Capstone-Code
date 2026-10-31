"""
Keyed permutation utilities.

The Arnold Cat Map (ACM) followed by the Catalan permutation, as implemented in
``utils/scrambler.py`` and ``utils/catalan.py``, only *moves* pixels: the composed
transform is a fixed bijection of the 1024 positions of a 32x32 watermark that is
fully determined by the secret parameters (ACM iterations, Catalan iterations,
Catalan key).

This module computes that bijection once as an index map so that it can be used
(a) for fast vectorised forward / inverse transforms and (b) as a fixed,
non-trainable layer inside the blind ANN extractor.  Nothing here uses the host
image; only the secret key parameters are needed.
"""
from functools import lru_cache
import logging

import numpy as np

from utils.catalan import CatalanTransform
from utils.scrambler import WatermarkScrambler


@lru_cache(maxsize=16)
def _forward_index_cached(size: int, acm_iterations: int, catalan_iterations: int, catalan_key: int) -> bytes:
    prev = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    try:
        idx = np.arange(size * size, dtype=np.float64).reshape(size, size)
        scr = WatermarkScrambler(default_size=(size, size)).arnold_cat_map(idx, acm_iterations)
        out = CatalanTransform().catalan_transform(scr, catalan_iterations, catalan_key)
    finally:
        logging.getLogger().setLevel(prev)
    fwd = out.ravel().astype(np.int64)
    if not np.array_equal(np.sort(fwd), np.arange(size * size)):
        raise RuntimeError("Arnold+Catalan composition is not a permutation for these parameters")
    return fwd.tobytes()


def forward_index(size: int = 32, acm_iterations: int = 10, catalan_iterations: int = 5,
                  catalan_key: int = 7) -> np.ndarray:
    """Index map ``fwd`` such that ``transformed.ravel() == watermark.ravel()[fwd]``."""
    return np.frombuffer(
        _forward_index_cached(size, acm_iterations, catalan_iterations, catalan_key), dtype=np.int64
    ).copy()


def inverse_index(size: int = 32, acm_iterations: int = 10, catalan_iterations: int = 5,
                  catalan_key: int = 7) -> np.ndarray:
    """Index map ``inv`` such that ``watermark.ravel() == transformed.ravel()[inv]``."""
    return np.argsort(forward_index(size, acm_iterations, catalan_iterations, catalan_key))


def apply_forward(wm: np.ndarray, **key) -> np.ndarray:
    s = wm.shape[0]
    return wm.ravel()[forward_index(s, **key)].reshape(wm.shape)


def apply_inverse(t: np.ndarray, **key) -> np.ndarray:
    s = t.shape[0]
    return t.ravel()[inverse_index(s, **key)].reshape(t.shape)

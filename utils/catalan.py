"""
Catalan transform module for additional watermark permutation.

This transform applies a deterministic, reversible permutation to a square
watermark array using Catalan-number-derived ordering.
"""
import logging
from typing import Optional

import numpy as np


class CatalanTransform:
    """
    Applies a reversible Catalan-number-based permutation to 2D watermark arrays.
    """

    def _catalan_sequence(self, length: int, offset: int) -> list:
        """
        Generate a Catalan sequence of a target length.

        Args:
            length: Number of values to generate
            offset: Start offset to vary per iteration/key

        Returns:
            List of Python integers (unbounded precision)
        """
        # Generate enough Catalan values then slice from offset.
        needed = length + max(0, offset)
        values = [1]

        for n in range(needed - 1):
            next_val = values[-1] * 2 * (2 * n + 1) // (n + 2)
            values.append(next_val)

        return values[offset:offset + length]

    def _build_permutation(self, length: int, key: int, iteration: int) -> np.ndarray:
        """
        Build a deterministic permutation using Catalan-derived sort keys.

        Args:
            length: Flattened watermark length
            key: User key component
            iteration: Current iteration index

        Returns:
            Permutation indices where output = input[perm]
        """
        offset = max(0, key + iteration)
        catalan = self._catalan_sequence(length=length, offset=offset)

        # Hash each (catalan_value, index, key) tuple to a 64-bit integer.
        # This avoids the modulo-collapse that causes many equal sort keys when
        # Catalan numbers grow large, which previously made the sort degenerate
        # into a near-identity permutation for large arrays.
        import hashlib
        def _sort_key(i: int) -> int:
            raw = f"{catalan[i]}:{i}:{key}:{iteration}".encode()
            return int(hashlib.blake2b(raw, digest_size=8).hexdigest(), 16)

        sortable = [(_sort_key(i), i) for i in range(length)]
        ordered = sorted(sortable, key=lambda item: item[0])
        perm = np.array([idx for _, idx in ordered], dtype=np.int64)

        return perm

    def catalan_transform(
        self,
        image: np.ndarray,
        iterations: int,
        key: int = 0
    ) -> Optional[np.ndarray]:
        """
        Apply forward Catalan permutation transform.

        Args:
            image: Input square 2D watermark
            iterations: Number of iterations
            key: Integer key that shifts permutation pattern

        Returns:
            Transformed image, or None on error
        """
        try:
            if image is None or image.size == 0:
                logging.error("Invalid input image: empty or None")
                return None

            if len(image.shape) != 2:
                logging.error(f"Image must be 2D, got shape: {image.shape}")
                return None

            h, w = image.shape
            if h != w:
                logging.error(f"Image must be square, got dimensions: {h}x{w}")
                return None

            if iterations < 0:
                logging.error("Iterations must be >= 0")
                return None

            transformed = image.copy()
            flat_len = transformed.size

            for iteration in range(iterations):
                perm = self._build_permutation(flat_len, key=key, iteration=iteration)
                transformed = transformed.flatten()[perm].reshape(h, w)

            logging.info(
                "Successfully applied Catalan transform: %dx%d, iterations=%d, key=%d",
                h,
                w,
                iterations,
                key,
            )
            return transformed

        except Exception as exc:
            logging.error(f"Error in Catalan transform: {str(exc)}")
            return None

    def inverse_catalan_transform(
        self,
        transformed_image: np.ndarray,
        iterations: int,
        key: int = 0
    ) -> Optional[np.ndarray]:
        """
        Reverse the Catalan permutation transform.

        Args:
            transformed_image: Transformed square 2D watermark
            iterations: Number of forward iterations originally used
            key: Integer key used in the forward transform

        Returns:
            Recovered image, or None on error
        """
        try:
            if transformed_image is None or transformed_image.size == 0:
                logging.error("Invalid input image: empty or None")
                return None

            if len(transformed_image.shape) != 2:
                logging.error(f"Image must be 2D, got shape: {transformed_image.shape}")
                return None

            h, w = transformed_image.shape
            if h != w:
                logging.error(f"Image must be square, got dimensions: {h}x{w}")
                return None

            if iterations < 0:
                logging.error("Iterations must be >= 0")
                return None

            recovered = transformed_image.copy()
            flat_len = recovered.size

            for iteration in range(iterations - 1, -1, -1):
                perm = self._build_permutation(flat_len, key=key, iteration=iteration)
                forward_flat = recovered.flatten()
                inverse_flat = np.empty_like(forward_flat)
                inverse_flat[perm] = forward_flat
                recovered = inverse_flat.reshape(h, w)

            logging.info(
                "Successfully reversed Catalan transform: %dx%d, iterations=%d, key=%d",
                h,
                w,
                iterations,
                key,
            )
            return recovered

        except Exception as exc:
            logging.error(f"Error in inverse Catalan transform: {str(exc)}")
            return None

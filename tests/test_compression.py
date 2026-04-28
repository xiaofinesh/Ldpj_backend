"""Tests for storage.compression (v2.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from storage.compression import (
    compress_float_array,
    decompress_float_array,
    estimate_compression_ratio,
)


class TestCompression:
    def test_round_trip_small(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        blob = compress_float_array(values)
        assert blob is not None
        recovered = decompress_float_array(blob)
        assert recovered == pytest.approx(values, rel=1e-6)

    def test_round_trip_70_points(self):
        # Realistic 70-point pressure curve: smooth ramp + slight noise
        rng = np.random.default_rng(42)
        base = np.linspace(0, 600, 70)
        noise = rng.normal(0, 0.5, 70)
        values = (base + noise).tolist()
        blob = compress_float_array(values)
        recovered = decompress_float_array(blob)
        assert recovered == pytest.approx(values, rel=1e-4)

    def test_compression_ratio_smooth_curve(self):
        """Slow-drift curves (typical hold section) compress noticeably."""
        # 70 samples, gentle linear drift — close to a real hold section
        values = [600.0 - i * 0.1 for i in range(70)]
        ratio = estimate_compression_ratio(values)
        assert ratio > 1.5  # at least 1.5× on smooth data

    def test_compression_ratio_noisy_does_not_explode(self):
        """High-entropy noise compresses poorly (float32 noise has high entropy);
        zlib should at worst add a small constant overhead, not double the size."""
        rng = np.random.default_rng(42)
        values = (np.linspace(0, 600, 70) + rng.normal(0, 0.5, 70)).tolist()
        ratio = estimate_compression_ratio(values)
        assert ratio > 0.8  # < 25% overhead even in the worst case

    def test_constant_curve_compresses_well(self):
        """A flat curve should compress very aggressively."""
        values = [600.0] * 70
        ratio = estimate_compression_ratio(values)
        assert ratio > 5.0

    def test_empty_input_returns_none(self):
        assert compress_float_array([]) is None
        assert compress_float_array(None) is None

    def test_decompress_none_or_empty(self):
        assert decompress_float_array(None) is None
        assert decompress_float_array(b"") is None

    def test_estimate_ratio_empty(self):
        assert estimate_compression_ratio([]) == 0.0

    def test_blob_is_bytes(self):
        blob = compress_float_array([1.0, 2.0, 3.0])
        assert isinstance(blob, bytes)

    def test_float32_precision(self):
        """Round-trip preserves float32 precision (not float64)."""
        values = [1.234567890123, 2.345678901234]
        blob = compress_float_array(values)
        recovered = decompress_float_array(blob)
        # float32 has ~7 decimal digits of precision
        assert recovered == pytest.approx(values, abs=1e-5)


class TestBlobVsJsonStorage:
    """The headline disk saving for v2.6 isn't zlib — it's switching the
    storage format from JSON text to a binary BLOB. This test pins down
    the magnitude so future regressions on the format are caught."""

    def test_blob_smaller_than_json_for_70_pressures(self):
        import json

        rng = np.random.default_rng(42)
        # Realistic 70-point pressure cycle
        values = (np.linspace(0, 600, 70) + rng.normal(0, 0.5, 70)).tolist()

        json_size = len(json.dumps(values).encode("utf-8"))
        blob = compress_float_array(values)
        blob_size = len(blob)

        # JSON of 70 floats is typically 800–1200 bytes; BLOB is 280–320.
        assert blob_size < json_size
        # Expect at least 2× savings vs JSON for typical pressure data
        assert json_size / blob_size > 2.0

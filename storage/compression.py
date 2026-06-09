"""Compression utilities for v2.6 storage.

Strategy:
- Encode List[float] as numpy float32 array → bytes (4 bytes/sample)
- zlib compress at level 6 (default, balanced speed/ratio)
- Store as BLOB in SQLite

Empirical compression ratios (compressed_bytes / raw_float32_bytes):
- Constant curves:      5–30×   (zlib finds repeating bytes)
- Smooth slow drift:    1.1–2×  (limited mantissa overlap)
- Noisy real curves:    ~1.0×   (high entropy, zlib breaks even)

The headline disk saving vs v2.5 comes from switching from JSON text
("[600.123, 601.456, ...]" ≈ 10 bytes/sample) to binary BLOB
(4 bytes/sample), independent of zlib — roughly 2.5× by itself.
zlib on top is icing for the smoother sections.

Decompression overhead: < 1 ms per 70-point curve.
"""

from __future__ import annotations

import logging
import zlib
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# zlib level: 1 (fastest) to 9 (best). For float32 with high-entropy
# mantissa (the realistic case for noisy pressure curves), levels 1 and 6
# achieve nearly identical compression ratios — but level 1 is roughly
# 2–3× faster on the encode side. We're on a write-heavy hot path
# (one log_record per cycle per cabin), so favor speed.
COMPRESSION_LEVEL = 1


def compress_float_array(values: Optional[List[float]]) -> Optional[bytes]:
    """Compress a list of floats into a zlib BLOB.

    Returns None if input is None or empty (so callers can use the result
    directly as a SQLite BLOB parameter).
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return zlib.compress(arr.tobytes(), COMPRESSION_LEVEL)


def decompress_float_array(blob: Optional[bytes]) -> Optional[List[float]]:
    """Decompress a BLOB into a list of floats.

    Returns None if blob is None or empty.
    """
    if not blob:
        return None
    # A corrupt/truncated BLOB must not abort a whole CSV export mid-file or
    # 500 the /records/{id} endpoint — degrade to None so callers skip the row.
    try:
        raw = zlib.decompress(blob)
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.tolist()
    except Exception as exc:
        logger.warning("decompress_float_array: corrupt BLOB (%d bytes): %s",
                       len(blob), exc)
        return None


def estimate_compression_ratio(values: List[float]) -> float:
    """Diagnostic: report raw_bytes / compressed_bytes for a list.

    Returns 0.0 for empty input.
    """
    if not values:
        return 0.0
    raw_size = len(values) * 4  # float32
    compressed = compress_float_array(values)
    if not compressed:
        return 0.0
    return raw_size / len(compressed)

"""Quantization utilities for index migration.

Provides idempotent dtype detection for reliable vector re-encoding.
"""

from typing import Dict, Optional

from redisvl.migration.models import DTYPE_BYTES

# Dtypes that share byte widths and are functionally interchangeable
# for idempotent detection purposes (same byte length per element).
_DTYPE_FAMILY: Dict[str, str] = {
    "float64": "8byte",
    "float32": "4byte",
    "float16": "2byte",
    "bfloat16": "2byte",
    "int8": "1byte",
    "uint8": "1byte",
}


def is_same_width_dtype_conversion(source_dtype: str, target_dtype: str) -> bool:
    """Return True when two dtypes share byte width but differ in encoding."""
    if source_dtype == target_dtype:
        return False
    source_family = _DTYPE_FAMILY.get(source_dtype)
    target_family = _DTYPE_FAMILY.get(target_dtype)
    if source_family is None or target_family is None:
        return False
    return source_family == target_family


# ---------------------------------------------------------------------------
# Idempotent Dtype Detection
# ---------------------------------------------------------------------------


def detect_vector_dtype(data: bytes, expected_dims: int) -> Optional[str]:
    """Inspect raw vector bytes and infer the storage dtype.

    Uses byte length and expected dimensions to determine which dtype
    the vector is currently stored as. Returns the canonical representative
    for each byte-width family (float16 for 2-byte, int8 for 1-byte),
    since dtypes within a family cannot be distinguished by length alone.

    Args:
        data: Raw vector bytes from Redis.
        expected_dims: Number of dimensions expected for this vector field.

    Returns:
        Detected dtype string (e.g. "float32", "float16", "int8") or None
        if the size does not match any known dtype.
    """
    if not data or expected_dims <= 0:
        return None

    nbytes = len(data)

    # Check each dtype in decreasing element size to avoid ambiguity.
    # Only canonical representatives are checked (float16 covers bfloat16,
    # int8 covers uint8) since they share byte widths.
    for dtype in ("float64", "float32", "float16", "int8"):
        if nbytes == expected_dims * DTYPE_BYTES[dtype]:
            return dtype

    return None


def is_already_quantized(
    data: bytes,
    expected_dims: int,
    source_dtype: str,
    target_dtype: str,
) -> bool:
    """Check whether a vector has already been converted to the target dtype.

    Uses byte-width families to handle ambiguous dtypes. For example,
    if source is float32 and target is float16, a vector detected as
    2-bytes-per-element is considered already quantized (the byte width
    shrank from 4 to 2, so conversion already happened).

    However, same-width conversions (e.g. float16 -> bfloat16 or
    int8 -> uint8) are NOT skipped because the encoding semantics
    differ even though the byte length is identical. We cannot
    distinguish these by length, so we must always re-encode.

    Args:
        data: Raw vector bytes.
        expected_dims: Number of dimensions.
        source_dtype: The dtype the vector was originally stored as.
        target_dtype: The dtype we want to convert to.

    Returns:
        True if the vector already matches the target dtype (skip conversion).
    """
    detected = detect_vector_dtype(data, expected_dims)
    if detected is None:
        return False

    detected_family = _DTYPE_FAMILY.get(detected)
    target_family = _DTYPE_FAMILY.get(target_dtype)
    source_family = _DTYPE_FAMILY.get(source_dtype)

    # If detected byte-width matches target family, the vector looks converted.
    # But if source and target share the same byte-width family (e.g.
    # float16 -> bfloat16), we cannot tell whether conversion happened,
    # so we must NOT skip -- always re-encode for same-width migrations.
    if source_family == target_family:
        return False

    return detected_family == target_family

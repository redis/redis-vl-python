"""
RedisVL Semantic Cache Schema (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.cache.llm.schema` instead.
"""

import warnings

from redisvl.extensions.cache.llm.schema import (
    CacheEntry,
    CacheHit,
    SemanticCacheIndexSchema,
)

warnings.warn(
    "Importing from redisvl.extensions.llmcache.schema is deprecated. "
    "Please import from redisvl.extensions.cache.llm.schema instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CacheEntry", "CacheHit", "SemanticCacheIndexSchema"]

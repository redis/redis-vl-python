"""
RedisVL LLM Cache Extensions (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.cache` instead.
"""

import warnings

from redisvl.extensions.cache.llm.base import BaseLLMCache
from redisvl.extensions.cache.llm.schema import (
    CacheEntry,
    CacheHit,
    SemanticCacheIndexSchema,
)
from redisvl.extensions.cache.llm.semantic import SemanticCache

warnings.warn(
    "Importing from redisvl.extensions.llmcache is deprecated. "
    "Please import from redisvl.extensions.cache.llm instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BaseLLMCache",
    "SemanticCache",
    "CacheEntry",
    "CacheHit",
    "SemanticCacheIndexSchema",
]

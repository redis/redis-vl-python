"""
RedisVL Semantic Cache (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.cache.llm.semantic` instead.
"""

import warnings

from redisvl.extensions.cache.llm.semantic import SemanticCache

warnings.warn(
    "Importing from redisvl.extensions.llmcache.semantic is deprecated. "
    "Please import from redisvl.extensions.cache.llm.semantic instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SemanticCache"]

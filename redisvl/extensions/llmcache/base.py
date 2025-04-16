"""
RedisVL Base LLM Cache (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.cache.llm.base` instead.
"""

import warnings

from redisvl.extensions.cache.llm.base import BaseLLMCache

warnings.warn(
    "Importing from redisvl.extensions.llmcache.base is deprecated. "
    "Please import from redisvl.extensions.cache.llm.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseLLMCache"]

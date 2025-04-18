"""
Redis Vector Library Cache Extensions

This module provides caching functionality for Redis Vector Library,
including both embedding caches and LLM response caches.
"""

from redisvl.extensions.cache.base import BaseCache

__all__ = ["BaseCache"]

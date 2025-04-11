"""
Redis Vector Library - Embeddings Cache Extensions

This module provides embedding caching functionality for RedisVL.
"""

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry

__all__ = ["EmbeddingsCache", "CacheEntry"]

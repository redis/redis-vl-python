from typing import Any, Dict, List, Optional

from redisvl.redis.utils import hashify


class BaseLLMCache:
    def __init__(self, ttl: Optional[int] = None):
        self._ttl: Optional[int] = None
        self.set_ttl(ttl)

    @property
    def ttl(self) -> Optional[int]:
        """The default TTL, in seconds, for entries in the cache."""
        return self._ttl

    def set_ttl(self, ttl: Optional[int] = None):
        """Set the default TTL, in seconds, for entries in the cache.

        Args:
            ttl (Optional[int], optional): The optional time-to-live expiration
                for the cache, in seconds.

        Raises:
            ValueError: If the time-to-live value is not an integer.
        """
        if ttl:
            if not isinstance(ttl, int):
                raise ValueError(f"TTL must be an integer value, got {ttl}")
            self._ttl = int(ttl)
        else:
            self._ttl = None

    def clear(self) -> None:
        """Clear the LLMCache of all keys in the index."""
        raise NotImplementedError

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
    ) -> List[dict]:
        raise NotImplementedError

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = {},
    ) -> str:
        """Stores the specified key-value pair in the cache along with
        metadata."""
        raise NotImplementedError

    def hash_input(self, prompt: str) -> str:
        """Hashes the input prompt using SHA256.

        Args:
            prompt (str): Input string to be hashed.

        Returns:
            str: Hashed string.
        """
        return hashify(prompt)

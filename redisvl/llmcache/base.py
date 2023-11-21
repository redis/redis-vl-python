import hashlib
from typing import List, Optional

from redisvl.index import SearchIndex


class BaseLLMCache:
    _ttl: Optional[int] = None

    @property
    def ttl(self) -> Optional[int]:
        """Returns the TTL for the cache.

        Returns:
            Optional[int]: The TTL for the cache.
        """
        return self._ttl

    def set_ttl(self, ttl: Optional[int] = None):
        """Sets the TTL for the cache.

        Args:
            ttl (Optional[int], optional): The optional time-to-live expiration
                for the cache.

        Raises:
            ValueError: If the time-to-live value is not an integer.
        """
        if ttl:
            if not isinstance(ttl, int):
                raise ValueError(f"TTL must be an integer value, got {ttl}")
            self._ttl = int(ttl)

    def clear(self) -> None:
        """Clear the LLMCache of all keys in the index."""
        raise NotImplementedError

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        raise NotImplementedError

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = {},
    ) -> None:
        """Stores the specified key-value pair in the cache along
        with metadata."""
        raise NotImplementedError

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

import hashlib
from typing import Callable, List, Optional


class BaseLLMCache:
    verbose: bool = True

    def clear(self):
        """Clear the LLMCache and create a new underlying index."""
        raise NotImplementedError

    def check(self, prompt: str) -> Optional[List[str]]:
        raise NotImplementedError

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[dict] = {},
        key: Optional[str] = None,
    ) -> None:
        """Stores the specified key-value pair in the cache along with metadata."""
        raise NotImplementedError

    def _refresh_ttl(self, key: str):
        """Refreshes the TTL for the specified key."""
        raise NotImplementedError

    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

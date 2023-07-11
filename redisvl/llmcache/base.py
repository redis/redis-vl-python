import hashlib
from typing import Callable, Optional, List


class BaseLLMCache:

    verbose: bool = True

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

    def cache_response(self, llm_callable: Callable):
        """Decorator method for wrapping custom callables"""

        def wrapper(*args, **kwargs):
            # Check LLM Cache first
            key = self.hash_input(*args, **kwargs)
            response = self.check(*args, **kwargs)
            if response:
                self._refresh_ttl(key)
                return response
            # Otherwise execute the llm callable here
            response = llm_callable(*args, **kwargs)
            return response

        return wrapper

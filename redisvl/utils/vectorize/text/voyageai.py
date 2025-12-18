from typing import Any, List, Optional

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.voyageai import VoyageAIVectorizer


@deprecated_class(
    name="VoyageAITextVectorizer", replacement="Use VoyageAIVectorizer instead."
)
class VoyageAITextVectorizer(VoyageAIVectorizer):
    """A backwards-compatible alias for VoyageAIVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: Any = "", text: Any = "", **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the VoyageAI API.

        Deprecated: Use `VoyageAIVectorizer.embed` instead.
        """
        content = content or text
        return super().embed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the VoyageAI API.

        Deprecated: Use `VoyageAIVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)

    @deprecated_argument("text", "content")
    async def aembed(self, content: Any = "", text: Any = "", **kwargs) -> List[float]:
        """Asynchronously generate a vector embedding for a single text using the VoyageAI API.

        Deprecated: Use `VoyageAIVectorizer.aembed` instead.
        """
        content = content or text
        return await super().aembed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    async def aembed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings for a batch of texts using the VoyageAI API.

        Deprecated: Use `VoyageAIVectorizer.aembed_many` instead.
        """
        contents = contents or texts
        return await super().aembed_many(contents=contents, **kwargs)

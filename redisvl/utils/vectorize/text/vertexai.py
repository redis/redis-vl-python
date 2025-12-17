from typing import Any, List, Optional

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.vertexai import VertexAIVectorizer


@deprecated_class(
    name="VertexAITextVectorizer", replacement="Use VertexAIVectorizer instead."
)
class VertexAITextVectorizer(VertexAIVectorizer):
    """A backwards-compatible alias for VertexAIVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: str = "", text: Any = "", **kwargs) -> List[float]:
        """Generate a vector embedding for a single input using the VertexAI API.

        Deprecated: Use `VertexAIVectorizer.embed` instead.
        """
        content = content or text
        return super().embed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: Optional[List[str]] = None,
        texts: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of inputs using the VertexAI API.

        Deprecated: Use `VertexAIVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)

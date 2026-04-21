from typing import Any

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.vertexai import VertexAIVectorizer


@deprecated_class(
    name="VertexAITextVectorizer", replacement="Use VertexAIVectorizer instead."
)
class VertexAITextVectorizer(VertexAIVectorizer):
    """A backwards-compatible alias for VertexAIVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: str = "", text: Any = "", **kwargs) -> list[float]:
        """Generate a vector embedding for a single input using the VertexAI API.

        Deprecated: Use `VertexAIVectorizer.embed` instead.
        """
        content = content or text
        return super().embed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: list[str] | None = None,
        texts: list[Any] | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Generate vector embeddings for a batch of inputs using the VertexAI API.

        Deprecated: Use `VertexAIVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)

from typing import Any

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.bedrock import BedrockVectorizer


@deprecated_class(
    name="BedrockTextVectorizer", replacement="Use BedrockVectorizer instead."
)
class BedrockTextVectorizer(BedrockVectorizer):
    """A backwards-compatible alias for BedrockVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: Any = "", text: Any = "", **kwargs) -> list[float] | bytes:
        """Generate a vector embedding for a single input using the AWS Bedrock API.

        Deprecated: Use `BedrockVectorizer.embed` instead.
        """
        content = content or text
        return super().embed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: list[Any] | None = None,
        texts: list[Any] | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Generate vector embeddings for a batch of inputs using the AWS Bedrock API.

        Deprecated: Use `BedrockVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)

from typing import Any

from redisvl.utils.utils import deprecated_argument, deprecated_class
from redisvl.utils.vectorize.custom import CustomVectorizer


@deprecated_class(
    name="CustomTextVectorizer", replacement="Use CustomVectorizer instead."
)
class CustomTextVectorizer(CustomVectorizer):
    """A backwards-compatible alias for CustomVectorizer."""

    @deprecated_argument("text", "content")
    def embed(self, content: Any = "", text: Any = "", **kwargs) -> list[float]:
        """Generate a vector embedding for a single input using the custom function.

        Deprecated: Use `CustomVectorizer.embed` instead.
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
        """Generate vector embeddings for a batch of inputs using the custom function.

        Deprecated: Use `CustomVectorizer.embed_many` instead.
        """
        contents = contents or texts
        return super().embed_many(contents=contents, **kwargs)

    @deprecated_argument("text", "content")
    async def aembed(self, content: Any = "", text: Any = "", **kwargs) -> list[float]:
        """Asynchronously generate a vector embedding for a single input using the custom function.

        Deprecated: Use `CustomVectorizer.aembed` instead.
        """
        content = content or text
        return await super().aembed(content=content, **kwargs)

    @deprecated_argument("texts", "contents")
    async def aembed_many(
        self,
        contents: list[Any] | None = None,
        texts: list[Any] | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Asynchronously generate vector embeddings for a batch of inputs using the custom function.

        Deprecated: Use `CustomVectorizer.aembed_many` instead.
        """
        contents = contents or texts
        return await super().aembed_many(contents=contents, **kwargs)

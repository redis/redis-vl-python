from redisvl.utils.utils import deprecated_class
from redisvl.utils.vectorize.vertexai import VertexAIVectorizer


@deprecated_class(
    name="VertexAITextVectorizer",
    replacement="Use GoogleGenAIVectorizer instead.",
)
class VertexAITextVectorizer(VertexAIVectorizer):
    """A backwards-compatible alias for VertexAIVectorizer, which is itself
    deprecated: use GoogleGenAIVectorizer instead.

    The `text`/`texts` keyword arguments still work, and still warn, via
    BaseVectorizer.
    """

from redisvl.utils.utils import deprecated_class
from redisvl.utils.vectorize.bedrock import BedrockVectorizer


@deprecated_class(
    name="BedrockTextVectorizer", replacement="Use BedrockVectorizer instead."
)
class BedrockTextVectorizer(BedrockVectorizer):
    """A backwards-compatible alias for BedrockVectorizer.

    The `text`/`texts` keyword arguments still work, and still warn, via
    BaseVectorizer.
    """

from redisvl.utils.utils import deprecated_class
from redisvl.utils.vectorize.voyageai import VoyageAIVectorizer


@deprecated_class(
    name="VoyageAITextVectorizer", replacement="Use VoyageAIVectorizer instead."
)
class VoyageAITextVectorizer(VoyageAIVectorizer):
    """A backwards-compatible alias for VoyageAIVectorizer.

    The `text`/`texts` keyword arguments still work, and still warn, via
    BaseVectorizer.
    """

from redisvl.utils.utils import deprecated_class
from redisvl.utils.vectorize.custom import CustomVectorizer


@deprecated_class(
    name="CustomTextVectorizer", replacement="Use CustomVectorizer instead."
)
class CustomTextVectorizer(CustomVectorizer):
    """A backwards-compatible alias for CustomVectorizer.

    The `text`/`texts` keyword arguments still work, and still warn, via
    BaseVectorizer.
    """

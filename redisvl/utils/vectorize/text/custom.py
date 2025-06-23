from typing import TYPE_CHECKING, Callable, List, Optional

from pydantic import ConfigDict

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.vectorize.base import BaseVectorizer


def _check_vector(result: list, method_name: str) -> None:
    """
    Validates the structure of returned embeddings.

    - For methods named "*_many", expects a list of lists of floats.
    - For single methods, expects a list of floats.

    Raises:
        ValueError: If the embeddings do not match the expected structure.
    """
    if method_name.endswith("_many"):
        # embed_many / aembed_many → list of lists
        if not isinstance(result, list) or not result:
            raise ValueError(f"{method_name} must return a non-empty list of lists.")
        if not isinstance(result[0], list) or not result[0]:
            raise ValueError(f"{method_name} must return a list of non-empty lists.")
        if not isinstance(result[0][0], float):
            raise ValueError(f"{method_name} must return a list of lists of floats.")
    else:
        # embed / aembed → a single list of floats
        if not isinstance(result, list) or not result:
            raise ValueError(f"{method_name} must return a non-empty list.")
        if not isinstance(result[0], float):
            raise ValueError(f"{method_name} must return a list of floats.")


class CustomTextVectorizer(BaseVectorizer):
    """The CustomTextVectorizer class wraps user-defined embedding methods to create
    embeddings for text data.

    This vectorizer is designed to accept a provided callable text vectorizer and
    provides a class definition to allow for compatibility with RedisVL.
    The vectorizer may support both synchronous and asynchronous operations which
    allows for batch processing of texts, but at a minimum only synchronous embedding
    is required to satisfy the 'embed()' method.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        # Basic usage with a custom embedding function
        vectorizer = CustomTextVectorizer(
            embed = my_vectorizer.generate_embedding
        )
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="my_embeddings_cache")

        vectorizer = CustomTextVectorizer(
            embed=my_vectorizer.generate_embedding,
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        embed: Callable,
        embed_many: Optional[Callable] = None,
        aembed: Optional[Callable] = None,
        aembed_many: Optional[Callable] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
    ):
        """Initialize the Custom vectorizer.

        Args:
            embed (Callable): a Callable function that accepts a string object and returns a list of floats.
            embed_many (Optional[Callable]): a Callable function that accepts a list of string objects and returns a list containing lists of floats. Defaults to None.
            aembed (Optional[Callable]): an asynchronous Callable function that accepts a string object and returns a lists of floats. Defaults to None.
            aembed_many (Optional[Callable]): an asynchronous Callable function that accepts a list of string objects and returns a list containing lists of floats. Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ValueError: if embedding validation fails.
        """
        # First, determine the dimensions
        try:
            test_result = embed("dimension test")
            _check_vector(test_result, "embed")
            dims = len(test_result)
        except Exception as e:
            raise ValueError(f"Failed to validate embed method: {e}")

        # Initialize parent with known information
        super().__init__(model="custom", dtype=dtype, dims=dims, cache=cache)

        # Now setup the functions and validation flags
        self._setup_functions(embed, embed_many, aembed, aembed_many)

    def _setup_functions(self, embed, embed_many, aembed, aembed_many):
        """Setup the user-provided embedding functions."""
        self._embed_func = embed
        self._embed_func_many = embed_many
        self._aembed_func = aembed
        self._aembed_func_many = aembed_many

        # Initialize validation flags
        self._aembed_validated = False
        self._aembed_many_validated = False

        # Validate the other functions if provided
        self._validate_optional_funcs()

    @property
    def type(self) -> str:
        return "custom"

    def _validate_optional_funcs(self) -> None:
        """
        Optionally validate the other user-provided functions if they exist.

        Raises:
            ValueError: If any provided function produces invalid results.
        """
        # Check embed_many if provided
        if self._embed_func_many:
            try:
                test_batch = self._embed_func_many(["dimension test (many)"])
                _check_vector(test_batch, "embed_many")
            except Exception as e:
                raise ValueError(f"Invalid embed_many function: {e}")

    def _embed(self, text: str, **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the provided user function.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the user function

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        try:
            result = self._embed_func(text, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the provided user function.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            **kwargs: Additional parameters to pass to the user function

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._embed_func_many:
            # Fallback: Use _embed for each text if no batch function provided
            return [self._embed(text, **kwargs) for text in texts]

        try:
            results = self._embed_func_many(texts, **kwargs)
            return results
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    async def _aembed(self, text: str, **kwargs) -> List[float]:
        """Asynchronously generate a vector embedding for a single text.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the user async function

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            NotImplementedError: If no aembed function was provided
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if not self._aembed_func:
            return self._embed(text, **kwargs)

        try:
            result = await self._aembed_func(text, **kwargs)

            # Validate result on first call
            if not self._aembed_validated:
                _check_vector(result, "aembed")
                self._aembed_validated = True

            return result
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    async def _aembed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            **kwargs: Additional parameters to pass to the user async function

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            NotImplementedError: If no aembed_many function was provided
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._aembed_func_many:
            return self._embed_many(texts, batch_size, **kwargs)

        try:
            results = await self._aembed_func_many(texts, **kwargs)

            # Validate result on first call
            if not self._aembed_many_validated:
                _check_vector(results, "aembed_many")
                self._aembed_many_validated = True

            return results
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

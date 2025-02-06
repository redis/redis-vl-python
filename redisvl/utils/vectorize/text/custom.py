from typing import Any, Callable, List, Optional

from pydantic.v1 import PrivateAttr

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


def validate_async(method):
    """
    Decorator that lazily validates the output of async methods (aembed, aembed_many).
    On first call, it checks the returned embeddings with _check_vector, then sets a flag
    so subsequent calls skip re-validation.
    """

    async def wrapper(self, *args, **kwargs):
        result = await method(self, *args, **kwargs)
        method_name = method.__name__
        validated_attr = f"_{method_name}_validated"

        try:
            if not getattr(self, validated_attr):
                _check_vector(result, method_name)
                setattr(self, validated_attr, True)
        except Exception as e:
            raise ValueError(f"Invalid embedding method: {e}")

        return result

    return wrapper


class CustomTextVectorizer(BaseVectorizer):
    """
    CustomTextVectorizer handles user-provided embedding callables (sync and async).
    Synchronous methods are validated during initialization to determine dimensions.
    Asynchronous methods are validated lazily on first usage.
    """

    # User-provided callables
    _embed: Callable = PrivateAttr()
    _embed_many: Optional[Callable] = PrivateAttr()
    _aembed: Optional[Callable] = PrivateAttr()
    _aembed_many: Optional[Callable] = PrivateAttr()

    # Validation flags for async methods
    _aembed_validated: bool = PrivateAttr(default=False)
    _aembed_many_validated: bool = PrivateAttr(default=False)

    def __init__(
        self,
        embed: Callable,
        embed_many: Optional[Callable] = None,
        aembed: Optional[Callable] = None,
        aembed_many: Optional[Callable] = None,
        dtype: str = "float32",
    ):
        """
        1. Store the provided functions for synergy or lazy usage.
        2. Manually validate the sync callables to discover the embedding dimension.
        3. Call the base initializer with the discovered dimension and provided dtype.
        4. Async callables remain lazy until first call.
        """
        # Store user-provided callables
        self._embed = embed
        self._embed_many = embed_many
        self._aembed = aembed
        self._aembed_many = aembed_many

        # Manually validate sync methods to discover dimension
        dims = self._validate_sync_callables()

        # Initialize the base class now that we know the dimension
        super().__init__(model=self.type, dims=dims, dtype=dtype)

    @property
    def type(self) -> str:
        return "custom"

    def _validate_sync_callables(self) -> int:
        """
        Validate the sync embed function with a test call and discover the dimension.
        Optionally validate embed_many if provided. Returns the discovered dimension.

        Raises:
            ValueError: If embed or embed_many produce malformed results or fail entirely.
        """
        # Check embed
        try:
            test_single = self._embed("dimension test")
            _check_vector(test_single, "embed")
            dims = len(test_single)
        except Exception as e:
            raise ValueError(f"Invalid embedding method: {e}")

        # Check embed_many
        if self._embed_many:
            try:
                test_batch = self._embed_many(["dimension test (many)"])
                _check_vector(test_batch, "embed_many")
            except Exception as e:
                raise ValueError(f"Invalid embedding method: {e}")

        return dims

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """
        Generate an embedding for a single piece of text using your sync embed function.

        Args:
            text (str): The text to embed.
            preprocess (Optional[Callable]): An optional callable to preprocess the text.
            as_buffer (bool): If True, return the embedding as a byte buffer.

        Returns:
            List[float]: The embedding of the input text.

        Raises:
            TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        try:
            result = self._embed(text, **kwargs)
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

        return self._process_embedding(result, as_buffer, dtype)

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple pieces of text in batches using your sync embed_many function.

        Args:
            texts (List[str]): A list of texts to embed.
            preprocess (Optional[Callable]): Optional preprocessing for each text.
            batch_size (int): Number of texts per batch.
            as_buffer (bool): If True, convert each embedding to a byte buffer.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.

        Raises:
            TypeError: If the input is not a list of strings.
            NotImplementedError: If no embed_many function was provided.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._embed_many:
            raise NotImplementedError("No embed_many function was provided.")

        dtype = kwargs.pop("dtype", self.dtype)
        embeddings: List[List[float]] = []

        try:
            for batch in self.batchify(texts, batch_size, preprocess):
                results = self._embed_many(batch, **kwargs)
                processed = [
                    self._process_embedding(r, as_buffer, dtype) for r in results
                ]
                embeddings.extend(processed)
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

        return embeddings

    @validate_async
    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """
        Asynchronously generate an embedding for a single piece of text.

        Args:
            text (str): The text to embed.
            preprocess (Optional[Callable]): An optional callable to preprocess the text.
            as_buffer (bool): If True, return the embedding as a byte buffer.

        Returns:
            List[float]: The embedding of the input text.

        Raises:
            TypeError: If the input is not a string.
            NotImplementedError: If no aembed function was provided.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if not self._aembed:
            raise NotImplementedError("No aembed function was provided.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        try:
            result = await self._aembed(text, **kwargs)
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

        return self._process_embedding(result, as_buffer, dtype)

    @validate_async
    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """
        Asynchronously generate embeddings for multiple pieces of text in batches.

        Args:
            texts (List[str]): The texts to embed.
            preprocess (Optional[Callable]): Optional preprocessing for each text.
            batch_size (int): Number of texts per batch.
            as_buffer (bool): If True, convert each embedding to a byte buffer.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.

        Raises:
            TypeError: If the input is not a list of strings.
            NotImplementedError: If no aembed_many function was provided.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._aembed_many:
            raise NotImplementedError("No aembed_many function was provided.")

        dtype = kwargs.pop("dtype", self.dtype)
        embeddings: List[List[float]] = []

        try:
            for batch in self.batchify(texts, batch_size, preprocess):
                results = await self._aembed_many(batch, **kwargs)
                processed = [
                    self._process_embedding(r, as_buffer, dtype) for r in results
                ]
                embeddings.extend(processed)
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

        return embeddings

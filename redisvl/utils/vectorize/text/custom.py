from typing import Any, Callable, List, Optional, Union

from pydantic import PrivateAttr

from redisvl.utils.utils import deprecated_argument
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
    """The CustomTextVectorizer class wraps user-defined embedding methods to create
    embeddings for text data.

    This vectorizer is designed to accept a provided callable text vectorizer and
    provides a class definition to allow for compatibility with RedisVL.
    The vectorizer may support both synchronous and asynchronous operations which
    allows for batch processing of texts, but at a minimum only syncronous embedding
    is required to satisfy the 'embed()' method.

    .. code-block:: python

        # Synchronous embedding of a single text
        vectorizer = CustomTextVectorizer(
            embed = my_vectorizer.generate_embedding
        )
        embedding = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

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
        """Initialize the Custom vectorizer.

        Args:
            embed (Callable): a Callable function that accepts a string object and returns a list of floats.
            embed_many (Optional[Callable)]: a Callable function that accepts a list of string objects and returns a list containing lists of floats. Defaults to None.
            aembed (Optional[Callable]): an asyncronous Callable function that accepts a string object and returns a lists of floats. Defaults to None.
            aembed_many (Optional[Callable]):  an asyncronous Callable function that accepts a list of string objects and returns a list containing lists of floats. Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.

        Raises:
            ValueError: if embedding validation fails.
        """
        super().__init__(model=self.type, dtype=dtype)

        # Store user-provided callables
        self._embed = embed
        self._embed_many = embed_many
        self._aembed = aembed
        self._aembed_many = aembed_many

        # Set dims
        self.dims = self._validate_sync_callables()

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

    @deprecated_argument("dtype")
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """
        Generate an embedding for a single piece of text using your sync embed function.

        Args:
            text (str): The text to embed.
            preprocess (Optional[Callable]): An optional callable to preprocess the text.
            as_buffer (bool): If True, return the embedding as a byte buffer.

        Returns:
            Union[List[float], bytes]: The embedding of the input text.

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

    @deprecated_argument("dtype")
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """
        Generate embeddings for multiple pieces of text in batches using your sync embed_many function.

        Args:
            texts (List[str]): A list of texts to embed.
            preprocess (Optional[Callable]): Optional preprocessing for each text.
            batch_size (int): Number of texts per batch.
            as_buffer (bool): If True, convert each embedding to a byte buffer.

        Returns:
            Union[List[List[float]], List[bytes]]: A list of embeddings, where each embedding is a list of floats or bytes.

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
        embeddings: Union[List[List[float]], List[bytes]] = []

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
    @deprecated_argument("dtype")
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
    @deprecated_argument("dtype")
    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """
        Asynchronously generate embeddings for multiple pieces of text in batches.

        Args:
            texts (List[str]): The texts to embed.
            preprocess (Optional[Callable]): Optional preprocessing for each text.
            batch_size (int): Number of texts per batch.
            as_buffer (bool): If True, convert each embedding to a byte buffer.

        Returns:
            Union[List[List[float]], List[bytes]]: A list of embeddings, where each embedding is a list of floats or bytes.

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
        embeddings: Union[List[List[float]], List[bytes]] = []

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

import os
from typing import Any, Callable, Dict, List, Optional

from pydantic.v1 import PrivateAttr

from redisvl.utils.vectorize.base import BaseVectorizer


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

    _embed_func: Callable = PrivateAttr()
    _embed_many_func: Optional[Callable] = PrivateAttr()
    _aembed_func: Optional[Callable] = PrivateAttr()
    _aembed_many_func: Optional[Callable] = PrivateAttr()

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
            ValueError: if any of the provided functions accept or return incorrect types.
            TypeError: if any of the provided functions are not Callable objects.
            ValueError: If an invalid dtype is provided.
        """

        self._validate_embed(embed)
        self._embed_func = embed
        if embed_many:
            self._validate_embed_many(embed_many)
            self._embed_many_func = embed_many

        if aembed:
            self._validate_aembed(aembed)
            self._aembed_func = aembed
        if aembed_many:
            self._validate_aembed_many(aembed_many)
            self._aembed_many_func = aembed_many

        super().__init__(model=self.type, dims=self._set_model_dims(), dtype=dtype)

    def _validate_embed(self, func: Callable):
        """calls the func with dummy input and validates that it returns a vector"""
        try:
            test_str = "this is a test sentence"
            candidate_vector = func(test_str)
            if type(candidate_vector) != list or type(candidate_vector[0]) != float:
                raise ValueError(
                    f"Candidate function for embed() does not have the correct return type. Please provide a function with with return type List[float]"
                )
        except TypeError:
            raise TypeError(f"{func} is not a callable object")

    def _validate_embed_many(self, func: Callable):
        """calls the func with dummy input and validates that it returns a list of vectors"""
        try:
            test_strs = ["first test sentence", "second test sentence"]
            candidate_vectors = func(test_strs)
            if (
                type(candidate_vectors) != list
                or type(candidate_vectors[0]) != list
                or type(candidate_vectors[0][0]) != float
            ):
                raise ValueError(
                    f"Candidate function for embed_many does not have the correct return type. Please provide a function with with return type List[List[float]]"
                )
        except TypeError:
            raise TypeError(f"{func} is not a callable object")

    def _validate_aembed(self, func: Callable):
        """calls the func with dummy input and validates that it returns a vector"""
        import asyncio

        try:
            test_str = "this is a test sentence"
            loop = asyncio.get_event_loop()
            candidate_vector = loop.run_until_complete(func(test_str))
            if type(candidate_vector) != list or type(candidate_vector[0]) != float:
                raise ValueError(
                    f"Candidate function for aembed() does not have the correct return type. Please provide a function with with return type List[float]"
                )
        except TypeError:
            raise TypeError(f"{func} is not a callable object")

    def _validate_aembed_many(self, func: Callable):
        """calls the func with dummy input and validates that it returns a list of vectors"""
        import asyncio

        try:
            test_strs = ["first test sentence", "second test sentence"]
            loop = asyncio.get_event_loop()
            candidate_vectors = loop.run_until_complete(func(test_strs))
            if (
                type(candidate_vectors) != list
                or type(candidate_vectors[0]) != list
                or type(candidate_vectors[0][0]) != float
            ):
                raise ValueError(
                    f"Candidate function for aembed_many does not have the correct return type. Please provide a function with with return type List[List[float]]"
                )
        except TypeError:
            raise TypeError(f"{func} is not a callable object")

    def _set_model_dims(self) -> int:
        try:
            test_string = "dimension test"
            embedding = self._embed_func(test_string)
        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(
                f"Error in checking model dimensions. Attempted to embed '{test_string}'. :{str(e)}"
            )
        return len(embedding)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Embed a chunk of text using the provided function.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = self._embed_func(text, **kwargs)
        return self._process_embedding(result, as_buffer, dtype)

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Embed many chunks of texts using the provided function.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing
                callable to perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
            NotImplementedError: if embed_many was not passed to constructor.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._embed_many_func:
            raise NotImplementedError

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            results = self._embed_many_func(batch, **kwargs)
            embeddings += [
                self._process_embedding(r, as_buffer, dtype) for r in results
            ]
        return embeddings

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Asynchronously embed a chunk of text.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
            NotImplementedError: if aembed was not passed to constructor.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if not self._aembed_func:
            raise NotImplementedError

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = await self._aembed_func(text, **kwargs)
        return self._process_embedding(result, as_buffer, dtype)

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Asynchronously embed many chunks of texts.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
            NotImplementedError: If aembed_many was not passed to constructor.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        if not self._aembed_many_func:
            raise NotImplementedError

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            results = await self._aembed_many_func(batch, **kwargs)
            embeddings += [
                self._process_embedding(r, as_buffer, dtype) for r in results
            ]
        return embeddings

    @property
    def type(self) -> str:
        return "custom"

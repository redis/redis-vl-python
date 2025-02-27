from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from redisvl.redis.utils import array_to_buffer
from redisvl.schema.fields import VectorDataType


class Vectorizers(Enum):
    azure_openai = "azure_openai"
    openai = "openai"
    cohere = "cohere"
    mistral = "mistral"
    vertexai = "vertexai"
    hf = "hf"
    voyageai = "voyageai"


class BaseVectorizer(BaseModel, ABC):
    """Base vectorizer interface."""

    model: str
    dtype: str = "float32"
    dims: Optional[int] = None

    @property
    def type(self) -> str:
        return "base"

    @field_validator("dtype")
    @classmethod
    def check_dtype(cls, dtype):
        try:
            VectorDataType(dtype.upper())
        except ValueError:
            raise ValueError(
                f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
            )
        return dtype

    @field_validator("dims")
    @classmethod
    def check_dims(cls, value):
        """Ensures the dims are a positive integer."""
        if value <= 0:
            raise ValueError("Dims must be a positive integer.")
        return value

    @abstractmethod
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Embed a chunk of text.

        Args:
            text: Text to embed
            preprocess: Optional function to preprocess text
            as_buffer: If True, returns a bytes object instead of a list

        Returns:
            Union[List[float], bytes]: Embedding as a list of floats, or as a bytes
            object if as_buffer=True
        """
        raise NotImplementedError

    @abstractmethod
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Embed multiple chunks of text.

        Args:
            texts: List of texts to embed
            preprocess: Optional function to preprocess text
            batch_size: Number of texts to process in each batch
            as_buffer: If True, returns each embedding as a bytes object

        Returns:
            Union[List[List[float]], List[bytes]]: List of embeddings as lists of floats,
            or as bytes objects if as_buffer=True
        """
        raise NotImplementedError

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Asynchronously embed multiple chunks of text.

        Args:
            texts: List of texts to embed
            preprocess: Optional function to preprocess text
            batch_size: Number of texts to process in each batch
            as_buffer: If True, returns each embedding as a bytes object

        Returns:
            Union[List[List[float]], List[bytes]]: List of embeddings as lists of floats,
            or as bytes objects if as_buffer=True
        """
        # Fallback to standard embedding call if no async support
        return self.embed_many(texts, preprocess, batch_size, as_buffer, **kwargs)

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Asynchronously embed a chunk of text.

        Args:
            text: Text to embed
            preprocess: Optional function to preprocess text
            as_buffer: If True, returns a bytes object instead of a list

        Returns:
            Union[List[float], bytes]: Embedding as a list of floats, or as a bytes
            object if as_buffer=True
        """
        # Fallback to standard embedding call if no async support
        return self.embed(text, preprocess, as_buffer, **kwargs)

    def batchify(self, seq: list, size: int, preprocess: Optional[Callable] = None):
        for pos in range(0, len(seq), size):
            if preprocess is not None:
                yield [preprocess(chunk) for chunk in seq[pos : pos + size]]
            else:
                yield seq[pos : pos + size]

    def _process_embedding(self, embedding: List[float], as_buffer: bool, dtype: str):
        if as_buffer:
            return array_to_buffer(embedding, dtype)
        return embedding

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional

from pydantic.v1 import BaseModel, validator

from redisvl.redis.utils import array_to_buffer
from redisvl.schema.fields import VectorDataType


class Vectorizers(Enum):
    azure_openai = "azure_openai"
    openai = "openai"
    cohere = "cohere"
    mistral = "mistral"
    vertexai = "vertexai"
    hf = "hf"


class BaseVectorizer(BaseModel, ABC):
    model: str
    dims: int
    dtype: str

    @property
    def type(self) -> str:
        return "base"

    @validator("dtype")
    def check_dtype(dtype):
        try:
            VectorDataType(dtype.upper())
        except ValueError:
            raise ValueError(
                f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
            )
        return dtype

    @validator("dims")
    @classmethod
    def check_dims(cls, value):
        """Ensures the dims are a positive integer."""
        if value <= 0:
            raise ValueError("Dims must be a positive integer.")
        return value

    @abstractmethod
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        raise NotImplementedError

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        # Fallback to standard embedding call if no async support
        return self.embed_many(texts, preprocess, batch_size, as_buffer, **kwargs)

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
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

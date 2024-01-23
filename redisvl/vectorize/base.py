from typing import Any, Callable, List, Optional

from pydantic import BaseModel, field_validator

from redisvl.utils.utils import array_to_buffer


class BaseVectorizer(BaseModel):
    model: str
    dims: int
    client: Any

    @field_validator("dims", mode="before")
    @classmethod
    def check_dims(cls, v):
        if v <= 0:
            raise ValueError("Dimension must be a positive integer")
        return v

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        raise NotImplementedError

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
        raise NotImplementedError

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        raise NotImplementedError

    def batchify(self, seq: list, size: int, preprocess: Optional[Callable] = None):
        for pos in range(0, len(seq), size):
            if preprocess is not None:
                yield [preprocess(chunk) for chunk in seq[pos : pos + size]]
            else:
                yield seq[pos : pos + size]

    def _process_embedding(self, embedding: List[float], as_buffer: bool):
        if as_buffer:
            return array_to_buffer(embedding)
        return embedding

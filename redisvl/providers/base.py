from typing import Callable, Dict, List, Optional


class BaseProvider:
    def __init__(self, model: str, dims: int, api_config: Optional[Dict] = None):
        self._dims = dims
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    def set_model(self, model, dims: int = None):
        self._model = model
        if dims:
            self._dims = dims

    def embed_many(
        self, inputs: List[str], preprocess: callable = None, chunk_size: int = 1000
    ) -> List[float]:
        raise NotImplementedError

    def embed(self, emb_input: str, preprocess: callable = None) -> List[float]:
        raise NotImplementedError

    async def aembed_many(
        self, inputs: List[str], preprocess: callable = None, chunk_size: int = 1000
    ) -> List[float]:
        raise NotImplementedError

    async def aembed(self, emb_input: str, preprocess: callable = None) -> List[float]:
        raise NotImplementedError

    def batchify(self, seq: list, size: int, preprocess: Callable = None):
        for pos in range(0, len(seq), size):
            if preprocess:
                yield [preprocess(chunk) for chunk in seq[pos : pos + size]]
            else:
                yield [chunk for chunk in seq[pos : pos + size]]

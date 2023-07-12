from typing import Callable, Dict, List, Optional

from redisvl.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        dims = 1536
        super().__init__(model, dims, api_config)
        if not api_config:
            raise ValueError("OpenAI API key is required in api_config")
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires openai library. Please install with pip install openai"
            )
        openai.api_key = api_config.get("api_key", None)
        self._model_client = openai.Embedding

    def embed_many(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        chunk_size: int = 1000,
    ) -> List[List[float]]:
        results = []
        for batch in self.batchify(inputs, chunk_size, preprocess):
            response = self._model_client.create(input=batch, engine=self._model)
            results += [r["embedding"] for r in response["data"]]
        return results

    def embed(
        self, emb_input: str, preprocess: Optional[Callable] = None
    ) -> List[float]:
        if preprocess:
            emb_input = preprocess(emb_input)
        result = self._model_client.create(input=[emb_input], engine=self._model)
        return result["data"][0]["embedding"]

    async def aembed_many(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        chunk_size: int = 1000,
    ) -> List[List[float]]:
        results = []
        for batch in self.batchify(inputs, chunk_size, preprocess):
            response = await self._model_client.acreate(input=batch, engine=self._model)
            results += [r["embedding"] for r in response["data"]]
        return results

    async def aembed(
        self, emb_input: str, preprocess: Optional[Callable] = None
    ) -> List[float]:
        if preprocess:
            emb_input = preprocess(emb_input)
        result = await self._model_client.acreate(input=[emb_input], engine=self._model)
        return result["data"][0]["embedding"]

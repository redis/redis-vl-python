from typing import Callable, Dict, List, Optional

from redisvl.vectorize.base import BaseVectorizer
from redisvl.utils.utils import array_to_buffer

class OpenAITextVectorizer(BaseVectorizer):
    # TODO - add docstring
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        dims = 1536
        super().__init__(model, dims, api_config)
        if not api_config:
            raise ValueError("OpenAI API key is required in api_config")
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI vectorizer requires openai library. Please install with pip install openai"
            )
        openai.api_key = api_config.get("api_key", None)
        self._model_client = openai.Embedding

    def _process_embedding(self, embedding: List[float], as_buffer: bool):
        if as_buffer:
            return array_to_buffer(embedding)
        return embedding

    def embed_many(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = 10,
        as_buffer: Optional[float] = False
    ) -> List[List[float]]:
        """Embed many chunks of texts using the OpenAI API.

        Args:
            inputs (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding to a byte string. Defaults to False.

        Returns:
            List[List[float]]: _description_
        """
        embeddings: List = []
        for batch in self.batchify(inputs, batch_size, preprocess):
            response = self._model_client.create(input=batch, engine=self._model)
            embeddings += [
                self._process_embedding(r["embedding"], as_buffer) for r in response["data"]
            ]
        return embeddings

    def embed(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = 10,
        as_buffer: Optional[float] = False
    ) -> List[float]:
        """Embed chunks of texts using the OpenAI API.

        Args:
            inputs (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding to a byte string. Defaults to False.

        Returns:
            List[List[float]]: _description_
        """
        if preprocess:
            emb_input = preprocess(emb_input)
        result = self._model_client.create(input=[emb_input], engine=self._model)
        return self._process_embedding(result["data"][0]["embedding"], as_buffer)


    async def aembed_many(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        chunk_size: int = 1000,
        as_buffer: Optional[bool] = False
    ) -> List[List[float]]:
        """_summary_

        Args:
            inputs (List[str]): _description_
            preprocess (Optional[Callable], optional): _description_. Defaults to None.
            chunk_size (int, optional): _description_. Defaults to 1000.
            as_buffer (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            List[List[float]]: _description_
        """
        embeddings: List = []
        for batch in self.batchify(inputs, chunk_size, preprocess):
            response = await self._model_client.acreate(input=batch, engine=self._model)
            embeddings += [
                self._process_embedding(r["embedding"], as_buffer) for r in response["data"]
            ]
        return embeddings

    async def aembed(
        self,
        emb_input: str,
        preprocess: Optional[Callable] = None,
        as_buffer: Optional[bool] = False
    ) -> List[float]:
        """_summary_

        Args:
            emb_input (str): _description_
            preprocess (Optional[Callable], optional): _description_. Defaults to None.
            as_buffer (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            List[float]: _description_
        """
        if preprocess:
            emb_input = preprocess(emb_input)
        result = await self._model_client.acreate(input=[emb_input], engine=self._model)
        return self._process_embedding(result["data"][0]["embedding"], as_buffer)

from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer


class OpenAITextVectorizer(BaseVectorizer):
    # TODO - add docstring
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        super().__init__(model)
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
        self._dims = self._model_dims()

    def _model_dims(self):
        embedding = self._model_client.create(
            input=["dimension test"],
            engine=self._model
        )["data"][0]["embedding"]
        return len(embedding)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = 10,
        as_buffer: Optional[float] = False,
    ) -> List[List[float]]:
        """Embed many chunks of texts using the OpenAI API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
                raise TypeError("Must pass in a list of str values to embed.")
        if  len(texts) > 0 and not isinstance(texts[0], str):
                raise TypeError("Must pass in a list of str values to embed.")

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._model_client.create(input=batch, engine=self._model)
            embeddings += [
                self._process_embedding(r["embedding"], as_buffer)
                for r in response["data"]
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: Optional[float] = False,
    ) -> List[float]:
        """Embed a chunk of text using the OpenAI API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)
        result = self._model_client.create(input=[text], engine=self._model)
        return self._process_embedding(result["data"][0]["embedding"], as_buffer)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: Optional[bool] = False,
    ) -> List[List[float]]:
        """Asynchronously embed many chunks of texts using the OpenAI API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
                raise TypeError("Must pass in a list of str values to embed.")
        if  len(texts) > 0 and not isinstance(texts[0], str):
                raise TypeError("Must pass in a list of str values to embed.")

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = await self._model_client.acreate(input=batch, engine=self._model)
            embeddings += [
                self._process_embedding(r["embedding"], as_buffer)
                for r in response["data"]
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: Optional[bool] = False,
    ) -> List[float]:
        """Asynchronously embed a chunk of text using the OpenAI API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)
        result = await self._model_client.acreate(input=[text], engine=self._model)
        return self._process_embedding(result["data"][0]["embedding"], as_buffer)

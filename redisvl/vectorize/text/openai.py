from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer


class OpenAITextVectorizer(BaseVectorizer):
    """OpenAI text vectorizer

    This vectorizer uses the OpenAI API to create embeddings for text. It requires an
    API key to be passed in the api_config dictionary. The API key can be obtained from
    https://api.openai.com/.
    """
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        """Initialize the OpenAI vectorizer.

        Args:
            model (str): Model to use for embedding.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.

        Raises:
            ImportError: If the openai library is not installed.
            ValueError: If the API key is not provided.
        """
        super().__init__(model)
        # Dynamic import of the openai module
        try:
            global openai
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI vectorizer requires the openai library. Please install with pip install openai"
            )

        if not api_config or "api_key" not in api_config:
            raise ValueError("OpenAI API key is required in api_config")

        openai.api_key = api_config["api_key"]
        self._model_client = openai.Embedding
        self._dims = self._set_model_dims()

    def _set_model_dims(self) -> int:
        try:
            embedding = self._model_client.create(
                input=["dimension test"],
                engine=self._model
            )["data"][0]["embedding"]
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the OpenAI API: {str(ke)}")
        except openai.error.AuthenticationError as ae:
            raise ValueError(f"Error authenticating with the OpenAI API: {str(ae)}")
        except Exception as e: # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
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

import os
from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer

# ignore that openai isn't imported
# mypy: disable-error-code="name-defined"


class OpenAITextVectorizer(BaseVectorizer):
    """The OpenAITextVectorizer class utilizes OpenAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with OpenAI's embeddings API,
    requiring an API key for authentication. The key can be provided directly
    in the `api_config` dictionary or through the `OPENAI_API_KEY` environment
    variable. Users must obtain an API key from OpenAI's website
    (https://api.openai.com/). Additionally, the `openai` python client must be
    installed with `pip install openai==0.28.1`.

    The vectorizer supports both synchronous and asynchronous operations,
    allowing for batch processing of texts and flexibility in handling
    preprocessing tasks.

    .. code-block:: python

        # Synchronous embedding of a single text
        vectorizer = OpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={"api_key": "your_api_key"} # OR set OPENAI_API_KEY in your env
        )
        embedding = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

    """

    def __init__(
        self, model: str = "text-embedding-ada-002", api_config: Optional[Dict] = None
    ):
        """Initialize the OpenAI vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to
                'text-embedding-ada-002'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API key. Defaults to None.

        Raises:
            ImportError: If the openai library is not installed.
            ValueError: If the OpenAI API key is not provided.
        """
        # Dynamic import of the openai module
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI vectorizer requires the openai library. \
                    Please install with `pip install openai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it in api_config or set the OPENAI_API_KEY\
                    environment variable."
            )

        openai.api_key = api_key
        client = openai.Embedding
        dims = self._set_model_dims(client, model)
        super().__init__(model=model, dims=dims, client=client)

    @staticmethod
    def _set_model_dims(client, model) -> int:
        try:
            embedding = client.create(input=["dimension test"], engine=model)["data"][
                0
            ]["embedding"]
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the OpenAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
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
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Embed many chunks of texts using the OpenAI API.

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
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self.client.create(input=batch, engine=self.model)
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
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Embed a chunk of text using the OpenAI API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
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
        result = self.client.create(input=[text], engine=self.model)
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
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Asynchronously embed many chunks of texts using the OpenAI API.

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
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = await self.client.acreate(input=batch, engine=self.model)
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
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Asynchronously embed a chunk of text using the OpenAI API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
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
        result = await self.client.acreate(input=[text], engine=self.model)
        return self._process_embedding(result["data"][0]["embedding"], as_buffer)

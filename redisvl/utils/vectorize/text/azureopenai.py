import os
from typing import Any, Callable, Dict, List, Optional

from pydantic.v1 import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that openai isn't imported
# mypy: disable-error-code="name-defined"


class AzureOpenAITextVectorizer(BaseVectorizer):
    """The AzureOpenAITextVectorizer class utilizes AzureOpenAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with AzureOpenAI's embeddings API,
    requiring an API key, an AzureOpenAI deployment endpoint and API version.
    These values can be provided directly in the `api_config` dictionary with
    the parameters 'azure_endpoint', 'api_version' and 'api_key' or through the
    environment variables 'AZURE_OPENAI_ENDPOINT', 'OPENAI_API_VERSION', and 'AZURE_OPENAI_API_KEY'.
    Users must obtain these values from the 'Keys and Endpoints' section in their Azure OpenAI service.
    Additionally, the `openai` python client must be installed with `pip install openai>=1.13.0`.

    The vectorizer supports both synchronous and asynchronous operations,
    allowing for batch processing of texts and flexibility in handling
    preprocessing tasks.

    .. code-block:: python

        # Synchronous embedding of a single text
        vectorizer = AzureOpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={
                "api_key": "your_api_key", # OR set AZURE_OPENAI_API_KEY in your env
                "api_version": "your_api_version", # OR set OPENAI_API_VERSION in your env
                "azure_endpoint": "your_azure_endpoint", # OR set AZURE_OPENAI_ENDPOINT in your env
            }
        )
        embedding = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

    """

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self, model: str = "text-embedding-ada-002", api_config: Optional[Dict] = None
    ):
        """Initialize the AzureOpenAI vectorizer.

        Args:
            model (str): Deployment to use for embedding. Must be the
                'Deployment name' not the 'Model name'. Defaults to
                'text-embedding-ada-002'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API key, API version, Azure endpoint, and any other API options.
                Defaults to None.

        Raises:
            ImportError: If the openai library is not installed.
            ValueError: If the AzureOpenAI API key, version, or endpoint are not provided.
        """
        self._initialize_clients(api_config)
        super().__init__(model=model, dims=self._set_model_dims(model))

    def _initialize_clients(self, api_config: Optional[Dict]):
        """
        Setup the OpenAI clients using the provided API key or an
        environment variable.
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the openai module
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError(
                "AzureOpenAI vectorizer requires the openai library. \
                    Please install with `pip install openai`"
            )

        # Fetch the API key, version and endpoint from api_config or environment variable
        azure_endpoint = (
            api_config.pop("azure_endpoint")
            if api_config
            else os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        if not azure_endpoint:
            raise ValueError(
                "AzureOpenAI API endpoint is required. "
                "Provide it in api_config or set the AZURE_OPENAI_ENDPOINT\
                    environment variable."
            )

        api_version = (
            api_config.pop("api_version")
            if api_config
            else os.getenv("OPENAI_API_VERSION")
        )

        if not api_version:
            raise ValueError(
                "AzureOpenAI API version is required. "
                "Provide it in api_config or set the OPENAI_API_VERSION\
                    environment variable."
            )

        api_key = (
            api_config.pop("api_key")
            if api_config
            else os.getenv("AZURE_OPENAI_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "AzureOpenAI API key is required. "
                "Provide it in api_config or set the AZURE_OPENAI_API_KEY\
                    environment variable."
            )

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            **api_config,
        )
        self._aclient = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            **api_config,
        )

    def _set_model_dims(self, model) -> int:
        try:
            embedding = (
                self._client.embeddings.create(input=["dimension test"], model=model)
                .data[0]
                .embedding
            )
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the AzureOpenAI API: {str(ke)}")
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
        """Embed many chunks of texts using the AzureOpenAI API.

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
            response = self._client.embeddings.create(input=batch, model=self.model)
            embeddings += [
                self._process_embedding(r.embedding, as_buffer, **kwargs)
                for r in response.data
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
        """Embed a chunk of text using the AzureOpenAI API.

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
        result = self._client.embeddings.create(input=[text], model=self.model)
        return self._process_embedding(result.data[0].embedding, as_buffer, **kwargs)

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
        """Asynchronously embed many chunks of texts using the AzureOpenAI API.

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
            response = await self._aclient.embeddings.create(
                input=batch, model=self.model
            )
            embeddings += [
                self._process_embedding(r.embedding, as_buffer, **kwargs)
                for r in response.data
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
        result = await self._aclient.embeddings.create(input=[text], model=self.model)
        return self._process_embedding(result.data[0].embedding, as_buffer, **kwargs)

    @property
    def type(self) -> str:
        return "azure_openai"

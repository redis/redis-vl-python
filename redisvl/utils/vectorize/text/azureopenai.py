import os
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
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

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        # Basic usage
        vectorizer = AzureOpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={
                "api_key": "your_api_key", # OR set AZURE_OPENAI_API_KEY in your env
                "api_version": "your_api_version", # OR set OPENAI_API_VERSION in your env
                "azure_endpoint": "your_azure_endpoint", # OR set AZURE_OPENAI_ENDPOINT in your env
            }
        )
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="azureopenai_embeddings_cache")

        vectorizer = AzureOpenAITextVectorizer(
            model="text-embedding-ada-002",
            api_config={
                "api_key": "your_api_key",
                "api_version": "your_api_version",
                "azure_endpoint": "your_azure_endpoint",
            },
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the AzureOpenAI vectorizer.

        Args:
            model (str): Deployment to use for embedding. Must be the
                'Deployment name' not the 'Model name'. Defaults to
                'text-embedding-ada-002'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API key, API version, Azure endpoint, and any other API options.
                Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the openai library is not installed.
            ValueError: If the AzureOpenAI API key, version, or endpoint are not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize clients and set up the model
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the AzureOpenAI clients and determine the embedding dimensions."""
        # Initialize clients
        self._initialize_clients(api_config, **kwargs)
        # Set model dimensions after client initialization
        self.dims = self._set_model_dims()

    def _initialize_clients(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the AzureOpenAI clients using the provided API key, API version,
        and Azure endpoint.

        Args:
            api_config: Dictionary with API configuration options
            **kwargs: Additional arguments to pass to AzureOpenAI clients

        Raises:
            ImportError: If the openai library is not installed
            ValueError: If required parameters are not provided
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the openai module
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError(
                "AzureOpenAI vectorizer requires the openai library. "
                "Please install with `pip install openai>=1.13.0`"
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
                "Provide it in api_config or set the AZURE_OPENAI_ENDPOINT environment variable."
            )

        api_version = (
            api_config.pop("api_version")
            if api_config
            else os.getenv("OPENAI_API_VERSION")
        )

        if not api_version:
            raise ValueError(
                "AzureOpenAI API version is required. "
                "Provide it in api_config or set the OPENAI_API_VERSION environment variable."
            )

        api_key = (
            api_config.pop("api_key")
            if api_config
            else os.getenv("AZURE_OPENAI_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "AzureOpenAI API key is required. "
                "Provide it in api_config or set the AZURE_OPENAI_API_KEY environment variable."
            )

        # Store clients as regular attributes instead of PrivateAttr
        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            **api_config,
            **kwargs,
        )
        self._aclient = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            **api_config,
            **kwargs,
        )

    def _set_model_dims(self) -> int:
        """
        Determine the dimensionality of the embedding model by making a test call.

        Returns:
            int: Dimensionality of the embedding model

        Raises:
            ValueError: If embedding dimensions cannot be determined
        """
        try:
            # Call the protected _embed method to avoid caching this test embedding
            embedding = self._embed("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the AzureOpenAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate a vector embedding for a single text using the AzureOpenAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the AzureOpenAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        try:
            result = self._client.embeddings.create(
                input=[text], model=self.model, **kwargs
            )
            return result.data[0].embedding
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of texts using the AzureOpenAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the AzureOpenAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        try:
            embeddings: List = []
            for batch in self.batchify(texts, batch_size):
                response = self._client.embeddings.create(
                    input=batch, model=self.model, **kwargs
                )
                embeddings.extend([r.embedding for r in response.data])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed(self, text: str, **kwargs) -> List[float]:
        """
        Asynchronously generate a vector embedding for a single text using the AzureOpenAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the AzureOpenAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        try:
            result = await self._aclient.embeddings.create(
                input=[text], model=self.model, **kwargs
            )
            return result.data[0].embedding
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Asynchronously generate vector embeddings for a batch of texts using the AzureOpenAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the AzureOpenAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        try:
            embeddings: List = []
            for batch in self.batchify(texts, batch_size):
                response = await self._aclient.embeddings.create(
                    input=batch, model=self.model, **kwargs
                )
                embeddings.extend([r.embedding for r in response.data])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @property
    def type(self) -> str:
        return "azure_openai"

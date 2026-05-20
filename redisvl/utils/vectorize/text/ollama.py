from typing import TYPE_CHECKING

from pydantic import ConfigDict
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.utils.vectorize.base import BaseVectorizer

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

# ignore that ollama isn't imported
# mypy: disable-error-code="name-defined"


class OllamaTextVectorizer(BaseVectorizer):
    """The OllamaTextVectorizer class uses Ollama to generate embeddings for
    text data.

    This vectorizer is designed to interact with a local Ollama server through
    Ollama's embedding API. Ollama must be installed and running locally
    (https://ollama.com/download), and the selected embedding model must be
    pulled before use. You can browse Ollama embedding models at
    https://ollama.com/search?c=embedding&p=1. Additionally, the `ollama`
    python client must be installed with `pip install ollama`.

    The vectorizer supports both synchronous and asynchronous operations,
    allowing for batch processing of texts and flexibility in handling
    preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        # Basic usage with Ollama embeddings
        # Install Ollama, then run: ollama pull nomic-embed-text
        vectorizer = OllamaTextVectorizer(
            model="nomic-embed-text"
        )
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="ollama_embeddings_cache")

        vectorizer = OllamaTextVectorizer(
            model="nomic-embed-text",
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
        model: str = "nomic-embed-text",
        dtype: str = "float32",
        host: str | None = None,
        cache: "EmbeddingsCache | None" = None,
        **kwargs,
    ):
        """Initialize the Ollama vectorizer.

        Args:
            model (str): Ollama embedding model to use. The model must already be
                available to the local Ollama server. Defaults to 'nomic-embed-text'.
            dtype (str): The default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            host (Optional[str]): The host URL for the Ollama server. If None, the
                Ollama client resolves the host from the OLLAMA_HOST environment
                variable, falling back to 'http://localhost:11434'. Defaults to None.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache
                embeddings for better performance with repeated texts. Defaults to None.
            **kwargs: Additional arguments to pass to the Ollama sync and async clients.
                Examples include `timeout`, `headers`, and `follow_redirects`.

        Raises:
            ImportError: If the ollama library is not installed.
            ValueError: If an invalid dtype is provided.
            ValueError: If embedding dimensions cannot be determined.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)

        # Ollama-specific client and model initialization
        self._setup(host=host, **kwargs)

    def _setup(self, host: str | None = None, **kwargs):
        """Set up the Ollama clients and determine the embedding dimensions."""
        self._initialize_clients(host=host, **kwargs)
        self.dims = self._set_model_dims()

    def _initialize_clients(self, host: str | None = None, **kwargs):
        """
        Setup the Ollama clients using the provided host or Ollama defaults.

        Args:
            host: Optional Ollama server URL
            **kwargs: Additional arguments to pass to Ollama clients

        Raises:
            ImportError: If the ollama library is not installed
        """
        # Dynamic import of the ollama module
        try:
            from ollama import AsyncClient, Client
        except ImportError:
            raise ImportError(
                "The ollama package is required to use OllamaTextVectorizer. "
                "Please install with `pip install ollama`"
            )

        self._client = Client(host=host, **kwargs)
        self._aclient = AsyncClient(host=host, **kwargs)

    def _set_model_dims(self) -> int:
        """
        Determine the dimensionality of the embedding model by making a test call.

        Returns:
            int: Dimensionality of the embedding model

        Raises:
            ConnectionError: If the Ollama server cannot be reached
            ValueError: If embedding dimensions cannot be determined
        """
        try:
            embedding = self._embed("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the Ollama API: {str(ke)}")
        except ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at the configured host. "
                f"Is the daemon running? Check `ollama serve` and your host settings e.g url and port. "
                f"Underlying error: {e}"
            )
        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ConnectionError)),
    )
    def _embed(self, content: str, **kwargs) -> list[float]:
        """Generate a vector embedding for a single text using the Ollama API.

        Args:
            content: Text to embed
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If content is not a string
            ConnectionError: If the Ollama server cannot be reached
            ValueError: If embedding fails
        """
        if not isinstance(content, str):
            raise TypeError(
                f"Input content must be a string to embed, got {type(content)}"
            )
        try:
            result = self._client.embed(model=self.model, input=content, **kwargs)
            return result["embeddings"][0]
        except ConnectionError:
            raise
        except Exception as e:
            raise ValueError(f"Error generating embedding with Ollama: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ConnectionError)),
    )
    def _embed_many(
        self, contents: list[str], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        """Generate vector embeddings for a batch of texts using the Ollama API.

        Args:
            contents: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If contents is not a list of strings
            ConnectionError: If the Ollama server cannot be reached
            ValueError: If embedding fails
        """

        if not isinstance(contents, list) or not all(
            isinstance(c, str) for c in contents
        ):
            raise TypeError(
                f"Input contents must be a list of strings to embed, got {type(contents)} with elements of types {[type(c) for c in contents]}"
            )

        embeddings: list[list[float]] = []

        for batch in self.batchify(contents, batch_size):
            try:
                response = self._client.embed(model=self.model, input=batch, **kwargs)
                embeddings.extend(response["embeddings"])
            except ConnectionError:
                raise
            except Exception as e:
                raise ValueError(f"Embedding text failed: {e}")

        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ConnectionError)),
    )
    async def _aembed(self, content: str, **kwargs) -> list[float]:
        """Asynchronously generate a vector embedding for a single text using the Ollama API.

        Args:
            content: Text to embed
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If content is not a string
            ConnectionError: If the Ollama server cannot be reached
            ValueError: If embedding fails
        """
        if not isinstance(content, str):
            raise TypeError(
                f"Input content must be a string to embed, got {type(content)}"
            )

        try:
            result = await self._aclient.embed(
                model=self.model, input=content, **kwargs
            )
            return result["embeddings"][0]
        except ConnectionError:
            raise
        except Exception as e:
            raise ValueError(f"Error generating embedding with Ollama: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type((TypeError, ConnectionError)),
    )
    async def _aembed_many(
        self, contents: list[str], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        """Asynchronously generate vector embeddings for a batch of texts using the Ollama API.

        Args:
            contents: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the Ollama API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If contents is not a list of strings
            ConnectionError: If the Ollama server cannot be reached
            ValueError: If embedding fails
        """
        if not isinstance(contents, list) or not all(
            isinstance(c, str) for c in contents
        ):
            raise TypeError(
                f"Input contents must be a list of strings to embed, got {type(contents)} with elements of types {[type(c) for c in contents]}"
            )

        embeddings: list[list[float]] = []

        for batch in self.batchify(contents, batch_size):
            try:
                response = await self._aclient.embed(
                    model=self.model, input=batch, **kwargs
                )
                embeddings.extend(response["embeddings"])
            except ConnectionError:
                raise
            except Exception as e:
                raise ValueError(f"Embedding texts failed: {e}")
        return embeddings

    @property
    def type(self) -> str:
        return "ollama"

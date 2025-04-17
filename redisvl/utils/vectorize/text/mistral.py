import os
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that mistralai isn't imported
# mypy: disable-error-code="name-defined"


class MistralAITextVectorizer(BaseVectorizer):
    """The MistralAITextVectorizer class utilizes MistralAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Mistral's embeddings API,
    requiring an API key for authentication. The key can be provided directly
    in the `api_config` dictionary or through the `MISTRAL_API_KEY` environment
    variable. Users must obtain an API key from Mistral's website
    (https://console.mistral.ai/). Additionally, the `mistralai` python client
    must be installed with `pip install mistralai`.

    The vectorizer supports both synchronous and asynchronous operations,
    allowing for batch processing of texts and flexibility in handling
    preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        # Basic usage
        vectorizer = MistralAITextVectorizer(
            model="mistral-embed",
            api_config={"api_key": "your_api_key"} # OR set MISTRAL_API_KEY in your env
        )
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="mistral_embeddings_cache")

        vectorizer = MistralAITextVectorizer(
            model="mistral-embed",
            api_config={"api_key": "your_api_key"},
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
        model: str = "mistral-embed",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the MistralAI vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to
                'mistral-embed'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API key. Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the mistralai library is not installed.
            ValueError: If the Mistral API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the MistralAI client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the Mistral client using the provided API key or an
        environment variable.

        Args:
            api_config: Dictionary with API configuration options
            **kwargs: Additional arguments to pass to MistralAI client

        Raises:
            ImportError: If the mistralai library is not installed
            ValueError: If no API key is provided
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the mistralai module
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "MistralAI vectorizer requires the mistralai library. "
                "Please install with `pip install mistralai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("MISTRAL_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "MISTRAL API key is required. "
                "Provide it in api_config or set the MISTRAL_API_KEY environment variable."
            )

        # Store client as a regular attribute instead of PrivateAttr
        self._client = Mistral(api_key=api_key, **kwargs)

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
            raise ValueError(f"Unexpected response from the MISTRAL API: {str(ke)}")
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
        Generate a vector embedding for a single text using the MistralAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the MistralAI API

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
                model=self.model, inputs=[text], **kwargs
            )
            return result.data[0].embedding  # type: ignore
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
        Generate vector embeddings for a batch of texts using the MistralAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the MistralAI API

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
                    model=self.model, inputs=batch, **kwargs
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
        Asynchronously generate a vector embedding for a single text using the MistralAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the MistralAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        try:
            result = await self._client.embeddings.create_async(
                model=self.model, inputs=[text], **kwargs
            )
            return result.data[0].embedding  # type: ignore
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
        Asynchronously generate vector embeddings for a batch of texts using the MistralAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the MistralAI API

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
                response = await self._client.embeddings.create_async(
                    model=self.model, inputs=batch, **kwargs
                )
                embeddings.extend([r.embedding for r in response.data])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @property
    def type(self) -> str:
        return "mistral"

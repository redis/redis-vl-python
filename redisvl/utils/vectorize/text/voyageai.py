import os
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that voyageai isn't imported
# mypy: disable-error-code="name-defined"


class VoyageAITextVectorizer(BaseVectorizer):
    """The VoyageAITextVectorizer class utilizes VoyageAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with VoyageAI's /embed API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `VOYAGE_API_KEY`
    environment variable. User must obtain an API key from VoyageAI's website
    (https://dash.voyageai.com/). Additionally, the `voyageai` python
    client must be installed with `pip install voyageai`.

    The vectorizer supports both synchronous and asynchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        from redisvl.utils.vectorize import VoyageAITextVectorizer

        # Basic usage
        vectorizer = VoyageAITextVectorizer(
            model="voyage-large-2",
            api_config={"api_key": "your-voyageai-api-key"} # OR set VOYAGE_API_KEY in your env
        )
        query_embedding = vectorizer.embed(
            text="your input query text here",
            input_type="query"
        )
        doc_embeddings = vectorizer.embed_many(
            texts=["your document text", "more document text"],
            input_type="document"
        )

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="voyageai_embeddings_cache")

        vectorizer = VoyageAITextVectorizer(
            model="voyage-large-2",
            api_config={"api_key": "your-voyageai-api-key"},
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed(
            text="your input query text here",
            input_type="query"
        )

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed(
            text="your input query text here",
            input_type="query"
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "voyage-large-2",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the VoyageAI vectorizer.

        Visit https://docs.voyageai.com/docs/embeddings to learn about embeddings and check the available models.

        Args:
            model (str): Model to use for embedding. Defaults to "voyage-large-2".
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the voyageai library is not installed.
            ValueError: If the API key is not provided.

        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the VoyageAI client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the VoyageAI clients using the provided API key or an
        environment variable.

        Args:
            api_config: Dictionary with API configuration options
            **kwargs: Additional arguments to pass to VoyageAI clients

        Raises:
            ImportError: If the voyageai library is not installed
            ValueError: If no API key is provided
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the voyageai module
        try:
            from voyageai import AsyncClient, Client
        except ImportError:
            raise ImportError(
                "VoyageAI vectorizer requires the voyageai library. "
                "Please install with `pip install voyageai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("VOYAGE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "VoyageAI API key is required. "
                "Provide it in api_config or set the VOYAGE_API_KEY environment variable."
            )

        self._client = Client(api_key=api_key, **kwargs)
        self._aclient = AsyncClient(api_key=api_key, **kwargs)

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
            embedding = self._embed("dimension check", input_type="document")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the VoyageAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    def _get_batch_size(self) -> int:
        """
        Determine the appropriate batch size based on the model being used.

        Returns:
            int: Recommended batch size for the current model
        """
        if self.model in ["voyage-2", "voyage-02"]:
            return 72
        elif self.model in ["voyage-3-lite", "voyage-3.5-lite"]:
            return 30
        elif self.model in ["voyage-3", "voyage-3.5"]:
            return 10
        else:
            return 7  # Default for other models

    def _validate_input(
        self, texts: List[str], input_type: Optional[str], truncation: Optional[bool]
    ):
        """
        Validate the inputs to the embedding methods.

        Args:
            texts: List of texts to embed
            input_type: Type of input (document or query)
            truncation: Whether to truncate long texts

        Raises:
            TypeError: If inputs are invalid
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if input_type is not None and input_type not in ["document", "query"]:
            raise TypeError(
                "Must pass in a allowed value for voyageai embedding input_type. "
                "See https://docs.voyageai.com/docs/embeddings."
            )
        if truncation is not None and not isinstance(truncation, bool):
            raise TypeError("Truncation (optional) parameter is a bool.")

    def _embed(self, text: str, **kwargs) -> List[float]:
        """
        Generate a vector embedding for a single text using the VoyageAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string or parameters are invalid
            ValueError: If embedding fails
        """
        # Simply call _embed_many with a single text and return the first result
        result = self._embed_many([text], **kwargs)
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, texts: List[str], batch_size: Optional[int] = None, **kwargs
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of texts using the VoyageAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings or parameters are invalid
            ValueError: If embedding fails
        """
        input_type = kwargs.pop("input_type", None)
        truncation = kwargs.pop("truncation", None)

        # Validate inputs
        self._validate_input(texts, input_type, truncation)

        # Determine batch size if not provided
        if batch_size is None:
            batch_size = self._get_batch_size()

        try:
            embeddings: List = []
            for batch in self.batchify(texts, batch_size):
                response = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation,
                    **kwargs,
                )
                embeddings.extend(response.embeddings)
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    async def _aembed(self, text: str, **kwargs) -> List[float]:
        """
        Asynchronously generate a vector embedding for a single text using the VoyageAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string or parameters are invalid
            ValueError: If embedding fails
        """
        # Simply call _aembed_many with a single text and return the first result
        result = await self._aembed_many([text], **kwargs)
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed_many(
        self, texts: List[str], batch_size: Optional[int] = None, **kwargs
    ) -> List[List[float]]:
        """
        Asynchronously generate vector embeddings for a batch of texts using the VoyageAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If texts is not a list of strings or parameters are invalid
            ValueError: If embedding fails
        """
        input_type = kwargs.pop("input_type", None)
        truncation = kwargs.pop("truncation", None)

        # Validate inputs
        self._validate_input(texts, input_type, truncation)

        # Determine batch size if not provided
        if batch_size is None:
            batch_size = self._get_batch_size()

        try:
            embeddings: List = []
            for batch in self.batchify(texts, batch_size):
                response = await self._aclient.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation,
                    **kwargs,
                )
                embeddings.extend(response.embeddings)
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @property
    def type(self) -> str:
        return "voyageai"

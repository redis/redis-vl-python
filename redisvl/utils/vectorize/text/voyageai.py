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

# Token limits for different VoyageAI models
VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-context-3": 32_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
    "voyage-multimodal-3": 120_000,
}


class VoyageAITextVectorizer(BaseVectorizer):
    """The VoyageAITextVectorizer class utilizes VoyageAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with VoyageAI's /embed API and
    /contextualized_embed API (for context models like voyage-context-3),
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `VOYAGE_API_KEY`
    environment variable. User must obtain an API key from VoyageAI's website
    (https://dash.voyageai.com/). Additionally, the `voyageai` python
    client must be installed with `pip install voyageai`.

    The vectorizer supports both synchronous and asynchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks. It automatically
    detects and handles contextualized embedding models (like voyage-context-3) which
    generate embeddings that are aware of the surrounding context within a document.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs. The vectorizer also provides token counting
    capabilities to help manage API usage and optimize batching strategies.

    .. code-block:: python

        from redisvl.utils.vectorize import VoyageAITextVectorizer

        # Basic usage
        vectorizer = VoyageAITextVectorizer(
            model="voyage-3.5",
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
            model="voyage-3.5",
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

        # Using contextualized embeddings (voyage-context-3)
        context_vectorizer = VoyageAITextVectorizer(
            model="voyage-context-3",
            api_config={"api_key": "your-voyageai-api-key"}
        )
        # Context models automatically use contextualized_embed API
        # which generates context-aware embeddings for document chunks
        context_embeddings = context_vectorizer.embed_many(
            texts=["chunk 1 of document", "chunk 2 of document", "chunk 3 of document"],
            input_type="document"
        )

        # Token counting for API usage management
        token_counts = vectorizer.count_tokens(["text one", "text two"])
        print(f"Token counts: {token_counts}")
        print(f"Model token limit: {VOYAGE_TOTAL_TOKEN_LIMITS.get(vectorizer.model, 120_000)}")

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str,
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the VoyageAI vectorizer.

        Visit https://docs.voyageai.com/docs/embeddings to learn about embeddings and check the available models.

        Args:
            model (str): Model to use for embedding (e.g., "voyage-3.5", "voyage-context-3").
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

        Uses token-aware batching to respect model token limits and optimize API calls.

        Args:
            texts: List of texts to embed
            batch_size: Deprecated. Token-aware batching is now always used.
            **kwargs: Additional parameters to pass to the VoyageAI API.

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

        # Use token-aware batching
        batches = self._build_token_aware_batches(texts)

        try:
            embeddings: List = []

            # Use contextualized embed API for context models
            if self._is_context_model():
                for batch in batches:
                    # Context models expect inputs as a list of lists
                    response = self._client.contextualized_embed(
                        inputs=[batch],
                        model=self.model,
                        input_type=input_type,
                        **kwargs,
                    )
                    # Extract embeddings from the first (and only) result
                    embeddings.extend(response.results[0].embeddings)
            else:
                # Use regular embed API for standard models
                for batch in batches:
                    response = self._client.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type,
                        truncation=truncation,  # type: ignore[assignment]
                        **kwargs,
                    )
                    embeddings.extend(response.embeddings)  # type: ignore[attr-defined]
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

        Uses token-aware batching to respect model token limits and optimize API calls.

        Args:
            texts: List of texts to embed
            batch_size: Deprecated. Token-aware batching is now always used.
            **kwargs: Additional parameters to pass to the VoyageAI API.

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

        # Use token-aware batching (synchronous - tokenization is sync-only)
        batches = self._build_token_aware_batches(texts)

        try:
            embeddings: List = []

            # Use contextualized embed API for context models
            if self._is_context_model():
                for batch in batches:
                    # Context models expect inputs as a list of lists
                    response = await self._aclient.contextualized_embed(
                        inputs=[batch],
                        model=self.model,
                        input_type=input_type,
                        **kwargs,
                    )
                    # Extract embeddings from the first (and only) result
                    embeddings.extend(response.results[0].embeddings)
            else:
                # Use regular embed API for standard models
                for batch in batches:
                    response = await self._aclient.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type,
                        truncation=truncation,  # type: ignore[assignment]
                        **kwargs,
                    )
                    embeddings.extend(response.embeddings)  # type: ignore[attr-defined]
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Count tokens for the given texts using VoyageAI's tokenization API.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            List[int]: List of token counts for each text.

        Raises:
            ValueError: If tokenization fails.

        Example:
            >>> vectorizer = VoyageAITextVectorizer(model="voyage-3.5")
            >>> token_counts = vectorizer.count_tokens(["Hello world", "Another text"])
            >>> print(token_counts)  # [2, 2]
        """
        if not texts:
            return []

        try:
            # Use the VoyageAI tokenize API to get token counts
            token_lists = self._client.tokenize(texts, model=self.model)
            return [len(token_list) for token_list in token_lists]
        except Exception as e:
            raise ValueError(f"Token counting failed: {e}")

    def _is_context_model(self) -> bool:
        """
        Check if the current model is a contextualized embedding model.

        Contextualized models (like voyage-context-3) use a different API
        endpoint and expect inputs formatted differently.

        Returns:
            bool: True if the model is a context model, False otherwise.
        """
        return "context" in self.model

    def _build_token_aware_batches(
        self, texts: List[str], max_batch_size: int = 1000
    ) -> List[List[str]]:
        """
        Generate batches of texts based on token limits and batch size constraints.

        This method uses VoyageAI's tokenization API to count tokens for all texts
        in a single call, then creates batches that respect both the model's token
        limit and a maximum batch size.

        Args:
            texts: List of texts to batch.
            max_batch_size: Maximum number of texts per batch (default: 1000).

        Returns:
            List[List[str]]: List of batches, where each batch is a list of texts.

        Raises:
            ValueError: If tokenization fails.
        """
        if not texts:
            return []

        max_tokens_per_batch = VOYAGE_TOTAL_TOKEN_LIMITS.get(self.model, 120_000)
        batches = []
        current_batch: List[str] = []
        current_batch_tokens = 0

        # Tokenize all texts in one API call for efficiency
        try:
            token_counts = self.count_tokens(texts)
        except Exception as e:
            raise ValueError(f"Failed to count tokens for batching: {e}")

        for i, text in enumerate(texts):
            n_tokens = token_counts[i]

            # Check if adding this text would exceed limits
            if current_batch and (
                len(current_batch) >= max_batch_size
                or (current_batch_tokens + n_tokens > max_tokens_per_batch)
            ):
                # Save the current batch and start a new one
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0

            current_batch.append(text)
            current_batch_tokens += n_tokens

        # Add the last batch if it has any texts
        if current_batch:
            batches.append(current_batch)

        return batches

    @property
    def type(self) -> str:
        return "voyageai"

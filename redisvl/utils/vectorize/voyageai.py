import os
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that voyageai isn't imported
# mypy: disable-error-code="name-defined"

# Per-request total token limits by model, used for token-aware batching of the
# plain text embedding path. Values track VoyageAI's documented per-request token
# limits (https://docs.voyageai.com/docs/embeddings). Unknown models fall back to
# the conservative default below.
VOYAGE_TOKEN_LIMITS: dict[str, int] = {
    "voyage-2": 320_000,
    "voyage-02": 320_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-3-large": 120_000,
    "voyage-3.5": 320_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-4-large": 120_000,
    "voyage-4-nano": 120_000,
    "voyage-code-2": 120_000,
    "voyage-code-3": 120_000,
    "voyage-code-4": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-multilingual-2": 120_000,
}
DEFAULT_VOYAGE_TOKEN_LIMIT = 120_000


class VoyageAIVectorizer(BaseVectorizer):
    """The VoyageAIVectorizer class utilizes VoyageAI's API to generate
    embeddings for text and multimodal (text / image / video) data.

    This vectorizer is designed to interact with VoyageAI's /embed, /multimodal_embed,
    and /contextualized_embed APIs. Any model identifier accepted by VoyageAI can be
    passed via ``model`` - for example the general-purpose ``voyage-4-large`` /
    ``voyage-4`` / ``voyage-4-lite`` / ``voyage-4-nano`` family, domain models such
    as ``voyage-code-4``, contextualized ``voyage-context-4`` / ``voyage-context-3``
    models, and multimodal ``voyage-multimodal-*`` models.
    See https://docs.voyageai.com/docs/embeddings for the current catalog.

    It requires an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `VOYAGE_API_KEY`
    environment variable. User must obtain an API key from VoyageAI's website
    (https://dash.voyageai.com/). Additionally, the `voyageai` python
    client must be installed with `pip install voyageai`. For image embeddings, the Pillow
    library must also be installed with `pip install pillow`.

    The vectorizer supports both synchronous and asynchronous operations, allows for batch
    processing of content and flexibility in handling preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        from redisvl.utils.vectorize import VoyageAIVectorizer

        # Basic usage
        vectorizer = VoyageAIVectorizer(
            model="voyage-3-large",
            api_config={"api_key": "your-voyageai-api-key"} # OR set VOYAGE_API_KEY in your env
        )
        query_embedding = vectorizer.embed(
            content="your input query text here",
            input_type="query"
        )
        doc_embeddings = vectorizer.embed_many(
            contents=["your document text", "more document text"],
            input_type="document"
        )

        # Contextualized embeddings (voyage-context-* models) - requires voyageai>=0.5.0
        # Each input string is treated as its own document (auto-chunked) and
        # embedded independently: inputs do not influence one another, which keeps
        # the one-embedding-per-input contract and cache determinism intact.
        context_vectorizer = VoyageAIVectorizer(
            model="voyage-context-4",
            api_config={"api_key": "your-voyageai-api-key"}
        )
        context_embeddings = context_vectorizer.embed_many(
            contents=["chunk one", "chunk two", "chunk three"],
            input_type="document"
        )
        # Retrieval queries use input_type="query"; auto-chunking is a
        # document-only feature, so query inputs are embedded as-is.
        context_query = context_vectorizer.embed(
            content="your query text here",
            input_type="query"
        )

        # Multimodal usage - requires Pillow and voyageai>=0.3.6

        vectorizer = VoyageAIVectorizer(
            model="voyage-multimodal-3.5",
            api_config={"api_key": "your-voyageai-api-key"} # OR set VOYAGE_API_KEY in your env
        )
        image_embedding = vectorizer.embed_image(
            "path/to/your/image.jpg",
            input_type="query"
        )
        video_embedding = vectorizer.embed_video(
            "path/to/your/video.mp4",
            input_type="document"
        )

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="voyageai_embeddings_cache")

        vectorizer = VoyageAIVectorizer(
            model="voyage-3-large",
            api_config={"api_key": "your-voyageai-api-key"},
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed(
            content="your input query text here",
            input_type="query"
        )

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed(
            content="your input query text here",
            input_type="query"
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "voyage-3-large",
        api_config: dict[str, Any] | None = None,
        dtype: str = "float32",
        cache: "EmbeddingsCache | None" = None,
        **kwargs,
    ):
        """Initialize the VoyageAI vectorizer.

        Visit https://docs.voyageai.com/docs/embeddings to learn about embeddings and check the available models.

        Args:
            model (str): Model to use for embedding. Defaults to "voyage-3-large".
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.
            dtype (str): the default datatype to use when embedding content as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated items. Defaults to None.

        Raises:
            ImportError: If the voyageai library is not installed.
            ValueError: If the API key is not provided.

        Notes:
            - Multimodal models require voyageai>=0.3.6 to be installed for video embeddings, as well as
                ffmpeg installed on the system. Image embeddings require pillow to be installed.
            - Contextualized (``voyage-context-*``) models require voyageai>=0.5.0. Each input
                string is sent as its own document with auto-chunking, so inputs are embedded
                independently (no cross-input contextualization) and the one-embedding-per-input
                contract and cache determinism are preserved. A document longer than ``chunk_size``
                (32000 tokens) auto-chunks into multiple chunks but only the first chunk's embedding
                is kept; the rest are dropped. ``truncation`` is not forwarded to the contextualized
                API (it does not accept it), so it is silently ignored for these models.
            - The plain text embedding path (``embed``/``embed_many`` for non-context,
                non-multimodal models) uses token-aware batching: inputs are grouped into requests
                bounded by both the per-model item cap and the model's per-request token limit, so
                large inputs are packed efficiently without exceeding VoyageAI's token budget.

        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    @property
    def is_multimodal(self) -> bool:
        """Whether a multimodal model has been configured."""
        return "multimodal" in self.model

    @property
    def is_context(self) -> bool:
        """Whether a contextualized-embedding model (voyage-context-*) has been configured."""
        return "context" in self.model

    def embed_image(self, image_path: str, **kwargs) -> list[float] | bytes:
        """Embed an image (from its path on disk) using VoyageAI's multimodal API. Requires pillow to be installed."""
        if not self.is_multimodal:
            raise ValueError("Cannot embed image with a non-multimodal model.")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow library is required for image embedding. "
                "Please install with `pip install pillow`"
            )
        return self.embed(Image.open(image_path), **kwargs)

    def embed_video(self, video_path: str, **kwargs) -> list[float] | bytes:
        """Embed a video (from its path on disk) using VoyageAI's multimodal API.

        Requires voyageai>=0.3.6 to be installed, as well as ffmpeg to be installed on the system.
        """
        if not self.is_multimodal:
            raise ValueError("Cannot embed video with a non-multimodal model.")

        try:
            from voyageai.video_utils import Video
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "voyageai>=0.3.6 is required for video embedding. "
                "Please install with `pip install voyageai>=0.3.6`"
            )

        video = Video.from_path(
            video_path,
            model=self.model,
        )
        return self.embed(video, **kwargs)

    def _setup(self, api_config: dict[str, Any] | None, **kwargs):
        """Set up the VoyageAI client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)

        if self.is_multimodal:
            self._embed_fn = self._client.multimodal_embed
            self._aembed_fn = self._aclient.multimodal_embed
        else:
            self._embed_fn = self._client.embed  # type: ignore[assignment]
            self._aembed_fn = self._aclient.embed  # type: ignore[assignment]

        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: dict[str, Any] | None, **kwargs):
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
        Determine the per-request item cap for the current model.

        For the plain text path this is combined with the model's per-request
        token limit (see :meth:`_batchify_by_tokens`); it is the sole bound for
        the context/multimodal paths.

        Returns:
            int: Recommended maximum number of items per request
        """
        if self.model in ["voyage-2", "voyage-02"]:
            return 72
        elif self.model in ["voyage-3-lite", "voyage-3.5-lite", "voyage-4-lite"]:
            return 30
        elif self.model in ["voyage-3", "voyage-3.5", "voyage-4"]:
            return 10
        else:
            # Default for other models (e.g. voyage-3-large, voyage-4-large,
            # voyage-code-*, voyage-finance-2, voyage-law-2, voyage-context-*).
            return 7

    def _token_limit(self) -> int:
        """Per-request token budget for the current model (token-aware batching)."""
        return VOYAGE_TOKEN_LIMITS.get(self.model, DEFAULT_VOYAGE_TOKEN_LIMIT)

    def _count_tokens(self, text: str) -> int:
        """Count tokens for a single text using VoyageAI's tokenizer."""
        return len(self._client.tokenize([text], model=self.model)[0])

    def _batchify_by_tokens(self, texts: list[str], batch_size: int):
        """Yield batches of texts bounded by both item count and token budget.

        Batches are grown until adding the next text would exceed either the
        per-model item cap (``batch_size``) or the model's per-request token limit.
        A single text larger than the token limit is still yielded on its own
        (never dropped), matching VoyageAI's own client batching behavior.
        """
        max_tokens = self._token_limit()
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            n_tokens = self._count_tokens(text)
            if batch and (
                len(batch) >= batch_size or batch_tokens + n_tokens > max_tokens
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n_tokens
        if batch:
            yield batch

    def _validate_input(
        self, contents: list[Any], input_type: str | None, truncation: bool | None
    ):
        """
        Validate the inputs to the embedding methods.

        Args:
            contents: List of items to embed
            input_type: Type of input (document or query)
            truncation: Whether to truncate long inputs

        Raises:
            TypeError: If inputs are invalid
        """
        if not isinstance(contents, list):
            raise TypeError(
                "Must pass in a list of str, PIL.Image.Image, or voyageai.video_utils.Video values to embed.",
            )
        if not self.is_multimodal and contents and not isinstance(contents[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if input_type is not None and input_type not in ["document", "query"]:
            raise TypeError(
                "Must pass in a allowed value for voyageai embedding input_type. "
                "See https://docs.voyageai.com/docs/embeddings."
            )
        if truncation is not None and not isinstance(truncation, bool):
            raise TypeError("Truncation (optional) parameter is a bool.")

    def _embed(self, content: Any, **kwargs) -> list[float]:
        """
        Generate a vector embedding for a single item using the VoyageAI API.

        Args:
            content: Item to embed - must be one of str, PIL.Image.Image, or voyageai.video_utils.Video. Images and
                video require a multimodal model to be configured.
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If parameters are invalid
            ValueError: If embedding fails
        """
        # Simply call _embed_many with a single input and return the first result
        result = self._embed_many([content], **kwargs)
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, contents: list[Any], batch_size: int | None = None, **kwargs
    ) -> list[list[float]]:
        """
        Generate vector embeddings for a batch of items using the VoyageAI API.

        Args:
            contents: List of items to embed - each item must be one of str, PIL.Image.Image, or
                voyageai.video_utils.Video. Images and video require a multimodal model to be configured.
            batch_size: Number of items to process in each API call
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If `contents` is not a list, or parameters are invalid
            ValueError: If embedding fails
        """
        from voyageai.error import InvalidRequestError

        input_type = kwargs.pop("input_type", None)
        truncation = kwargs.pop("truncation", True)

        # Validate inputs
        self._validate_input(contents, input_type, truncation)

        # Determine batch size if not provided
        if batch_size is None:
            batch_size = self._get_batch_size()

        # The plain text path uses token-aware batching; context/multimodal paths
        # stay on fixed item-count batching (auto-chunking / opaque media inputs).
        if self.is_context or self.is_multimodal:
            batches: Any = self.batchify(contents, batch_size)
        else:
            batches = self._batchify_by_tokens(contents, batch_size)

        try:
            embeddings: list[Any] = []
            for batch in batches:
                if self.is_context:
                    embeddings.extend(
                        self._embed_context_batch(batch, input_type, **kwargs)
                    )
                    continue
                response = self._embed_fn(
                    (
                        # Multimodal wraps each item as its own single-part input
                        # so one embedding is returned per requested content.
                        [[item] for item in batch]
                        if self.is_multimodal
                        else batch
                    ),
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation,
                    **kwargs,  # type: ignore
                )
                embeddings.extend(response.embeddings)
            return embeddings
        except InvalidRequestError as e:
            raise TypeError(f"Invalid input for embedding: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def _context_embed_kwargs(
        self, batch: list[str], input_type: str | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the kwargs for a ``contextualized_embed`` call.

        The batch is passed as a flat ``list[str]``. Auto-chunking is only valid
        for ``input_type="document"`` (VoyageAI rejects it for queries), so every
        non-query input - including the default where ``input_type`` is omitted -
        is treated as a document and chunked with a large ``chunk_size``, making
        each input resolve to a single chunk. Explicit queries skip chunking and
        are sent as a flat ``list[str]`` as-is. Either way the first chunk per
        input is kept, yielding one embedding per requested item.
        """
        enable_auto_chunking = input_type != "query"
        # Auto-chunking requires input_type="document"; a flat list[str] with no
        # type is rejected, so default (None) callers - e.g. SemanticRouter and
        # SemanticCache, which never set input_type - resolve to "document".
        effective_input_type = "document" if enable_auto_chunking else input_type
        call_kwargs: dict[str, Any] = {
            "inputs": batch,
            "model": self.model,
            "input_type": effective_input_type,
            "enable_auto_chunking": enable_auto_chunking,
            **kwargs,
        }
        if enable_auto_chunking:
            call_kwargs["chunk_size"] = 32000
        return call_kwargs

    def _embed_context_batch(
        self, batch: list[str], input_type: str | None, **kwargs
    ) -> list[list[float]]:
        """Embed a batch with a contextualized (voyage-context-*) model."""
        # contextualized_embed is only present on recent voyageai clients; the
        # attr-defined ignore keeps mypy happy without vendored stubs.
        response = self._client.contextualized_embed(  # type: ignore[attr-defined]
            **self._context_embed_kwargs(batch, input_type, kwargs),
        )
        # Take the first chunk per input to keep one embedding per input.
        return cast(
            "list[list[float]]",
            [result.embeddings[0] for result in response.results],
        )

    async def _aembed_context_batch(
        self, batch: list[str], input_type: str | None, **kwargs
    ) -> list[list[float]]:
        """Asynchronously embed a batch with a contextualized model.

        See :meth:`_embed_context_batch` for details on the input format.
        """
        response = await self._aclient.contextualized_embed(  # type: ignore[attr-defined]
            **self._context_embed_kwargs(batch, input_type, kwargs),
        )
        # Take the first chunk per input to keep one embedding per input.
        return cast(
            "list[list[float]]",
            [result.embeddings[0] for result in response.results],
        )

    async def _aembed(self, content: Any, **kwargs) -> list[float]:
        """
        Asynchronously generate a vector embedding for a single item using the VoyageAI API.

        Args:
            content: Item to embed - must be one of str, PIL.Image.Image, or voyageai.video_utils.Video. Images and
                video require a multimodal model to be configured.
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If parameters are invalid
            ValueError: If embedding fails
        """
        # Simply call _aembed_many with a single item and return the first result
        result = await self._aembed_many([content], **kwargs)
        return result[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    async def _aembed_many(
        self, contents: list[Any], batch_size: int | None = None, **kwargs
    ) -> list[list[float]]:
        """
        Asynchronously generate vector embeddings for a batch of items using the VoyageAI API.

        Args:
            contents: List of items to embed - each item must be one of str, PIL.Image.Image, or
                voyageai.video_utils.Video. Images and video require a multimodal model to be configured.
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the VoyageAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If `contents` is not a list, or parameters are invalid
            ValueError: If embedding fails
        """
        from voyageai.error import InvalidRequestError

        input_type = kwargs.pop("input_type", None)
        truncation = kwargs.pop("truncation", True)

        # Validate inputs
        self._validate_input(contents, input_type, truncation)

        # Determine batch size if not provided
        if batch_size is None:
            batch_size = self._get_batch_size()

        # The plain text path uses token-aware batching; context/multimodal paths
        # stay on fixed item-count batching (auto-chunking / opaque media inputs).
        if self.is_context or self.is_multimodal:
            batches: Any = self.batchify(contents, batch_size)
        else:
            batches = self._batchify_by_tokens(contents, batch_size)

        try:
            embeddings: list[Any] = []
            for batch in batches:
                if self.is_context:
                    embeddings.extend(
                        await self._aembed_context_batch(batch, input_type, **kwargs)
                    )
                    continue
                response = await self._aembed_fn(
                    (
                        # Multimodal wraps each item as its own single-part input
                        # so one embedding is returned per requested content.
                        [[item] for item in batch]
                        if self.is_multimodal
                        else batch
                    ),
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation,
                    **kwargs,  # type: ignore
                )
                embeddings.extend(response.embeddings)
            return embeddings
        except InvalidRequestError as e:
            raise TypeError(f"Invalid input for embedding: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def _serialize_for_cache(self, content: Any) -> bytes | str:
        """Convert content to a cacheable format."""
        try:
            from voyageai.video_utils import Video
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "voyageai>=0.3.6 is required for video embedding. "
                "Please install with `pip install voyageai>=0.3.6`"
            )

        if isinstance(content, Video):
            return content.to_bytes()
        return super()._serialize_for_cache(content)

    @property
    def type(self) -> str:
        return "voyageai"

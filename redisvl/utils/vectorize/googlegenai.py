import os
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from redisvl.utils.vectorize.base import BaseVectorizer

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache


class GoogleGenAIVectorizer(BaseVectorizer):
    """The GoogleGenAIVectorizer creates embeddings with Google's `google-genai`
    SDK, the supported replacement for the deprecated Vertex AI model-garden SDK
    used by :class:`VertexAIVectorizer`.

    One client, two backends. `google-genai` reaches both of Google's embedding
    backends; this vectorizer auto-selects one from your config/environment (or an
    explicit override):

    - **Vertex AI / Gemini Enterprise** (GCP project auth) — for existing Vertex users.
    - **Gemini Developer API** (a single API key) — the simplest way to get started.

    Credentials are resolved in this order (explicit beats ambient):

    1. Explicit override — ``api_config={"backend": "vertex" | "gemini"}``.
    2. Explicit creds in ``api_config`` — ``api_key`` selects Gemini; ``project_id``
       (+ ``location``) selects Vertex.
    3. Environment — a project (``GOOGLE_CLOUD_PROJECT`` | ``GCP_PROJECT_ID`` with
       ``GOOGLE_CLOUD_LOCATION`` | ``GCP_LOCATION`` and Application Default
       Credentials via ``GOOGLE_APPLICATION_CREDENTIALS``) selects Vertex; otherwise
       ``GEMINI_API_KEY`` | ``GOOGLE_API_KEY`` selects Gemini. If both a project and a
       key are present, Vertex wins.

    Install the client with ``pip install redisvl[google-genai]``.

    .. note::
        The default model ``gemini-embedding-001`` returns **3072-dimensional**
        vectors (≈4× the width of the legacy ``textembedding-gecko``, so ≈4× the
        index memory). A reduced ``output_dimensionality`` returns shorter vectors that
        Google does **not** re-normalize; this is invisible under the COSINE metric
        (scale-invariant) but matters for inner-product / L2 — normalize yourself if your
        index needs it. Embeddings from different models (or dimensions) are not
        interchangeable — reindex when you change them.

    .. code-block:: python

        # Vertex AI backend
        from redisvl.utils.vectorize import GoogleGenAIVectorizer

        vectorizer = GoogleGenAIVectorizer(
            model="gemini-embedding-001",
            api_config={
                "project_id": "your-gcp-project",  # or set GOOGLE_CLOUD_PROJECT
                "location": "us-central1",         # or set GOOGLE_CLOUD_LOCATION
            },
        )
        embedding = vectorizer.embed("Hello, world!")

        # Gemini Developer API backend
        vectorizer = GoogleGenAIVectorizer(
            model="gemini-embedding-001",
            api_config={"api_key": "your-gemini-api-key"},  # or set GEMINI_API_KEY
        )

        # Reduced dimensions (Matryoshka) + caching
        from redisvl.extensions.cache.embeddings import EmbeddingsCache

        vectorizer = GoogleGenAIVectorizer(
            model="gemini-embedding-001",
            output_dimensionality=768,
            cache=EmbeddingsCache(name="genai_embeddings_cache"),
        )

        # Asynchronous batch embedding
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"], batch_size=2
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_config: dict[str, Any] | None = None,
        dtype: str = "float32",
        cache: "EmbeddingsCache | None" = None,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
        **kwargs,
    ):
        """Initialize the Google GenAI vectorizer.

        Args:
            model (str): The embedding model to use. Defaults to
                'gemini-embedding-001', which works on both backends.
            api_config (Optional[Dict]): Auth/client configuration. Recognized keys:
                ``backend`` ("vertex"|"gemini"), ``project_id``, ``location``,
                ``credentials`` (Vertex), and ``api_key`` (Gemini). Model-behavior
                options are the ``task_type``/``output_dimensionality`` arguments
                below, not ``api_config`` keys. Defaults to None.
            dtype (str): The default datatype to use when embedding text as byte
                arrays. Used when setting ``as_buffer=True``. Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional cache for repeated inputs.
            task_type (Optional[str]): Default embedding task type (e.g.
                'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'). Overridable per call.
            output_dimensionality (Optional[int]): Request shorter (Matryoshka)
                embeddings of this width; sets ``dims`` accordingly. Fixed for the
                vectorizer's lifetime (cannot be overridden per call). Note Google does
                not re-normalize reduced vectors.
            **kwargs: Additional arguments forwarded to ``google.genai.Client``
                (e.g. ``http_options``).

        Raises:
            ImportError: If the google-genai library is not installed.
            ValueError: If a backend cannot be resolved, or an invalid dtype is given.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        self._setup(api_config, task_type, output_dimensionality, **kwargs)

    def _setup(
        self,
        api_config: dict[str, Any] | None,
        task_type: str | None,
        output_dimensionality: int | None,
        **kwargs,
    ):
        """Build the default embed config, initialize the client, and set dims."""
        # Default embed config is set BEFORE _set_model_dims so the dimension-probe
        # embed uses the same config (e.g. output_dimensionality) as real calls.
        self._embed_config: dict[str, Any] = {}
        if task_type is not None:
            self._embed_config["task_type"] = task_type
        if output_dimensionality is not None:
            self._embed_config["output_dimensionality"] = output_dimensionality

        self._initialize_client(api_config, **kwargs)
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: dict[str, Any] | None, **kwargs):
        """Resolve the backend and construct a single google-genai client.

        Raises:
            ImportError: If the google-genai library is not installed.
            ValueError: If no backend can be resolved from config or environment.
        """
        # Copy so we never mutate the caller's dict.
        api_config = dict(api_config or {})

        try:
            from google import genai
            from google.genai.types import EmbedContentConfig
        except ImportError:
            raise ImportError(
                "GoogleGenAIVectorizer requires the google-genai library. "
                "Please install with `pip install redisvl[google-genai]`"
            )

        self._config_cls = EmbedContentConfig
        backend = self._resolve_backend(api_config)

        if backend == "vertex":
            project = (
                api_config.get("project_id")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCP_PROJECT_ID")
            )
            location = (
                api_config.get("location")
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or os.getenv("GCP_LOCATION")
            )
            if not location:
                raise ValueError(
                    "Vertex AI backend selected but no location was provided. "
                    "Set api_config['location'] or the GOOGLE_CLOUD_LOCATION "
                    "(or GCP_LOCATION) environment variable."
                )
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=api_config.get("credentials"),
                **kwargs,
            )
        else:
            api_key = (
                api_config.get("api_key")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            self._client = genai.Client(api_key=api_key, **kwargs)

        self._backend = backend

    @staticmethod
    def _resolve_backend(api_config: dict[str, Any]) -> str:
        """Return 'vertex' or 'gemini' using explicit-beats-ambient precedence."""
        # 1. Explicit override.
        explicit = api_config.get("backend")
        if explicit is not None:
            if explicit not in ("vertex", "gemini"):
                raise ValueError(
                    f"Invalid api_config['backend']: {explicit!r}. "
                    "Must be 'vertex' or 'gemini'."
                )
            return explicit
        if "vertexai" in api_config:
            return "vertex" if api_config["vertexai"] else "gemini"

        # 2. Explicit credentials in api_config.
        if api_config.get("api_key"):
            return "gemini"
        if api_config.get("project_id"):
            return "vertex"

        # 3. Ambient environment (project beats key).
        if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID"):
            return "vertex"
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"

        raise ValueError(
            "Could not resolve a Google backend. Provide Vertex AI config "
            "(api_config={'project_id': ..., 'location': ...} or set "
            "GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION [or GCP_PROJECT_ID/GCP_LOCATION] "
            "with Application Default Credentials), or a Gemini Developer API key "
            "(api_config={'api_key': ...} or set GEMINI_API_KEY/GOOGLE_API_KEY)."
        )

    @property
    def backend(self) -> str:
        """The resolved backend: 'vertex' or 'gemini'."""
        return self._backend

    def _set_model_dims(self) -> int:
        """Determine embedding dimensionality via a single probe embed."""
        try:
            embedding = self._embed("dimension check")
            return len(embedding)
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Unexpected response from the Google GenAI API: {e}")
        except Exception as e:  # pylint: disable=broad-except
            # Avoid interpolating provider/transport errors, which can embed
            # credentials, into the message. Chain the original for debugging.
            raise ValueError(
                "Error setting embedding model dimensions with the Google GenAI API. "
                "Verify the model name, backend selection, and credentials."
            ) from e

    def _build_config(self, extra: dict[str, Any]) -> Any:
        """Merge stored config defaults with per-call kwargs into an EmbedContentConfig.

        ``output_dimensionality`` is fixed at construction (it determines ``self.dims``),
        so it cannot be overridden per call — that would desync returned embeddings from
        the index's vector width. Other fields (e.g. ``task_type``) may vary per call.
        """
        if "output_dimensionality" in extra:
            raise TypeError(
                "output_dimensionality cannot be overridden per call; set it on the "
                "GoogleGenAIVectorizer(...) constructor instead (it determines dims)."
            )
        merged = {**self._embed_config, **extra}
        return self._config_cls(**merged) if merged else None

    @staticmethod
    def _embeddings_from_response(response: Any, expected: int) -> list:
        """Return the response embeddings, guarding against None / count mismatch."""
        embeddings = response.embeddings
        if not embeddings or len(embeddings) != expected:
            count = 0 if not embeddings else len(embeddings)
            raise ValueError(
                f"Google GenAI API returned {count} embedding(s) for {expected} "
                "input(s)."
            )
        return embeddings

    @staticmethod
    def _values(embedding: Any) -> list[float]:
        """Return an embedding's raw values, guarding against a null payload."""
        if embedding.values is None:
            raise ValueError("No embedding returned from the Google GenAI API.")
        return list(embedding.values)

    @staticmethod
    def _validate_many(contents: list[str]) -> None:
        """Validate batch input without masking the validation error."""
        if not isinstance(contents, list):
            raise TypeError(
                f"Input contents must be a list of strings to embed, got {type(contents)}"
            )
        if not all(isinstance(c, str) for c in contents):
            raise TypeError("Input contents must be a list of strings to embed.")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
        reraise=True,
    )
    def _embed(self, content: str, **kwargs) -> list[float]:
        """Generate a vector embedding for a single string."""
        if not isinstance(content, str):
            raise TypeError(
                f"Input content must be a string to embed, got {type(content)}"
            )
        config = self._build_config(kwargs)
        response = self._client.models.embed_content(
            model=self.model, contents=content, config=config
        )
        embeddings = self._embeddings_from_response(response, 1)
        return self._values(embeddings[0])

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
        reraise=True,
    )
    def _embed_many(
        self, contents: list[str], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        """Generate vector embeddings for a batch of strings."""
        self._validate_many(contents)
        config = self._build_config(kwargs)
        embeddings: list[list[float]] = []
        for batch in self.batchify(contents, batch_size):
            response = self._client.models.embed_content(
                model=self.model, contents=batch, config=config
            )
            batch_embeddings = self._embeddings_from_response(response, len(batch))
            embeddings.extend(self._values(e) for e in batch_embeddings)
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
        reraise=True,
    )
    async def _aembed(self, content: str, **kwargs) -> list[float]:
        """Asynchronously generate a vector embedding for a single string."""
        if not isinstance(content, str):
            raise TypeError(
                f"Input content must be a string to embed, got {type(content)}"
            )
        config = self._build_config(kwargs)
        response = await self._client.aio.models.embed_content(
            model=self.model, contents=content, config=config
        )
        embeddings = self._embeddings_from_response(response, 1)
        return self._values(embeddings[0])

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
        reraise=True,
    )
    async def _aembed_many(
        self, contents: list[str], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        """Asynchronously generate vector embeddings for a batch of strings."""
        self._validate_many(contents)
        config = self._build_config(kwargs)
        embeddings: list[list[float]] = []
        for batch in self.batchify(contents, batch_size):
            response = await self._client.aio.models.embed_content(
                model=self.model, contents=batch, config=config
            )
            batch_embeddings = self._embeddings_from_response(response, len(batch))
            embeddings.extend(self._values(e) for e in batch_embeddings)
        return embeddings

    @property
    def type(self) -> str:
        return "google_genai"

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer


class VertexAITextVectorizer(BaseVectorizer):
    """The VertexAITextVectorizer uses Google's VertexAI Palm 2 embedding model
    API to create text embeddings.

    This vectorizer is tailored for use in
    environments where integration with Google Cloud Platform (GCP) services is
    a key requirement.

    Utilizing this vectorizer requires an active GCP project and location
    (region), along with appropriate application credentials. These can be
    provided through the `api_config` dictionary or set the GOOGLE_APPLICATION_CREDENTIALS
    env var. Additionally, the vertexai python client must be
    installed with `pip install google-cloud-aiplatform>=1.26`.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        # Basic usage
        vectorizer = VertexAITextVectorizer(
            model="textembedding-gecko",
            api_config={
                "project_id": "your_gcp_project_id", # OR set GCP_PROJECT_ID
                "location": "your_gcp_location",     # OR set GCP_LOCATION
            })
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="vertexai_embeddings_cache")

        vectorizer = VertexAITextVectorizer(
            model="textembedding-gecko",
            api_config={
                "project_id": "your_gcp_project_id",
                "location": "your_gcp_location",
            },
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Batch embedding of multiple texts
        embeddings = vectorizer.embed_many(
            ["Hello, world!", "Goodbye, world!"],
            batch_size=2
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "textembedding-gecko",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the VertexAI vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to
                'textembedding-gecko'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API config details. Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the google-cloud-aiplatform library is not installed.
            ValueError: If the API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the VertexAI client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the VertexAI client using the provided config options or
        environment variables.

        Args:
            api_config: Dictionary with GCP configuration options
            **kwargs: Additional arguments for initialization

        Raises:
            ImportError: If the google-cloud-aiplatform library is not installed
            ValueError: If required parameters are not provided
        """
        # Fetch the project_id and location from api_config or environment variables
        project_id = (
            api_config.get("project_id") if api_config else os.getenv("GCP_PROJECT_ID")
        )
        location = (
            api_config.get("location") if api_config else os.getenv("GCP_LOCATION")
        )

        if not project_id:
            raise ValueError(
                "Missing project_id. "
                "Provide the id in the api_config with key 'project_id' "
                "or set the GCP_PROJECT_ID environment variable."
            )

        if not location:
            raise ValueError(
                "Missing location. "
                "Provide the location (region) in the api_config with key 'location' "
                "or set the GCP_LOCATION environment variable."
            )

        # Check for credentials
        credentials = api_config.get("credentials") if api_config else None

        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(
                project=project_id, location=location, credentials=credentials
            )
        except ImportError:
            raise ImportError(
                "VertexAI vectorizer requires the google-cloud-aiplatform library. "
                "Please install with `pip install google-cloud-aiplatform>=1.26`"
            )

        # Store client as a regular attribute instead of PrivateAttr
        self._client = TextEmbeddingModel.from_pretrained(self.model)

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
            raise ValueError(f"Unexpected response from the VertexAI API: {str(ke)}")
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
        Generate a vector embedding for a single text using the VertexAI API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the VertexAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If text is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        try:
            result = self._client.get_embeddings([text], **kwargs)
            return result[0].values
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
        Generate vector embeddings for a batch of texts using the VertexAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the VertexAI API

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
                response = self._client.get_embeddings(batch, **kwargs)
                embeddings.extend([r.values for r in response])
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    @property
    def type(self) -> str:
        return "vertexai"

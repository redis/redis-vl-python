import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.utils import lazy_import

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.vectorize.base import BaseVectorizer

InvalidArgument = lazy_import("google.api_core.exceptions.InvalidArgument")


class VertexAIVectorizer(BaseVectorizer):
    """The VertexAIVectorizer uses Google's VertexAI embedding model
    API to create embeddings.

    This vectorizer is tailored for use in
    environments where integration with Google Cloud Platform (GCP) services is
    a key requirement.

    Utilizing this vectorizer requires an active GCP project and location
    (region), along with appropriate application credentials. These can be
    provided through the `api_config` dictionary or set the GOOGLE_APPLICATION_CREDENTIALS
    env var. Additionally, the vertexai python client must be
    installed with `pip install google-cloud-aiplatform>=1.26`.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated inputs.

    .. code-block:: python

        # Basic usage
        vectorizer = VertexAIVectorizer(
            model="textembedding-gecko",
            api_config={
                "project_id": "your_gcp_project_id", # OR set GCP_PROJECT_ID
                "location": "your_gcp_location",     # OR set GCP_LOCATION
            })
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="vertexai_embeddings_cache")

        vectorizer = VertexAIVectorizer(
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

        # Multimodal usage
        from vertexai.vision_models import Image, Video

        vectorizer = VertexAIVectorizer(
            model="multimodalembedding@001",
            api_config={
                "project_id": "your_gcp_project_id", # OR set GCP_PROJECT_ID
                "location": "your_gcp_location",     # OR set GCP_LOCATION
            }
        )
        text_embedding = vectorizer.embed("Hello, world!")
        image_embedding = vectorizer.embed(Image.load_from_file("path/to/your/image.jpg"))
        video_embedding = vectorizer.embed(Video.load_from_file("path/to/your/video.mp4"))

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

    @property
    def is_multimodal(self) -> bool:
        """Whether a multimodal model has been configured."""
        return "multimodal" in self.model

    @cached_property
    def _client(self):
        """Get the appropriate client based on the model type."""
        if self.is_multimodal:
            from vertexai.vision_models import MultiModalEmbeddingModel

            return MultiModalEmbeddingModel.from_pretrained(self.model)

        from vertexai.language_models import TextEmbeddingModel

        return TextEmbeddingModel.from_pretrained(self.model)

    def embed_image(self, image_path: str, **kwargs) -> Union[List[float], bytes]:
        """Embed an image (from its path on disk) using a VertexAI multimodal model."""
        if not self.is_multimodal:
            raise ValueError("Cannot embed image with a non-multimodal model.")

        from vertexai.vision_models import Image

        return self.embed(Image.load_from_file(image_path), **kwargs)

    def embed_video(self, video_path: str, **kwargs) -> Union[List[float], bytes]:
        """Embed a video (from its path on disk) using a VertexAI multimodal model."""
        if not self.is_multimodal:
            raise ValueError("Cannot embed video with a non-multimodal model.")

        from vertexai.vision_models import Video

        return self.embed(Video.load_from_file(video_path), **kwargs)

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

            vertexai.init(
                project=project_id, location=location, credentials=credentials
            )

        except ImportError:
            raise ImportError(
                "VertexAI vectorizer requires the google-cloud-aiplatform library. "
                "Please install with `pip install google-cloud-aiplatform>=1.26`"
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
            raise ValueError(f"Unexpected response from the VertexAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed(self, content: Any, **kwargs) -> List[float]:
        """
        Generate a vector embedding for a single input using the VertexAI API.

        Args:
            content: Input to embed
            **kwargs: Additional parameters to pass to the VertexAI API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            ValueError: If embedding fails
        """
        try:
            if self.is_multimodal:
                from vertexai.vision_models import Image, Video

                if isinstance(content, str):
                    result = self._client.get_embeddings(
                        contextual_text=content,
                        **kwargs,
                    )
                    if result.text_embedding is None:
                        raise ValueError("No text embedding returned from VertexAI.")
                    return result.text_embedding
                elif isinstance(content, Image):
                    result = self._client.get_embeddings(
                        image=content,
                        **kwargs,
                    )
                    if result.image_embedding is None:
                        raise ValueError("No image embedding returned from VertexAI.")
                    return result.image_embedding
                elif isinstance(content, Video):
                    result = self._client.get_embeddings(
                        video=content,
                        **kwargs,
                    )
                    if result.video_embeddings is None:
                        raise ValueError("No video embedding returned from VertexAI.")
                    return result.video_embeddings[0].embedding
                else:
                    raise TypeError(
                        "Invalid input type for multimodal embedding. "
                        "Must be str, Image, or Video."
                    )

            else:
                return self._client.get_embeddings([content], **kwargs)[0].values

        except InvalidArgument as e:
            raise TypeError(f"Invalid input for embedding: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Embedding input failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, contents: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of texts using the VertexAI API.

        Args:
            contents: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the VertexAI API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If contents is not a list of strings
            ValueError: If embedding fails
        """
        if self.is_multimodal:
            raise NotImplementedError(
                "Batch embedding is not supported for multimodal models with VertexAI."
            )
        if not isinstance(contents, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if contents and not isinstance(contents[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        try:
            embeddings: List = []
            for batch in self.batchify(contents, batch_size):
                response = self._client.get_embeddings(batch, **kwargs)
                embeddings.extend([r.values for r in response])
            return embeddings
        except InvalidArgument as e:
            raise TypeError(f"Invalid input for embedding: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def _serialize_for_cache(self, content: Any) -> Union[bytes, str]:
        """Convert content to a cacheable format."""
        from vertexai.vision_models import Image, Video

        if isinstance(content, Image):
            return content._image_bytes
        elif isinstance(content, Video):
            return content._video_bytes
        return super()._serialize_for_cache(content)

    @property
    def type(self) -> str:
        return "vertexai"

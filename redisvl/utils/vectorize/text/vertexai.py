import os
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

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

    .. code-block:: python

        # Synchronous embedding of a single text
        vectorizer = VertexAITextVectorizer(
            model="textembedding-gecko",
            api_config={
                "project_id": "your_gcp_project_id", # OR set GCP_PROJECT_ID
                "location": "your_gcp_location",     # OR set GCP_LOCATION
            })
        embedding = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.embed_many(
            ["Hello, world!", "Goodbye, world!"],
            batch_size=2
        )

    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "textembedding-gecko",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
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

        Raises:
            ImportError: If the google-cloud-aiplatform library is not installed.
            ValueError: If the API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype)
        # Init client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after init
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the VertexAI clients using the provided API key or an
        environment variable.
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

        self._client = TextEmbeddingModel.from_pretrained(self.model)

    def _set_model_dims(self) -> int:
        try:
            embedding = self.embed("dimension check")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the VertexAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Embed many chunks of text using the VertexAI Embeddings API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            Union[List[List[float]], List[bytes]]: List of embeddings as lists of floats,
            or as bytes objects if as_buffer=True

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._client.get_embeddings(batch, **kwargs)
            embeddings += [
                self._process_embedding(r.values, as_buffer, dtype) for r in response
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Embed a chunk of text using the VertexAI Embeddings API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            Union[List[float], bytes]: Embedding as a list of floats, or as a bytes
            object if as_buffer=True

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = self._client.get_embeddings([text], **kwargs)
        return self._process_embedding(result[0].values, as_buffer, dtype)

    @property
    def type(self) -> str:
        return "vertexai"

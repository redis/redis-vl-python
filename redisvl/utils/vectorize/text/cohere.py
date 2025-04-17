import os
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that cohere isn't imported
# mypy: disable-error-code="name-defined"


class CohereTextVectorizer(BaseVectorizer):
    """The CohereTextVectorizer class utilizes Cohere's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Cohere's /embed API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `COHERE_API_KEY`
    environment variable. User must obtain an API key from Cohere's website
    (https://dashboard.cohere.com/). Additionally, the `cohere` python
    client must be installed with `pip install cohere`.

    The vectorizer supports only synchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    .. code-block:: python

        from redisvl.utils.vectorize import CohereTextVectorizer

        # Basic usage
        vectorizer = CohereTextVectorizer(
            model="embed-english-v3.0",
            api_config={"api_key": "your-cohere-api-key"} # OR set COHERE_API_KEY in your env
        )
        query_embedding = vectorizer.embed(
            text="your input query text here",
            input_type="search_query"
        )
        doc_embeddings = vectorizer.embed_many(
            texts=["your document text", "more document text"],
            input_type="search_document"
        )

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="cohere_embeddings_cache")

        vectorizer = CohereTextVectorizer(
            model="embed-english-v3.0",
            api_config={"api_key": "your-cohere-api-key"},
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed(
            text="your input query text here",
            input_type="search_query"
        )

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed(
            text="your input query text here",
            input_type="search_query"
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the Cohere vectorizer.

        Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            model (str): Model to use for embedding. Defaults to 'embed-english-v3.0'.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                'float32' will use Cohere's float embeddings, 'int8' and 'uint8' will map
                to Cohere's corresponding embedding types. Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ImportError: If the cohere library is not installed.
            ValueError: If the API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the Cohere client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the Cohere client using the provided API key or an
        environment variable.

        Args:
            api_config: Dictionary with API configuration options
            **kwargs: Additional arguments to pass to Cohere client

        Raises:
            ImportError: If the cohere library is not installed
            ValueError: If no API key is provided
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the cohere module
        try:
            from cohere import Client
        except ImportError:
            raise ImportError(
                "Cohere vectorizer requires the cohere library. "
                "Please install with `pip install cohere`"
            )

        api_key = (
            api_config.get("api_key") if api_config else os.getenv("COHERE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Cohere API key is required. "
                "Provide it in api_config or set the COHERE_API_KEY environment variable."
            )

        self._client = Client(api_key=api_key, client_name="redisvl", **kwargs)

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
            embedding = self._embed("dimension check", input_type="search_document")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the Cohere API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    def _get_cohere_embedding_type(self, dtype: str) -> List[str]:
        """
        Map dtype to appropriate Cohere embedding_types value.

        Args:
            dtype: The data type to map to Cohere embedding types

        Returns:
            List of embedding type strings compatible with Cohere API
        """
        if dtype == "int8":
            return ["int8"]
        elif dtype == "uint8":
            return ["uint8"]
        else:
            return ["float"]

    def _validate_input_type(self, input_type) -> None:
        """
        Validate that a proper input_type parameter was provided.

        Args:
            input_type: The input type parameter to validate

        Raises:
            TypeError: If input_type is not a string
        """
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for cohere embedding input_type. "
                "See https://docs.cohere.com/reference/embed."
            )

    def _embed(self, text: str, **kwargs) -> List[Union[float, int]]:
        """
        Generate a vector embedding for a single text using the Cohere API.

        Args:
            text: Text to embed
            **kwargs: Additional parameters to pass to the Cohere API,
                      must include 'input_type'

        Returns:
            Union[List[float], List[int], bytes]:
            - If as_buffer=True: Returns a bytes object
            - If as_buffer=False:
              - For dtype="float32": Returns a list of floats
              - For dtype="int8" or "uint8": Returns a list of integers

        Raises:
            TypeError: If text is not a string or input_type is not provided
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        input_type = kwargs.pop("input_type", None)
        self._validate_input_type(input_type)

        # Check if embedding_types was provided and warn user
        if "embedding_types" in kwargs:
            warnings.warn(
                "The 'embedding_types' parameter is not supported in CohereTextVectorizer. "
                "Please use the 'dtype' parameter instead. Your 'embedding_types' value will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop("embedding_types")

        # Map dtype to appropriate embedding_type
        embedding_types = self._get_cohere_embedding_type(self.dtype)

        try:
            response = self._client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                embedding_types=embedding_types,
                **kwargs,
            )

            # Extract the appropriate embedding based on embedding_types
            embed_type = embedding_types[0]
            if hasattr(response.embeddings, embed_type):
                embedding = getattr(response.embeddings, embed_type)[0]
            else:
                embedding = response.embeddings[0]  # type: ignore

            return embedding
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[Union[float, int]]]:
        """
        Generate vector embeddings for a batch of texts using the Cohere API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            **kwargs: Additional parameters to pass to the Cohere API,
                      must include 'input_type'

        Returns:
            List[List[Union[float, int]]]: List of vector embeddings

        Raises:
            TypeError: If texts is not a list of strings or input_type is not provided
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        input_type = kwargs.pop("input_type", None)
        self._validate_input_type(input_type)

        # Check if embedding_types was provided and warn user
        if "embedding_types" in kwargs:
            warnings.warn(
                "The 'embedding_types' parameter is not supported in CohereTextVectorizer. "
                "Please use the 'dtype' parameter instead. Your 'embedding_types' value will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop("embedding_types")

        # Map dtype to appropriate embedding_type
        embedding_types = self._get_cohere_embedding_type(self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size):
            try:
                response = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    embedding_types=embedding_types,
                    **kwargs,
                )

                # Extract the appropriate embeddings based on embedding_types
                embed_type = embedding_types[0]
                if hasattr(response.embeddings, embed_type):
                    batch_embeddings = getattr(response.embeddings, embed_type)
                else:
                    # Fallback for older API versions
                    batch_embeddings = response.embeddings

                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise ValueError(f"Embedding texts failed: {e}")

        return embeddings

    @property
    def type(self) -> str:
        return "cohere"

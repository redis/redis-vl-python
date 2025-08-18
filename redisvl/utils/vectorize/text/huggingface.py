from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from pydantic.v1 import PrivateAttr

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer


class HFTextVectorizer(BaseVectorizer):
    """The HFTextVectorizer class leverages Hugging Face's Sentence Transformers
    for generating vector embeddings from text input.

    This vectorizer is particularly useful in scenarios where advanced natural language
    processing and understanding are required, and ideal for running on your own
    hardware without usage fees.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated text inputs.

    Utilizing this vectorizer involves specifying a pre-trained model from
    Hugging Face's vast collection of Sentence Transformers. These models are
    trained on a variety of datasets and tasks, ensuring versatility and
    robust performance across different embedding needs.

    Requirements:
        - The `sentence-transformers` library must be installed with pip.

    .. code-block:: python

        # Basic usage
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
        embedding = vectorizer.embed("Hello, world!")

        # With caching enabled
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="my_embeddings_cache")

        vectorizer = HFTextVectorizer(
            model="sentence-transformers/all-mpnet-base-v2",
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Batch processing
        embeddings = vectorizer.embed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )
    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the Hugging Face text vectorizer.

        Args:
            model (str): The pre-trained model from Hugging Face's Sentence
                Transformers to be used for embedding. Defaults to
                'sentence-transformers/all-mpnet-base-v2'.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.
            **kwargs: Additional parameters to pass to the SentenceTransformer
                constructor.

        Raises:
            ImportError: If the sentence-transformers library is not installed.
            ValueError: If there is an error setting the embedding model dimensions.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Init client
        self._initialize_client(model, **kwargs)
        # Set model dimensions after init
        self.dims = self._set_model_dims()

    def _initialize_client(self, model: str, **kwargs):
        """Setup the HuggingFace client"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "HFTextVectorizer requires the sentence-transformers library. "
                "Please install with `pip install sentence-transformers`"
            )

        self._client = SentenceTransformer(model, **kwargs)

    def _set_model_dims(self):
        try:
            embedding = self._embed("dimension check")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Empty response from the embedding model: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    def _embed(self, text: str, **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the Hugging Face model.

        Args:
            text: Text to embed
            **kwargs: Additional model-specific parameters

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If the input is not a string
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if "show_progress_bar" not in kwargs:
            # disable annoying tqdm by default
            kwargs["show_progress_bar"] = False

        embedding = self._client.encode([text], **kwargs)[0]
        return embedding.tolist()

    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the Hugging Face model.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            **kwargs: Additional model-specific parameters

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If the input is not a list of strings
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if "show_progress_bar" not in kwargs:
            # disable annoying tqdm by default
            kwargs["show_progress_bar"] = False

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, None):
            batch_embeddings = self._client.encode(batch, **kwargs)
            embeddings.extend([embedding.tolist() for embedding in batch_embeddings])
        return embeddings

    @property
    def type(self) -> str:
        return "hf"

from typing import TYPE_CHECKING, Any, List, Optional

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

    Note:
        Some multimodal models can make use of sentence-transformers by passing
        PIL Image objects in place of strings (e.g. CLIP). To enable those use
        cases, this class follows the SentenceTransformer convention of hinting
        that it expects string inputs, but never enforcing it.

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

        # Multimodal usage
        from PIL import Image
        vectorizer = HFTextVectorizer(model="sentence-transformers/clip-ViT-L-14")
        embeddings1 = vectorizer.embed("Hello, world!")
        embeddings2 = vectorizer.embed(Image.open("path/to/your/image.jpg"))

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

    @deprecated_argument("text", "content")
    def _embed(self, content: str = "", text: str = "", **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the Hugging Face model.

        Args:
            content: Text to embed
            text: Text to embed (deprecated - use `content` instead)
            **kwargs: Additional model-specific parameters

        Returns:
            List[float]: Vector embedding as a list of floats
        """
        content = content or text
        if "show_progress_bar" not in kwargs:
            # disable annoying tqdm by default
            kwargs["show_progress_bar"] = False

        embedding = self._client.encode([content], **kwargs)[0]
        return embedding.tolist()

    @deprecated_argument("texts", "contents")
    def _embed_many(
        self,
        contents: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 10,
        **kwargs,
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the Hugging Face model.

        Args:
            contents: List of texts to embed
            texts: List of texts to embed (deprecated - use `contents` instead)
            batch_size: Number of texts to process in each batch
            **kwargs: Additional model-specific parameters

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats
        """
        contents = contents or texts
        if not isinstance(contents, list):
            raise TypeError("Must pass in a list of values to embed.")
        if "show_progress_bar" not in kwargs:
            # disable annoying tqdm by default
            kwargs["show_progress_bar"] = False

        embeddings: List = []
        for batch in self.batchify(contents, batch_size, None):
            batch_embeddings = self._client.encode(batch, **kwargs)
            embeddings.extend([embedding.tolist() for embedding in batch_embeddings])
        return embeddings

    @property
    def type(self) -> str:
        return "hf"

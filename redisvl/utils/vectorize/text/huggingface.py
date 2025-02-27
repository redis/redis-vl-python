from typing import Any, Callable, List, Optional, Union

from pydantic.v1 import PrivateAttr

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer


class HFTextVectorizer(BaseVectorizer):
    """The HFTextVectorizer class is designed to leverage the power of Hugging
    Face's Sentence Transformers for generating text embeddings. This vectorizer
    is particularly useful in scenarios where advanced natural language
    processing and understanding are required, and ideal for running on your own
    hardware (for free).

    Utilizing this vectorizer involves specifying a pre-trained model from
    Hugging Face's vast collection of Sentence Transformers. These models are
    trained on a variety of datasets and tasks, ensuring versatility and
    robust performance across different text embedding needs. Additionally,
    make sure the `sentence-transformers` library is installed with
    `pip install sentence-transformers==2.2.2`.

    .. code-block:: python

        # Embedding a single text
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
        embedding = vectorizer.embed("Hello, world!")

        # Embedding a batch of texts
        embeddings = vectorizer.embed_many(["Hello, world!", "How are you?"], batch_size=2)

    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        dtype: str = "float32",
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

        Raises:
            ImportError: If the sentence-transformers library is not installed.
            ValueError: If there is an error setting the embedding model dimensions.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype)
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
            embedding = self.embed("dimension check")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Empty response from the embedding model: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    @deprecated_argument("dtype")
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Embed a chunk of text using the Hugging Face sentence transformer.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing
                callable to perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            Union[List[float], bytes]: Embedding as a list of floats, or as a bytes
            object if as_buffer=True

        Raises:
            TypeError: If the wrong input type is passed in for the text.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        embedding = self._client.encode([text], **kwargs)[0]
        return self._process_embedding(embedding.tolist(), as_buffer, dtype)

    @deprecated_argument("dtype")
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Asynchronously embed many chunks of texts using the Hugging Face
        sentence transformer.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing
                callable to perform before vectorization. Defaults to None.
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
            batch_embeddings = self._client.encode(batch, **kwargs)
            embeddings.extend(
                [
                    self._process_embedding(embedding.tolist(), as_buffer, dtype)
                    for embedding in batch_embeddings
                ]
            )
        return embeddings

    @property
    def type(self) -> str:
        return "hf"

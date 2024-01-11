from typing import Callable, Dict, List, Optional

from redisvl.vectorize.base import BaseVectorizer


class HFTextVectorizer(BaseVectorizer):
    """
    The HFTextVectorizer class is designed to leverage the power of
    Hugging Face's Sentence Transformers for generating text embeddings.
    This vectorizer is particularly useful in scenarios where advanced
    natural language processing and understanding are required, and ideal for
    running on your own hardware (for free).

    Utilizing this vectorizer involves specifying a pre-trained model from
    Hugging Face's vast collection of Sentence Transformers. These models are
    trained on a variety of datasets and tasks, ensuring versatility and
    robust performance across different text embedding needs. Additionally,
    make sure the `sentence-transformers` library is installed with
    `pip install sentence-transformers==2.2.2`.

    Example:
        # Embedding a single text
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
        embedding = vectorizer.embed("Hello, world!")

        # Embedding a batch of texts
        embeddings = vectorizer.embed_many(["Hello, world!", "How are you?"], batch_size=2)
    """

    def __init__(
        self, model: str = "sentence-transformers/all-mpnet-base-v2", **kwargs
    ):
        """
        Initialize the Hugging Face text vectorizer.

        Args:
            model (str): The pre-trained model from Hugging Face's Sentence
                Transformers to be used for embedding. Defaults to
                'sentence-transformers/all-mpnet-base-v2'.

        Raises:
            ImportError: If the sentence-transformers library is not installed.
            ValueError: If there is an error setting the embedding model dimensions.
        """
        super().__init__(model)

        # Load the SentenceTransformer model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "HFTextVectorizer requires the sentence-transformers library. "
                "Please install with `pip install sentence-transformers`"
            )

        self._model_client = SentenceTransformer(model)

        # Initialize model dimensions
        try:
            self._dims = self._set_model_dims()
        except Exception as e:
            raise ValueError(f"Error setting embedding model dimensions: {e}")

    def _set_model_dims(self):
        embedding = self._model_client.encode(["dimension check"])[0]
        return len(embedding)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
    ) -> List[float]:
        """
        Embed a chunk of text using the Hugging Face sentence transformer.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing
                callable to perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)
        embedding = self._model_client.encode([text])[0]
        return self._process_embedding(embedding.tolist(), as_buffer)

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
    ) -> List[List[float]]:
        """
        Asynchronously embed many chunks of texts using the Hugging Face
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
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            batch_embeddings = self._model_client.encode(batch)
            embeddings.extend(
                [
                    self._process_embedding(embedding.tolist(), as_buffer)
                    for embedding in batch_embeddings
                ]
            )
        return embeddings

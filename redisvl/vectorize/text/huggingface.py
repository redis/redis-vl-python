from typing import Callable, Dict, List, Optional

from redisvl.vectorize.base import BaseVectorizer


class HFTextVectorizer(BaseVectorizer):
    # TODO - add docstring
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        # TODO set dims based on model
        dims = 768
        super().__init__(model, dims, api_config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "HFTextVectorizer requires sentence-transformers library. Please install with pip install sentence-transformers"
            )

        self._model_client = SentenceTransformer(model)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: Optional[float] = False
    ) -> List[float]:
        """Embed a chunk of text using the Hugging Face sentence transformer.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.
        """
        if preprocess:
            text = preprocess(text)
        embedding = self._model_client.encode([text])[0]
        return self._process_embedding(embedding.tolist(), as_buffer)

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: Optional[float] = None
    ) -> List[List[float]]:
        """Asynchronously embed many chunks of texts using the Hugging Face sentence
        transformer.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.
        """
        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            batch_embeddings = self._model_client.encode(batch)
            embeddings.extend([
                self._process_embedding(embedding.tolist(), as_buffer) for embedding in batch_embeddings
            ])
        return embeddings

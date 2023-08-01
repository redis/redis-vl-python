from typing import Callable, Dict, List, Optional

from redisvl.vectorize.base import BaseVectorizer


class HuggingfaceVectorizer(BaseVectorizer):
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        # TODO set dims based on model
        dims = 768
        super().__init__(model, dims, api_config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Huggingface vectorizer requires sentence-transformers library. Please install with pip install sentence-transformers"
            )

        self._model_client = SentenceTransformer(model)

    def embed(
        self, emb_input: str, preprocess: Optional[Callable] = None
    ) -> List[float]:
        if preprocess:
            emb_input = preprocess(emb_input)
        embedding = self._model_client.encode([emb_input])[0]
        return embedding.tolist()

    def embed_many(
        self,
        inputs: List[str],
        preprocess: Optional[Callable] = None,
        chunk_size: int = 1000,
    ) -> List[List[float]]:
        embeddings = []
        for batch in self.batchify(inputs, chunk_size, preprocess):
            batch_embeddings = self._model_client.encode(batch)
            embeddings.extend([embedding.tolist() for embedding in batch_embeddings])
        return embeddings

from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from redisvl.providers.base import BaseProvider


class HuggingfaceProvider(BaseProvider):
    def __init__(self, model: str, api_config: Optional[Dict] = None):
        # TODO set dims based on model
        dims = 768
        super().__init__(model, dims, api_config)
        self._model_client = SentenceTransformer(model)

    def embed(self, emb_input: str, preprocess: callable = None) -> List[float]:
        if preprocess:
            emb_input = preprocess(emb_input)
        embedding = self._model_client.encode([emb_input])[0]
        return embedding.tolist()

    def embed_many(
        self, inputs: List[str], preprocess: callable = None, chunk_size: int = 1000
    ) -> List[List[float]]:

        embeddings = []
        for batch in self.batchify(inputs, chunk_size, preprocess):
            batch_embeddings = self._model_client.encode(batch)
            embeddings.extend([embedding.tolist() for embedding in batch_embeddings])
        return embeddings

"""Download the HuggingFace models that the user-guide notebooks load.

The Notebook Validation job runs this before nbval, so model downloads happen
outside nbval's per-cell timeout. The job's HF cache key hashes this file, so
editing either list below produces a new cache entry that includes the change.
nbval then runs with HF_HUB_OFFLINE=1, so a notebook that loads a model missing
from these lists fails immediately instead of downloading inside a cell.
"""

import time

from redisvl.utils.rerank import HFCrossEncoderReranker
from redisvl.utils.vectorize import HFTextVectorizer

EMBEDDING_MODELS = [
    "redis/langcache-embed-v1",  # SemanticCache default
    "redis/langcache-embed-v2",
    "sentence-transformers/all-mpnet-base-v2",  # HFTextVectorizer default
]

RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
]


def main() -> None:
    # Load through the same classes the notebooks use, so exactly the files
    # they need end up in the cache.
    for loader, models in (
        (HFTextVectorizer, EMBEDDING_MODELS),
        (HFCrossEncoderReranker, RERANKER_MODELS),
    ):
        for model in models:
            start = time.monotonic()
            loader(model=model)
            print(f"{model}: {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

import os
from typing import Optional

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize.base import BaseVectorizer, Vectorizers
from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer
from redisvl.utils.vectorize.text.bedrock import BedrockTextVectorizer
from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer
from redisvl.utils.vectorize.text.custom import CustomTextVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
from redisvl.utils.vectorize.text.mistral import MistralAITextVectorizer
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer
from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer
from redisvl.utils.vectorize.text.voyageai import VoyageAITextVectorizer

__all__ = [
    "BaseVectorizer",
    "CohereTextVectorizer",
    "HFTextVectorizer",
    "OpenAITextVectorizer",
    "VertexAITextVectorizer",
    "AzureOpenAITextVectorizer",
    "MistralAITextVectorizer",
    "CustomTextVectorizer",
    "BedrockTextVectorizer",
    "VoyageAITextVectorizer",
]


def vectorizer_from_dict(
    vectorizer: dict,
    cache: dict = {},
    cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
) -> BaseVectorizer:
    vectorizer_type = Vectorizers(vectorizer["type"])
    model = vectorizer["model"]

    args = {"model": model}
    if cache:
        emb_cache = EmbeddingsCache(**cache)
        args["cache"] = emb_cache

    if vectorizer_type == Vectorizers.cohere:
        return CohereTextVectorizer(**args)
    elif vectorizer_type == Vectorizers.openai:
        return OpenAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.azure_openai:
        return AzureOpenAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.hf:
        return HFTextVectorizer(**args)
    elif vectorizer_type == Vectorizers.mistral:
        return MistralAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.vertexai:
        return VertexAITextVectorizer(**args)
    elif vectorizer_type == Vectorizers.voyageai:
        return VoyageAITextVectorizer(**args)
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

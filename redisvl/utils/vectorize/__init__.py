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


def vectorizer_from_dict(vectorizer: dict) -> BaseVectorizer:
    vectorizer_type = Vectorizers(vectorizer["type"])
    model = vectorizer["model"]
    if vectorizer_type == Vectorizers.cohere:
        return CohereTextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.openai:
        return OpenAITextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.azure_openai:
        return AzureOpenAITextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.hf:
        return HFTextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.mistral:
        return MistralAITextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.vertexai:
        return VertexAITextVectorizer(model=model)
    elif vectorizer_type == Vectorizers.voyageai:
        return VoyageAITextVectorizer(model=model)
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

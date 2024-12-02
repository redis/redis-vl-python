from redisvl.utils.vectorize.base import BaseVectorizer, Vectorizers
from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer
from redisvl.utils.vectorize.text.bedrock import BedrockTextVectorizer
from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer
from redisvl.utils.vectorize.text.custom import CustomTextVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
from redisvl.utils.vectorize.text.mistral import MistralAITextVectorizer
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer
from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer

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
]


def vectorizer_from_dict(vectorizer: dict) -> BaseVectorizer:
    vectorizer_type = Vectorizers(vectorizer["type"])
    model = vectorizer["model"]
    if vectorizer_type == Vectorizers.cohere:
        return CohereTextVectorizer(model)
    elif vectorizer_type == Vectorizers.openai:
        return OpenAITextVectorizer(model)
    elif vectorizer_type == Vectorizers.azure_openai:
        return AzureOpenAITextVectorizer(model)
    elif vectorizer_type == Vectorizers.hf:
        return HFTextVectorizer(model)
    elif vectorizer_type == Vectorizers.mistral:
        return MistralAITextVectorizer(model)
    elif vectorizer_type == Vectorizers.vertexai:
        return VertexAITextVectorizer(model)

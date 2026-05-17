---
description: Vectorizer providers for RedisVL: OpenAI, Cohere, HuggingFace, Vertex, Bedrock, Mistral, VoyageAI.
---

# Vectorizers

Vectorizers convert text into embedding vectors. RedisVL ships with adapters
for common cloud and local providers, all behind a single interface.

!!! note "Backwards compatibility"

    Several vectorizers have deprecated aliases in `redisvl.utils.vectorize.text`:

    - `VoyageAITextVectorizer` → use `VoyageAIVectorizer`
    - `VertexAITextVectorizer` → use `VertexAIVectorizer`
    - `BedrockTextVectorizer` → use `BedrockVectorizer`
    - `CustomTextVectorizer` → use `CustomVectorizer`

    These aliases are deprecated as of version 0.13.0 and will be removed in a
    future major release.

## HFTextVectorizer

::: redisvl.utils.vectorize.text.huggingface.HFTextVectorizer
    options:
      show_root_heading: true

## OpenAITextVectorizer

::: redisvl.utils.vectorize.text.openai.OpenAITextVectorizer
    options:
      show_root_heading: true

## AzureOpenAITextVectorizer

::: redisvl.utils.vectorize.text.azureopenai.AzureOpenAITextVectorizer
    options:
      show_root_heading: true

## VertexAIVectorizer

::: redisvl.utils.vectorize.vertexai.VertexAIVectorizer
    options:
      show_root_heading: true

## CohereTextVectorizer

::: redisvl.utils.vectorize.text.cohere.CohereTextVectorizer
    options:
      show_root_heading: true

## BedrockVectorizer

::: redisvl.utils.vectorize.bedrock.BedrockVectorizer
    options:
      show_root_heading: true

## CustomVectorizer

::: redisvl.utils.vectorize.custom.CustomVectorizer
    options:
      show_root_heading: true

## VoyageAIVectorizer

::: redisvl.utils.vectorize.voyageai.VoyageAIVectorizer
    options:
      show_root_heading: true

## MistralAITextVectorizer

::: redisvl.utils.vectorize.text.mistral.MistralAITextVectorizer
    options:
      show_root_heading: true

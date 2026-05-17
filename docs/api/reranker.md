---
description: Reranker providers for RedisVL: Cohere, HuggingFace cross-encoder, VoyageAI.
---

# Rerankers

Rerankers reorder a candidate list using a higher-quality model after the
initial vector or hybrid search. RedisVL provides adapters for several
reranking services and local cross-encoders.

## CohereReranker

::: redisvl.utils.rerank.cohere.CohereReranker
    options:
      show_root_heading: true

## HFCrossEncoderReranker

::: redisvl.utils.rerank.hf_cross_encoder.HFCrossEncoderReranker
    options:
      show_root_heading: true

## VoyageAIReranker

::: redisvl.utils.rerank.voyageai.VoyageAIReranker
    options:
      show_root_heading: true

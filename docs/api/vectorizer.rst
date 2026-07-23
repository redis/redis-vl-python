***********
Vectorizers
***********

.. note::
   **Backwards Compatibility:** Several vectorizers have deprecated aliases
   available in the ``redisvl.utils.vectorize.text`` module for backwards
   compatibility:

   - ``VoyageAITextVectorizer`` → Use ``VoyageAIVectorizer`` instead
   - ``VertexAITextVectorizer`` → Use ``GoogleGenAIVectorizer`` instead
     (``VertexAIVectorizer`` is itself deprecated; see below)
   - ``BedrockTextVectorizer`` → Use ``BedrockVectorizer`` instead
   - ``CustomTextVectorizer`` → Use ``CustomVectorizer`` instead

   These aliases are deprecated as of version 0.13.0 and will be removed
   in a future major release.

HFTextVectorizer
================

.. _hftextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.huggingface

.. autoclass:: HFTextVectorizer
   :show-inheritance:
   :members:


OpenAITextVectorizer
====================

.. _openaitextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.openai

.. autoclass:: OpenAITextVectorizer
   :show-inheritance:
   :members:


AzureOpenAITextVectorizer
=========================

.. _azureopenaitextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.azureopenai

.. autoclass:: AzureOpenAITextVectorizer
   :show-inheritance:
   :members:


VertexAIVectorizer
======================

.. _vertexaivectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.vertexai

.. note::
    ``VertexAIVectorizer`` is **deprecated**. It uses Google's Vertex AI
    model-garden SDK, which Google has deprecated with a scheduled removal. Use
    :class:`~redisvl.utils.vectorize.googlegenai.GoogleGenAIVectorizer` for text
    embeddings on the supported ``google-genai`` SDK. The alias
    ``VertexAITextVectorizer`` (in ``redisvl.utils.vectorize.text``) is likewise
    deprecated. Multimodal (image/video) migration is tracked in
    `issue #620 <https://github.com/redis/redis-vl-python/issues/620>`_.

.. autoclass:: VertexAIVectorizer
   :show-inheritance:
   :members:


GoogleGenAIVectorizer
=====================

.. _googlegenaivectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.googlegenai

.. autoclass:: GoogleGenAIVectorizer
   :show-inheritance:
   :members:


CohereTextVectorizer
====================

.. _coheretextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.cohere

.. autoclass:: CohereTextVectorizer
   :show-inheritance:
   :members:


BedrockVectorizer
=====================

.. _bedrockvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.bedrock

.. note::
    For backwards compatibility, an alias ``BedrockTextVectorizer`` is available
    in the ``redisvl.utils.vectorize.text`` module. This alias is deprecated
    as of version 0.13.0 and will be removed in a future major release.

.. autoclass:: BedrockVectorizer
   :show-inheritance:
   :members:


CustomVectorizer
====================

.. _customvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.custom

.. note::
    For backwards compatibility, an alias ``CustomTextVectorizer`` is available
    in the ``redisvl.utils.vectorize.text`` module. This alias is deprecated
    as of version 0.13.0 and will be removed in a future major release.

.. autoclass:: CustomVectorizer
   :show-inheritance:
   :members:


VoyageAIVectorizer
======================

.. _voyageaivectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.voyageai

.. note::
    For backwards compatibility, an alias ``VoyageAITextVectorizer`` is available
    in the ``redisvl.utils.vectorize.text`` module. This alias is deprecated
    as of version 0.13.0 and will be removed in a future major release.

.. autoclass:: VoyageAIVectorizer
   :show-inheritance:
   :members:


MistralAITextVectorizer
========================

.. _mistralaitextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.mistral

.. autoclass:: MistralAITextVectorizer
   :show-inheritance:
   :members:

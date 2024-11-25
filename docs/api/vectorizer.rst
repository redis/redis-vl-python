***********
Vectorizers
***********

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


VertexAITextVectorizer
======================

.. _vertexaitextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.vertexai

.. autoclass:: VertexAITextVectorizer
   :show-inheritance:
   :members:


CohereTextVectorizer
====================

.. _coheretextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.cohere

.. autoclass:: CohereTextVectorizer
   :show-inheritance:
   :members:

BedrockTextVectorizer
====================

.. _bedrocktextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.bedrock

.. autoclass:: BedrockTextVectorizer
   :show-inheritance:
   :members:

   The BedrockTextVectorizer class utilizes Amazon Bedrock's API to generate
   embeddings for text data. This vectorizer requires AWS credentials, which can be provided
   via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) or
   through the api_config parameter.

   Example::

        # Initialize with environment variables
        vectorizer = BedrockTextVectorizer(model_id="amazon.titan-embed-text-v2:0")

        # Or with explicit credentials
        vectorizer = BedrockTextVectorizer(
            model_id="amazon.titan-embed-text-v2:0",
            api_config={
                "aws_access_key_id": "your_access_key",
                "aws_secret_access_key": "your_secret_key",
                "region_name": "us-east-1"
            }
        )

        # Generate embeddings
        embedding = vectorizer.embed("Hello world")
        embeddings = vectorizer.embed_many(["Hello", "World"])


CustomTextVectorizer
====================

.. _customtextvectorizer_api:

.. currentmodule:: redisvl.utils.vectorize.text.custom

.. autoclass:: CustomTextVectorizer
   :show-inheritance:
   :members:
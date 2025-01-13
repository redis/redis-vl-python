import os
from typing import Any, Callable, Dict, List, Optional

from pydantic.v1 import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that voyageai isn't imported
# mypy: disable-error-code="name-defined"


class VoyageAITextVectorizer(BaseVectorizer):
    """The VoyageAITextVectorizer class utilizes VoyageAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with VoyageAI's /embed API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `VOYAGE_API_KEY`
    environment variable. User must obtain an API key from VoyageAI's website
    (https://dash.voyageai.com/). Additionally, the `voyageai` python
    client must be installed with `pip install voyageai`.

    The vectorizer supports both synchronous and asynchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks.

    .. code-block:: python

        from redisvl.utils.vectorize import VoyageAITextVectorizer

        vectorizer = VoyageAITextVectorizer(
            model="voyage-large-2",
            api_config={"api_key": "your-voyageai-api-key"} # OR set VOYAGE_API_KEY in your env
        )
        query_embedding = vectorizer.embed(
            text="your input query text here",
            input_type="search_query"
        )
        doc_embeddings = vectorizer.embed_many(
            texts=["your document text", "more document text"],
            input_type="search_document"
        )

    """

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self, model: str, api_config: Optional[Dict] = None
    ):
        """Initialize the VoyageAI vectorizer.

        Visit https://docs.voyageai.com/docs/embeddings to learn about embeddings and check the available models.

        Args:
            model (str): Model to use for embedding.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.

        Raises:
            ImportError: If the voyageai library is not installed.
            ValueError: If the API key is not provided.

        """
        self._initialize_client(api_config)
        super().__init__(model=model, dims=self._set_model_dims(model))

    def _initialize_client(self, api_config: Optional[Dict]):
        """
        Setup the VoyageAI clients using the provided API key or an
        environment variable.
        """
        # Dynamic import of the voyageai module
        try:
            from voyageai import AsyncClient, Client
        except ImportError:
            raise ImportError(
                "VoyageAI vectorizer requires the voyageai library. \
                    Please install with `pip install voyageai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("VOYAGE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "VoyageAI API key is required. "
                "Provide it in api_config or set the VOYAGE_API_KEY environment variable."
            )
        self._client = Client(api_key=api_key)
        self._aclient = AsyncClient(api_key=api_key)

    def _set_model_dims(self, model) -> int:
        try:
            embedding = self._client.embed(
                texts=["dimension test"],
                model=model,
                input_type="document",
            ).embeddings[0]
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the VoyageAI API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Embed a chunk of text using the VoyageAI Embeddings API.

        Can provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model. For retrieval/search use cases,
        we recommend specifying this argument when encoding queries or documents to enhance retrieval quality.
        Embeddings generated with and without the input_type argument are compatible.

        Supported input types are ``document`` and ``query``

        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type="document" and when you are
        querying the database, you should set the input_type="query".

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
            truncation (bool): Whether to truncate the input texts to fit within the context length.
                Check https://docs.voyageai.com/docs/embeddings

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: In an invalid input_type is provided.
        """
        return self.embed_many(
            texts=[text],
            preprocess=preprocess,
            as_buffer=as_buffer,
            **kwargs
        )[0]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Embed many chunks of text using the VoyageAI Embeddings API.

        Can provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model. For retrieval/search use cases,
        we recommend specifying this argument when encoding queries or documents to enhance retrieval quality.
        Embeddings generated with and without the input_type argument are compatible.

        Supported input types are ``document`` and ``query``

        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type="document" and when you are
        querying the database, you should set the input_type="query".

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. .
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
            truncation (bool): Whether to truncate the input texts to fit within the context length.
                Check https://docs.voyageai.com/docs/embeddings

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: In an invalid input_type is provided.

        """
        input_type = kwargs.get("input_type")
        truncation = kwargs.get("truncation")

        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if input_type is not None and input_type not in ['document', 'query']:
            raise TypeError(
                "Must pass in a allowed value for voyageai embedding input_type. \
                    See https://docs.voyageai.com/docs/embeddings."
            )

        if truncation is not None and not isinstance(truncation, bool):
            raise TypeError("Truncation (optional) parameter is a bool.")

        if batch_size is None:
            batch_size = 72 if self.model in ["voyage-2", "voyage-02"] else 7

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._client.embed(
                texts=batch, model=self.model, input_type=input_type
            )
            embeddings += [
                self._process_embedding(embedding, as_buffer)
                for embedding in response.embeddings
            ]
        return embeddings

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Embed many chunks of text using the VoyageAI Embeddings API.

        Can provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model. For retrieval/search use cases,
        we recommend specifying this argument when encoding queries or documents to enhance retrieval quality.
        Embeddings generated with and without the input_type argument are compatible.

        Supported input types are ``document`` and ``query``

        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type="document" and when you are
        querying the database, you should set the input_type="query".

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. .
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
            truncation (bool): Whether to truncate the input texts to fit within the context length.
                Check https://docs.voyageai.com/docs/embeddings

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: In an invalid input_type is provided.

        """
        input_type = kwargs.get("input_type")
        truncation = kwargs.get("truncation")

        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if input_type is not None and input_type not in ['document', 'query']:
            raise TypeError(
                "Must pass in a allowed value for voyageai embedding input_type. \
                    See https://docs.voyageai.com/docs/embeddings."
            )

        if truncation is not None and not isinstance(truncation, bool):
            raise TypeError("Truncation (optional) parameter is a bool.")

        if batch_size is None:
            batch_size = 72 if self.model in ["voyage-2", "voyage-02"] else 7

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = await self._aclient.embed(
                texts=batch, model=self.model, input_type=input_type
            )
            embeddings += [
                self._process_embedding(embedding, as_buffer)
                for embedding in response.embeddings
            ]
        return embeddings

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Embed a chunk of text using the VoyageAI Embeddings API.

        Can provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model. For retrieval/search use cases,
        we recommend specifying this argument when encoding queries or documents to enhance retrieval quality.
        Embeddings generated with and without the input_type argument are compatible.

        Supported input types are ``document`` and ``query``

        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type="document" and when you are
        querying the database, you should set the input_type="query".

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
            truncation (bool): Whether to truncate the input texts to fit within the context length.
                Check https://docs.voyageai.com/docs/embeddings

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: In an invalid input_type is provided.
        """

        result = await self.aembed_many(
            texts=[text],
            preprocess=preprocess,
            as_buffer=as_buffer,
            **kwargs
        )
        return result[0]

import os
from typing import Any, Callable, Dict, List, Optional

from pydantic.v1 import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that cohere isn't imported
# mypy: disable-error-code="name-defined"


class CohereTextVectorizer(BaseVectorizer):
    """The CohereTextVectorizer class utilizes Cohere's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Cohere's /embed API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `COHERE_API_KEY`
    environment variable. User must obtain an API key from Cohere's website
    (https://dashboard.cohere.com/). Additionally, the `cohere` python
    client must be installed with `pip install cohere`.

    The vectorizer supports only synchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks.

    .. code-block:: python

        from redisvl.utils.vectorize import CohereTextVectorizer

        vectorizer = CohereTextVectorizer(
            model="embed-english-v3.0",
            api_config={"api_key": "your-cohere-api-key"} # OR set COHERE_API_KEY in your env
        )
        query_embedding = vectorizer.embed(
            text="your input query text here",
            input_type="search_query"
        )
        doc_embeddings = cohere.embed_many(
            texts=["your document text", "more document text"],
            input_type="search_document"
        )

    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
    ):
        """Initialize the Cohere vectorizer.

        Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            model (str): Model to use for embedding. Defaults to 'embed-english-v3.0'.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.

        Raises:
            ImportError: If the cohere library is not installed.
            ValueError: If the API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        self._initialize_client(api_config)
        super().__init__(model=model, dims=self._set_model_dims(model), dtype=dtype)

    def _initialize_client(self, api_config: Optional[Dict]):
        """
        Setup the Cohere clients using the provided API key or an
        environment variable.
        """
        # Dynamic import of the cohere module
        try:
            from cohere import AsyncClient, Client
        except ImportError:
            raise ImportError(
                "Cohere vectorizer requires the cohere library. \
                    Please install with `pip install cohere`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("COHERE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Cohere API key is required. "
                "Provide it in api_config or set the COHERE_API_KEY environment variable."
            )
        self._client = Client(api_key=api_key, client_name="redisvl")

    def _set_model_dims(self, model) -> int:
        try:
            embedding = self._client.embed(
                texts=["dimension test"],
                model=model,
                input_type="search_document",
            ).embeddings[0]
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the Cohere API: {str(ke)}")
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
        """Embed a chunk of text using the Cohere Embeddings API.

        Must provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model.

        Supported input types:
            - ``search_document``: Used for embeddings stored in a vector database for search use-cases.
            - ``search_query``: Used for embeddings of search queries run against a vector DB to find relevant documents.
            - ``classification``: Used for embeddings passed through a text classifier
            - ``clustering``: Used for the embeddings run through a clustering algorithm.

        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type= "search_document" and when you are
        querying the database, you should set the input_type = "search query".
        If you want to use the embeddings for a classification or clustering
        task downstream, you should set input_type= "classification" or
        "clustering".

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
                Required for embedding models v3 and higher.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: In an invalid input_type is provided.

        """
        input_type = kwargs.get("input_type")

        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for cohere embedding input_type. \
                    See https://docs.cohere.com/reference/embed."
            )

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        embedding = self._client.embed(
            texts=[text], model=self.model, input_type=input_type
        ).embeddings[0]
        return self._process_embedding(embedding, as_buffer, dtype)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Embed many chunks of text using the Cohere Embeddings API.

        Must provide the embedding `input_type` as a `kwarg` to this method
        that specifies the type of input you're giving to the model.

        Supported input types:
            - ``search_document``: Used for embeddings stored in a vector database for search use-cases.
            - ``search_query``: Used for embeddings of search queries run against a vector DB to find relevant documents.
            - ``classification``: Used for embeddings passed through a text classifier
            - ``clustering``: Used for the embeddings run through a clustering algorithm.


        When hydrating your Redis DB, the documents you want to search over
        should be embedded with input_type= "search_document" and when you are
        querying the database, you should set the input_type = "search query".
        If you want to use the embeddings for a classification or clustering
        task downstream, you should set input_type= "classification" or
        "clustering".

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.
            input_type (str): Specifies the type of input passed to the model.
                Required for embedding models v3 and higher.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: In an invalid input_type is provided.

        """
        input_type = kwargs.get("input_type")

        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for cohere embedding input_type.\
                    See https://docs.cohere.com/reference/embed."
            )

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._client.embed(
                texts=batch, model=self.model, input_type=input_type
            )
            embeddings += [
                self._process_embedding(embedding, as_buffer, dtype)
                for embedding in response.embeddings
            ]
        return embeddings

    @property
    def type(self) -> str:
        return "cohere"

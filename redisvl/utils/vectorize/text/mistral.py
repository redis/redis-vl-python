import os
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.utils import deprecated_argument
from redisvl.utils.vectorize.base import BaseVectorizer

# ignore that mistralai isn't imported
# mypy: disable-error-code="name-defined"


class MistralAITextVectorizer(BaseVectorizer):
    """The MistralAITextVectorizer class utilizes MistralAI's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Mistral's embeddings API,
    requiring an API key for authentication. The key can be provided directly
    in the `api_config` dictionary or through the `MISTRAL_API_KEY` environment
    variable. Users must obtain an API key from Mistral's website
    (https://console.mistral.ai/). Additionally, the `mistralai` python client
    must be installed with `pip install mistralai`.

    The vectorizer supports both synchronous and asynchronous operations,
    allowing for batch processing of texts and flexibility in handling
    preprocessing tasks.

    .. code-block:: python

        # Synchronous embedding of a single text
        vectorizer = MistralAITextVectorizer(
            model="mistral-embed"
            api_config={"api_key": "your_api_key"} # OR set MISTRAL_API_KEY in your env
        )
        embedding = vectorizer.embed("Hello, world!")

        # Asynchronous batch embedding of multiple texts
        embeddings = await vectorizer.aembed_many(
            ["Hello, world!", "How are you?"],
            batch_size=2
        )

    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "mistral-embed",
        api_config: Optional[Dict] = None,
        dtype: str = "float32",
        **kwargs,
    ):
        """Initialize the MistralAI vectorizer.

        Args:
            model (str): Model to use for embedding. Defaults to
                'text-embedding-ada-002'.
            api_config (Optional[Dict], optional): Dictionary containing the
                API key. Defaults to None.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.

        Raises:
            ImportError: If the mistralai library is not installed.
            ValueError: If the Mistral API key is not provided.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype)
        # Init client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after init
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the Mistral clients using the provided API key or an
        environment variable.
        """
        if api_config is None:
            api_config = {}

        # Dynamic import of the mistralai module
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "MistralAI vectorizer requires the mistralai library. \
                    Please install with `pip install mistralai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("MISTRAL_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "MISTRAL API key is required. "
                "Provide it in api_config or set the MISTRAL_API_KEY\
                    environment variable."
            )

        self._client = Mistral(api_key=api_key, **kwargs)

    def _set_model_dims(self) -> int:
        try:
            embedding = self.embed("dimension check")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the MISTRAL API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Embed many chunks of texts using the Mistral API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing
                callable to perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            Union[List[List[float]], List[bytes]]: List of embeddings as lists of floats,
            or as bytes objects if as_buffer=True

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._client.embeddings.create(
                model=self.model, inputs=batch, **kwargs
            )
            embeddings += [
                self._process_embedding(r.embedding, as_buffer, dtype)
                for r in response.data
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Embed a chunk of text using the Mistral API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            Union[List[float], bytes]: Embedding as a list of floats, or as a bytes
            object if as_buffer=True

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = self._client.embeddings.create(
            model=self.model, inputs=[text], **kwargs
        )
        return self._process_embedding(result.data[0].embedding, as_buffer, dtype)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Asynchronously embed many chunks of texts using the Mistral API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[List[float]]: List of embeddings.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")

        dtype = kwargs.pop("dtype", self.dtype)

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = await self._client.embeddings.create_async(
                model=self.model, inputs=batch, **kwargs
            )
            embeddings += [
                self._process_embedding(r.embedding, as_buffer, dtype)
                for r in response.data
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    @deprecated_argument("dtype")
    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Asynchronously embed a chunk of text using the MistralAPI.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the test.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)

        dtype = kwargs.pop("dtype", self.dtype)

        result = await self._client.embeddings.create_async(
            model=self.model, inputs=[text], **kwargs
        )
        return self._process_embedding(result.data[0].embedding, as_buffer, dtype)

    @property
    def type(self) -> str:
        return "mistral"

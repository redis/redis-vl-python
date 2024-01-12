import os
from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer


class CohereTextVectorizer(BaseVectorizer):
    """
    The CohereTextVectorizer class utilizes Cohere's API to generate embeddings
    for text data.

    This vectorizer is designed to interact with Cohere's /embed API, requiring an API key for authentication. The key
    can be provided directly in the `api_config` dictionary or through the `COHERE_API_KEY` environment variable. Users
    must obtain an API key from Cohere's website (https://dashboard.cohere.com/). Additionally, the `cohere` python
    client must be installed with `pip install cohere`.

    The vectorizer supports only synchronous operations, allows for batch processing of texts and flexibility in
    handling preprocessing tasks.
    """

    def __init__(
        self, model: str = "embed-english-v3.0", api_config: Optional[Dict] = None
    ):
        """Initialize the Cohere vectorizer. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            model (str): Model to use for embedding. Defaults to 'embed-english-v3.0'.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.

        Raises:
            ImportError: If the cohere library is not installed.
            ValueError: If the API key is not provided.
        """
        super().__init__(model)
        # Dynamic import of the cohere module
        try:
            global cohere
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere vectorizer requires the cohere library. Please install with `pip install cohere`"
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

        self._model = model
        self._model_client = cohere.Client(api_key)
        self._dims = self._set_model_dims()

    def _set_model_dims(self) -> int:
        try:
            embedding = self._model_client.embed(
                texts=["dimension test"],
                model=self._model,
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
        input_type: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
    ) -> List[float]:
        """Embed a chunk of text using the Cohere API.

        Args:
            text (str): Chunk of text to embed.
            input_type (str): Specifies the type of input you're giving to the model.
                Not required for older versions of the embedding models (i.e. anything lower than v3), but is required
                for more recent versions (i.e. anything bigger than v2).
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (bool, optional): Whether to convert the raw embedding
                to a byte string. Defaults to False.

        Returns:
            List[float]: Embedding.

        Raises:
            TypeError: If the wrong input type is passed in for the text.
        """
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for input_type. See https://docs.cohere.com/reference/embed."
            )
        if preprocess:
            text = preprocess(text)
        embedding = self._model_client.embed(
            texts=[text], model=self._model, input_type=input_type
        ).embeddings[0]
        return self._process_embedding(embedding, as_buffer)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed_many(
        self,
        texts: List[str],
        input_type: str,
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
    ) -> List[List[float]]:
        """Embed many chunks of texts using the Cohere API.

        Args:
            texts (List[str]): List of text chunks to embed.
            input_type (str): Specifies the type of input you're giving to the model.
                Not required for older versions of the embedding models (i.e. anything lower than v3), but is required
                for more recent versions (i.e. anything bigger than v2).
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
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for input_type. See https://docs.cohere.com/reference/embed."
            )
        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._model_client.embed(
                texts=batch, model=self._model, input_type=input_type
            )
            embeddings += [
                self._process_embedding(embedding, as_buffer)
                for embedding in response.embeddings
            ]
        return embeddings

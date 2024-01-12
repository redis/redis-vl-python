from typing import Callable, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer

class CohereTextVectorizer(BaseVectorizer):
    """Cohere text vectorizer.

    This vectorizer uses the Cohere API to create embeddings for text. It
    requires an API key to be passed to the initialization. The API key
    can be obtained from <PUT URL HERE>
    """

    def __init__(self, model: str, truncate: Optional[str] = 'NONE', api_config: Optional[Dict] = None):
        """Initialize the Cohere vectorizer. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            model (str): Model to use for embedding.
            kwargs  (Dict): Additional arguments to pass to the Cohere API.
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

        if not api_config or "api_key" not in api_config:
            raise ValueError("Cohere API key is required in api_config")

        self._model = model
        self.truncate = truncate
        self._model_client = cohere.Client(api_config["api_key"])
        self._dims = self._set_model_dims()
    
    def _set_model_dims(self) -> int:
        try:
            embedding = self._model_client.embed(
                texts=["dimension test"], model=self._model, input_type="search_document",
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
            raise TypeError("Must pass in a str value for input_type. See https://docs.cohere.com/reference/embed.")
        if preprocess:
            text = preprocess(text)
        embedding = self._model_client.embed(texts=[text], model=self._model, input_type=input_type, truncate=self.truncate).embeddings[0]
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
            raise TypeError("Must pass in a str value for input_type. See https://docs.cohere.com/reference/embed.")
        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._model_client.embed(texts=batch, model=self._model, input_type=input_type)
            embeddings += [
                self._process_embedding(embedding, as_buffer)
                for embedding in response.embeddings
            ]
        return embeddings
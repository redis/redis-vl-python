from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.vectorize.base import BaseVectorizer


class VertexAITextVectorizer(BaseVectorizer):
    """VertexAI text vectorizer

    This vectorizer uses the VertexAI Palm 2 embedding model API to create embeddings for text. It requires an
    active GCP project, location, and application credentials.
    """

    def __init__(
        self, model: str = "textembedding-gecko", api_config: Optional[Dict] = None
    ):
        """Initialize the VertexAI vectorizer.

        Args:
            model (str): Model to use for embedding.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.

        Raises:
            ImportError: If the google-cloud-aiplatform library is not installed.
            ValueError: If the API key is not provided.
        """
        super().__init__(model)
        try:
            global vertexai
            import vertexai
            from vertexai.preview.language_models import TextEmbeddingModel
        except ImportError:
            raise ImportError(
                "VertexAI vectorizer requires the google-cloud-aiplatform library."
                "Please install with pip install google-cloud-aiplatform>=1.26"
            )

        if (
            not api_config
            or "project_id" not in api_config
            or "location" not in api_config
        ):
            raise ValueError(
                "GCP project id and valid location are required in the api_config"
            )

        self._model_client = TextEmbeddingModel.from_pretrained(model)
        self._dims = self._set_model_dims()

    def _set_model_dims(self) -> int:
        try:
            embedding = self._model_client.get_embeddings(["dimension test"])[0].values
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the VertexAI API: {str(ke)}")
        # TODO - except openai.error.AuthenticationError as ae:
        #     raise ValueError(f"Error authenticating with the OpenAI API: {str(ae)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = 10,
        as_buffer: Optional[float] = False,
    ) -> List[List[float]]:
        """Embed many chunks of texts using the VertexAI API.

        Args:
            texts (List[str]): List of text chunks to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            batch_size (int, optional): Batch size of texts to use when creating
                embeddings. Defaults to 10.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
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

        embeddings: List = []
        for batch in self.batchify(texts, batch_size, preprocess):
            response = self._model_client.get_embeddings(batch)
            embeddings += [
                self._process_embedding(r.values, as_buffer) for r in response
            ]
        return embeddings

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: Optional[float] = False,
    ) -> List[float]:
        """Embed a chunk of text using the VertexAI API.

        Args:
            text (str): Chunk of text to embed.
            preprocess (Optional[Callable], optional): Optional preprocessing callable to
                perform before vectorization. Defaults to None.
            as_buffer (Optional[float], optional): Whether to convert the raw embedding
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
        result = self._model_client.get_embeddings([text])
        return self._process_embedding(result[0].values, as_buffer)

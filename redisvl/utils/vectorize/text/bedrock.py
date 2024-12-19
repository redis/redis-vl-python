import json
import os
from typing import Any, Callable, Dict, List, Optional

from pydantic.v1 import PrivateAttr
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from redisvl.utils.vectorize.base import BaseVectorizer


class BedrockTextVectorizer(BaseVectorizer):
    """The AmazonBedrockTextVectorizer class utilizes Amazon Bedrock's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Amazon Bedrock API,
    requiring AWS credentials for authentication. The credentials can be provided
    directly in the `api_config` dictionary or through environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (defaults to us-east-1)

    The vectorizer supports synchronous operations with batch processing and
    preprocessing capabilities.

    .. code-block:: python

        # Initialize with explicit credentials
        vectorizer = AmazonBedrockTextVectorizer(
            model="amazon.titan-embed-text-v2:0",
            api_config={
                "aws_access_key_id": "your_access_key",
                "aws_secret_access_key": "your_secret_key",
                "aws_region": "us-east-1"
            }
        )

        # Initialize using environment variables
        vectorizer = AmazonBedrockTextVectorizer()

        # Generate embeddings
        embedding = vectorizer.embed("Hello, world!")
        embeddings = vectorizer.embed_many(["Hello", "World"], batch_size=2)
    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        api_config: Optional[Dict[str, str]] = None,
        dtype: str = "float32",
    ) -> None:
        """Initialize the AWS Bedrock Vectorizer.

        Args:
            model (str): The Bedrock model ID to use. Defaults to amazon.titan-embed-text-v2:0
            api_config (Optional[Dict[str, str]]): AWS credentials and config.
                Can include: aws_access_key_id, aws_secret_access_key, aws_region
                If not provided, will use environment variables.
            dtype (str): the default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.

        Raises:
            ValueError: If credentials are not provided in config or environment.
            ImportError: If boto3 is not installed.
            ValueError: If an invalid dtype is provided.
        """
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError(
                "Amazon Bedrock vectorizer requires boto3. "
                "Please install with `pip install boto3`"
            )

        if api_config is None:
            api_config = {}

        aws_access_key_id = api_config.get(
            "aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID")
        )
        aws_secret_access_key = api_config.get(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        aws_region = api_config.get("aws_region", os.getenv("AWS_REGION", "us-east-1"))

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS credentials required. Provide via api_config or environment variables "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )

        self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        super().__init__(model=model, dims=self._set_model_dims(model), dtype=dtype)

    def _set_model_dims(self, model: str) -> int:
        """Initialize model and determine embedding dimensions."""
        try:
            response = self._client.invoke_model(
                modelId=model, body=json.dumps({"inputText": "dimension test"})
            )
            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]
            return len(embedding)
        except Exception as e:
            raise ValueError(f"Error initializing Bedrock model: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs,
    ) -> List[float]:
        """Embed a chunk of text using Amazon Bedrock.

        Args:
            text (str): Text to embed.
            preprocess (Optional[Callable]): Optional preprocessing function.
            as_buffer (bool): Whether to return as byte buffer.

        Returns:
            List[float]: The embedding vector.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        if preprocess:
            text = preprocess(text)

        response = self._client.invoke_model(
            modelId=self.model, body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())
        embedding = response_body["embedding"]

        dtype = kwargs.pop("dtype", self.dtype)
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
        """Embed multiple texts using Amazon Bedrock.

        Args:
            texts (List[str]): List of texts to embed.
            preprocess (Optional[Callable]): Optional preprocessing function.
            batch_size (int): Size of batches for processing.
            as_buffer (bool): Whether to return as byte buffers.

        Returns:
            List[List[float]]: List of embedding vectors.

        Raises:
            TypeError: If texts is not a list of strings.
        """
        if not isinstance(texts, list):
            raise TypeError("Texts must be a list of strings")
        if texts and not isinstance(texts[0], str):
            raise TypeError("Texts must be a list of strings")

        embeddings: List[List[float]] = []
        dtype = kwargs.pop("dtype", self.dtype)

        for batch in self.batchify(texts, batch_size, preprocess):
            # Process each text in the batch individually since Bedrock
            # doesn't support batch embedding
            batch_embeddings = []
            for text in batch:
                response = self._client.invoke_model(
                    modelId=self.model, body=json.dumps({"inputText": text})
                )
                response_body = json.loads(response["body"].read())
                batch_embeddings.append(response_body["embedding"])

            embeddings.extend(
                [
                    self._process_embedding(embedding, as_buffer, dtype)
                    for embedding in batch_embeddings
                ]
            )

        return embeddings

    @property
    def type(self) -> str:
        return "bedrock"

import base64
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redisvl.utils.vectorize.base import BaseVectorizer

try:
    from PIL import Image
except ImportError:
    _PILLOW_INSTALLED = False
else:
    _PILLOW_INSTALLED = True


class BedrockVectorizer(BaseVectorizer):
    """The BedrockVectorizer class utilizes Amazon Bedrock's API to generate
    embeddings for text or image data.

    This vectorizer is designed to interact with Amazon Bedrock API,
    requiring AWS credentials for authentication. The credentials can be provided
    directly in the `api_config` dictionary or through environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (defaults to us-east-1)

    The vectorizer supports synchronous operations with batch processing and
    preprocessing capabilities.

    You can optionally enable caching to improve performance when generating
    embeddings for repeated inputs.

    .. code-block:: python

        # Basic usage with explicit credentials
        vectorizer = BedrockVectorizer(
            model="amazon.titan-embed-text-v2:0",
            api_config={
                "aws_access_key_id": "your_access_key",
                "aws_secret_access_key": "your_secret_key",
                "aws_region": "us-east-1"
            }
        )

        # With environment variables and caching
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
        cache = EmbeddingsCache(name="bedrock_embeddings_cache")

        vectorizer = BedrockVectorizer(
            model="amazon.titan-embed-text-v2:0",
            cache=cache
        )

        # First call will compute and cache the embedding
        embedding1 = vectorizer.embed("Hello, world!")

        # Second call will retrieve from cache
        embedding2 = vectorizer.embed("Hello, world!")

        # Generate batch embeddings
        embeddings = vectorizer.embed_many(["Hello", "World"], batch_size=2)

        # Multimodal usage
        from pathlib import Path
        vectorizer = BedrockVectorizer(
            model="amazon.titan-embed-image-v1:0",
            api_config={
                "aws_access_key_id": "your_access_key",
                "aws_secret_access_key": "your_secret_key",
                "aws_region": "us-east-1"
            }
        )
        image_embedding = vectorizer.embed(Path("path/to/your/image.jpg"))

        # Embedding a list of mixed modalities
        embeddings = vectorizer.embed_many(
            ["Hello", "world!", Path("path/to/your/image.jpg")],
            batch_size=2
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        api_config: Optional[Dict[str, str]] = None,
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
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
            cache (Optional[EmbeddingsCache]): Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.

        Raises:
            ValueError: If credentials are not provided in config or environment.
            ImportError: If boto3 is not installed.
            ValueError: If an invalid dtype is provided.
        """
        super().__init__(model=model, dtype=dtype, cache=cache)
        # Initialize client and set up the model
        self._setup(api_config, **kwargs)

    def embed_image(self, image_path: str, **kwargs) -> Union[List[float], bytes]:
        """Embed an image (from its path on disk) using a Bedrock multimodal model."""
        return self.embed(Path(image_path), **kwargs)

    def _setup(self, api_config: Optional[Dict], **kwargs):
        """Set up the Bedrock client and determine the embedding dimensions."""
        # Initialize client
        self._initialize_client(api_config, **kwargs)
        # Set model dimensions after initialization
        self.dims = self._set_model_dims()

    def _initialize_client(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the Bedrock client using the provided API keys or
        environment variables.

        Args:
            api_config: Dictionary with AWS credentials and configuration
            **kwargs: Additional arguments to pass to boto3 client

        Raises:
            ImportError: If boto3 is not installed
            ValueError: If AWS credentials are not provided
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

        # Store client as a regular attribute instead of PrivateAttr
        self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
            **kwargs,
        )

    def _set_model_dims(self) -> int:
        """
        Determine the dimensionality of the embedding model by making a test call.

        Returns:
            int: Dimensionality of the embedding model

        Raises:
            ValueError: If embedding dimensions cannot be determined
        """
        try:
            # Call the protected _embed method to avoid caching this test embedding
            embedding = self._embed("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the Bedrock API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed(self, content: Any, **kwargs) -> List[float]:
        """
        Generate a vector embedding for a single input using the AWS Bedrock API.

        Args:
            content: Text or PIL.Image.Image or Path to image-file to embed
            **kwargs: Additional parameters to pass to the AWS Bedrock API

        Returns:
            List[float]: Vector embedding as a list of floats

        Raises:
            TypeError: If content is not a string, Path, or PIL.Image.Image
            ValueError: If attempting to embed an image with a text model
            ValueError: If embedding fails
        """
        from botocore.exceptions import ValidationError

        body = self._serialize_request_body(content)

        try:
            response = self._client.invoke_model(
                modelId=self.model, body=json.dumps(body), **kwargs
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
        except ValidationError as e:
            if "Malformed input request" in str(e) and "inputImage" in body:
                raise ValueError("Attempted to embed image with text model.") from e
            raise ValueError(f"Embedding text failed: {e}")
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(TypeError),
    )
    def _embed_many(
        self, contents: List[Any], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of inputs using the AWS Bedrock API.

        Args:
            contents: List of text/images to embed. Images must be Paths to image-file or PIL.Image.Image
            batch_size: Number of inputs to process in each API call
            **kwargs: Additional parameters to pass to the AWS Bedrock API

        Returns:
            List[List[float]]: List of vector embeddings as lists of floats

        Raises:
            TypeError: If `contents` is not a list
            TypeError: If each item in `contents` is not a string, Path, or PIL.Image.Image
            ValueError: If attempting to embed an image with a text model
            ValueError: If embedding fails
        """
        from botocore.exceptions import ValidationError

        if not isinstance(contents, list):
            raise TypeError("`contents` must be a list")

        try:
            embeddings: List[List[float]] = []

            for batch in self.batchify(contents, batch_size):
                # Process each text in the batch individually since Bedrock
                # doesn't support batch embedding
                batch_embeddings = []
                for content in batch:
                    body = self._serialize_request_body(content)

                    try:
                        response = self._client.invoke_model(
                            modelId=self.model,
                            body=json.dumps(body),
                            **kwargs,
                        )
                    except ValidationError as e:
                        if "Malformed input request" in str(e) and "inputImage" in body:
                            raise ValueError(
                                "Attempted to embed image with text model."
                            ) from e
                        raise e

                    response_body = json.loads(response["body"].read())
                    batch_embeddings.append(response_body["embedding"])

                embeddings.extend(batch_embeddings)

            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")

    def _serialize_request_body(
        self, content: Any
    ) -> dict[Literal["inputText", "inputImage"], str]:
        """Serialize the request body for the Bedrock API."""
        if isinstance(content, str):
            return {"inputText": content}
        elif isinstance(content, Path):
            return {"inputImage": self._b64encode_image(content.read_bytes())}
        elif _PILLOW_INSTALLED and isinstance(content, Image.Image):
            bytes_data = io.BytesIO()
            content.save(bytes_data, format="PNG")
            return {"inputImage": self._b64encode_image(bytes_data.getvalue())}
        raise TypeError(
            "Content must be a string, Path to image-file, or PIL.Image.Image"
        )

    @staticmethod
    def _b64encode_image(bytes_data: bytes) -> str:
        """Encode an image as a base64 string."""
        return base64.b64encode(bytes_data).decode("utf-8")

    @property
    def type(self) -> str:
        return "bedrock"

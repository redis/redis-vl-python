import os, boto3
from typing import Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from redisvl.utils.vectorize.base import BaseVectorizer

class AmazonBedrockTextVectorizer(BaseVectorizer):
    
    """The AmazonTextVectorizer class utilizes Amazon Bedrock's API to generate
    embeddings for text data.

    This vectorizer is designed to interact with Amazon Bedrock API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` & `AWS_REGION`
    environment variable. User must obtain an API key from AWS's website
    (https://aws.amazon.com/bedrock/).

    The vectorizer supports only synchronous operations, allows for batch
    processing of texts and flexibility in handling preprocessing tasks.

    .. code-block:: python

        from redisvl.utils.vectorize import AmazonBedrockTextVectorizer

        vectorizer = AmazonBedrockTextVectorizer(
            model_id="amazon.titan-embed-text-v1", 
            api_config = { "aws_access_key_id": "your_aws_access_id", "aws_secret_access_key": "your_aws_secret_key","region_name": "us-east-1"} # OR set in your env
        )

    """

    def __init__(
        self, model_id: str = "amazon.titan-embed-text-v1", api_config: Optional[Dict] = None
    ):
        """Initialize the AWS Bedrock Vectorizer

        Args:
            model(str): Model to use for embedding. Defaults to amazon.titan-embed-text-v1.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.
        
        """
        # Set up AWS credentials
        aws_access_key_id = api_config.get("aws_access_key_id") if api_config else os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = api_config.get("aws_secret_access_key") if api_config else os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = api_config.get("region_name", "us-east-1") if api_config else os.getenv("AWS_REGION", "us-east-1")
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS access key and secret key are required."
                "Provide it in the api_config or set the environment varibales."
                )
        
        # Initialize Bedrock client
        client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        dims = self._set_model_dims(model=model_id, client=client)
        super().__init__(model=model_id, dims=dims, client=client)

    @staticmethod
    def _set_model_dims(client, model) -> int:
        try:
            bedrock_em_model = BedrockEmbeddings(model_id=model,client=client)
            embedding = bedrock_em_model.embed_query("dimension test")
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the AWS Bedrock API: {str(ke)}")
        except Exception as e:  # pylint: disable=broad-except
            # fall back (TODO get more specific)
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
        return len(embedding)
    
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
        if not isinstance(text, str):
            raise TypeError("Must pass in a str value to embed.")

        if preprocess:
            text = preprocess(text)
        
        bedrock_em_model = BedrockEmbeddings(model_id=self.model,client=self.client)

        embedding = bedrock_em_model.embed_query(text)

        return self._process_embedding(embedding, as_buffer)
    
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
        input_type = kwargs.get("input_type")

        if not isinstance(texts, list):
            raise TypeError("Must pass in a list of str values to embed.")
        if len(texts) > 0 and not isinstance(texts[0], str):
            raise TypeError("Must pass in a list of str values to embed.")
        if not isinstance(input_type, str):
            raise TypeError(
                "Must pass in a str value for AWS Bedrock embedding input_type.\
                    See AWS" 
            )

        embeddings: List = []

        bedrock_em_model = BedrockEmbeddings(model_id=self.model,client=self.client)

        for batch in self.batchify(texts, batch_size, preprocess):


            response = bedrock_em_model.embed_query(texts=batch)

            embeddings += [
                self._process_embedding(embedding, as_buffer)
                for embedding in response.embeddings
            ]

        return embeddings

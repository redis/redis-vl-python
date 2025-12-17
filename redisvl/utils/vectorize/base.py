import io
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.redis.utils import array_to_buffer
from redisvl.schema.fields import VectorDataType
from redisvl.utils.utils import deprecated_argument

try:
    from PIL.Image import Image
except ImportError:
    _PILLOW_INSTALLED = False
else:
    _PILLOW_INSTALLED = True

logger = logging.getLogger(__name__)


class Vectorizers(Enum):
    azure_openai = "azure_openai"
    openai = "openai"
    cohere = "cohere"
    mistral = "mistral"
    vertexai = "vertexai"
    hf = "hf"
    voyageai = "voyageai"


class BaseVectorizer(BaseModel):
    """Base RedisVL vectorizer interface.

    This class defines the interface for vectorization with an optional
    caching layer to improve performance by avoiding redundant API calls.

    Attributes:
        model: The name of the embedding model.
        dtype: The data type of the embeddings, defaults to "float32".
        dims: The dimensionality of the vectors.
        cache: Optional embedding cache to store and retrieve embeddings.
    """

    model: str
    dtype: str = "float32"
    dims: Annotated[Optional[int], Field(strict=True, gt=0)] = None
    cache: Optional[EmbeddingsCache] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def type(self) -> str:
        """Return the type of vectorizer."""
        return "base"

    @field_validator("dtype")
    @classmethod
    def check_dtype(cls, dtype):
        """Validate the data type is supported."""
        try:
            VectorDataType(dtype.upper())
        except ValueError:
            raise ValueError(
                f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
            )
        return dtype

    @deprecated_argument("text", "content")
    def embed(
        self,
        content: Any = None,
        text: Any = None,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        skip_cache: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Generate a vector embedding for content.

        Args:
            content: The content to convert to a vector embedding
            text: The text to convert to a vector embedding (deprecated - use `content` instead)
            preprocess: Function to apply to the content before embedding
            as_buffer: Return the embedding as a binary buffer instead of a list
            skip_cache: Bypass the cache for this request
            **kwargs: Additional model-specific parameters

        Returns:
            The vector embedding as either a list of floats or binary buffer

        Examples:
            >>> embedding = text_vectorizer.embed("Hello world")
            >>> embedding = image_vectorizer.embed(Image.open("test.png"))
        """
        content = content or text
        if not content:
            raise ValueError("No content provided to embed.")

        # Apply preprocessing if provided
        if preprocess is not None:
            content = preprocess(content)

        # Check cache if available and not skipped
        if self.cache is not None and not skip_cache:
            try:
                cache_result = self.cache.get(
                    content=self._serialize_for_cache(content), model_name=self.model
                )
                if cache_result:
                    logger.debug(f"Cache hit for content with model {self.model}")
                    return self._process_embedding(
                        cache_result["embedding"], as_buffer, self.dtype
                    )
            except Exception as e:
                logger.warning(f"Error accessing embedding cache: {str(e)}")

        # Generate embedding using provider-specific implementation
        cache_metadata = kwargs.pop("metadata", {})
        embedding = self._embed(content, **kwargs)

        # Store in cache if available and not skipped
        if self.cache is not None and not skip_cache:
            try:
                self.cache.set(
                    content=self._serialize_for_cache(content),
                    model_name=self.model,
                    embedding=embedding,
                    metadata=cache_metadata,
                )
            except Exception as e:
                logger.warning(f"Error storing in embedding cache: {str(e)}")

        # Process and return result
        return self._process_embedding(embedding, as_buffer, self.dtype)

    @deprecated_argument("texts", "contents")
    def embed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        skip_cache: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Generate vector embeddings for multiple items efficiently.

        Args:
            contents: List of content to convert to vector embeddings
            texts: List of texts to convert to vector embeddings (deprecated - use `contents` instead)
            preprocess: Function to apply to each item before embedding
            batch_size: Number of items to process in each API call
            as_buffer: Return embeddings as binary buffers instead of lists
            skip_cache: Bypass the cache for this request
            **kwargs: Additional model-specific parameters

        Returns:
            List of vector embeddings in the same order as the inputs

        Examples:
            >>> embeddings = vectorizer.embed_many(["Hello", "World"], batch_size=2)
        """
        contents = contents or texts
        if not contents:
            return []

        # Apply preprocessing if provided
        if preprocess is not None:
            processed_contents = [preprocess(item) for item in contents]
        else:
            processed_contents = contents

        # Get cached embeddings and identify misses
        results, cache_misses, cache_miss_indices = self._get_from_cache_batch(
            processed_contents, skip_cache
        )

        # Generate embeddings for cache misses
        if cache_misses:
            cache_metadata = kwargs.pop("metadata", {})
            new_embeddings = self._embed_many(
                contents=cache_misses, batch_size=batch_size, **kwargs
            )

            # Store new embeddings in cache
            self._store_in_cache_batch(
                cache_misses, new_embeddings, cache_metadata, skip_cache
            )

            # Insert new embeddings into results array
            for idx, embedding in zip(cache_miss_indices, new_embeddings):
                results[idx] = embedding

        # Process and return results
        return [self._process_embedding(emb, as_buffer, self.dtype) for emb in results]

    @deprecated_argument("text", "content")
    async def aembed(
        self,
        content: Any = None,
        text: Any = None,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        skip_cache: bool = False,
        **kwargs,
    ) -> Union[List[float], bytes]:
        """Asynchronously generate a vector embedding for an item of content.

        Args:
            content: The content to convert to a vector embedding
            text: The text to convert to a vector embedding (deprecated - use `content` instead)
            preprocess: Function to apply to the content before embedding
            as_buffer: Return the embedding as a binary buffer instead of a list
            skip_cache: Bypass the cache for this request
            **kwargs: Additional model-specific parameters

        Returns:
            The vector embedding as either a list of floats or binary buffer

        Examples:
            >>> embedding = await vectorizer.aembed("Hello world")
        """
        content = content or text
        if not content:
            raise ValueError("No content provided to embed.")

        # Apply preprocessing if provided
        if preprocess is not None:
            content = preprocess(content)

        # Check cache if available and not skipped
        if self.cache is not None and not skip_cache:
            try:
                cache_result = await self.cache.aget(
                    content=self._serialize_for_cache(content), model_name=self.model
                )
                if cache_result:
                    logger.debug(f"Async cache hit for content with model {self.model}")
                    return self._process_embedding(
                        cache_result["embedding"], as_buffer, self.dtype
                    )
            except Exception as e:
                logger.warning(
                    f"Error accessing embedding cache asynchronously: {str(e)}"
                )

        # Generate embedding using provider-specific implementation
        cache_metadata = kwargs.pop("metadata", {})
        embedding = await self._aembed(content, **kwargs)

        # Store in cache if available and not skipped
        if self.cache is not None and not skip_cache:
            try:
                await self.cache.aset(
                    content=self._serialize_for_cache(content),
                    model_name=self.model,
                    embedding=embedding,
                    metadata=cache_metadata,
                )
            except Exception as e:
                logger.warning(
                    f"Error storing in embedding cache asynchronously: {str(e)}"
                )

        # Process and return result
        return self._process_embedding(embedding, as_buffer, self.dtype)

    @deprecated_argument("texts", "contents")
    async def aembed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        preprocess: Optional[Callable] = None,
        batch_size: int = 10,
        as_buffer: bool = False,
        skip_cache: bool = False,
        **kwargs,
    ) -> Union[List[List[float]], List[bytes]]:
        """Asynchronously generate vector embeddings for multiple items efficiently.

        Args:
            contents: List of content to convert to vector embeddings
            texts: List of texts to convert to vector embeddings (deprecated - use `contents` instead)
            preprocess: Function to apply to each item before embedding
            batch_size: Number of texts to process in each API call
            as_buffer: Return embeddings as binary buffers instead of lists
            skip_cache: Bypass the cache for this request
            **kwargs: Additional model-specific parameters

        Returns:
            List of vector embeddings in the same order as the inputs

        Examples:
            >>> embeddings = await vectorizer.aembed_many(["Hello", "World"], batch_size=2)
        """
        contents = contents or texts
        if not contents:
            return []

        # Apply preprocessing if provided
        if preprocess is not None:
            processed_contents = [preprocess(item) for item in contents]
        else:
            processed_contents = contents

        # Get cached embeddings and identify misses
        results, cache_misses, cache_miss_indices = await self._aget_from_cache_batch(
            processed_contents, skip_cache
        )

        # Generate embeddings for cache misses
        if cache_misses:
            cache_metadata = kwargs.pop("metadata", {})
            new_embeddings = await self._aembed_many(
                contents=cache_misses, batch_size=batch_size, **kwargs
            )

            # Store new embeddings in cache
            await self._astore_in_cache_batch(
                cache_misses, new_embeddings, cache_metadata, skip_cache
            )

            # Insert new embeddings into results array
            for idx, embedding in zip(cache_miss_indices, new_embeddings):
                results[idx] = embedding

        # Process and return results
        return [self._process_embedding(emb, as_buffer, self.dtype) for emb in results]

    @deprecated_argument("text", "content")
    def _embed(self, text: Any = "", content: Any = "", **kwargs) -> List[float]:
        """Generate a vector embedding for a single item."""
        raise NotImplementedError

    @deprecated_argument("texts", "contents")
    def _embed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        batch_size: int = 10,
        **kwargs,
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of items."""
        raise NotImplementedError

    @deprecated_argument("text", "content")
    async def _aembed(self, content: Any = "", text: Any = "", **kwargs) -> List[float]:
        """Asynchronously generate a vector embedding for a single item."""
        logger.warning(
            "This vectorizer has no async embed method. Falling back to sync."
        )
        return self._embed(content=content or text, **kwargs)

    @deprecated_argument("texts", "contents")
    async def _aembed_many(
        self,
        contents: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        batch_size: int = 10,
        **kwargs,
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings for a batch of items."""
        logger.warning(
            "This vectorizer has no async embed_many method. Falling back to sync."
        )
        return self._embed_many(
            contents=contents or texts, batch_size=batch_size, **kwargs
        )

    def _get_from_cache_batch(
        self, contents: List[Any], skip_cache: bool
    ) -> tuple[List[Optional[List[float]]], List[str], List[int]]:
        """Get vector embeddings from cache and track cache misses.

        Args:
            contents: List of content to get from cache
            skip_cache: Whether to skip cache lookup

        Returns:
            Tuple of (results, cache_misses, cache_miss_indices)
        """
        results = [None] * len(contents)
        cache_misses = []
        cache_miss_indices = []

        # Skip cache if requested or no cache available
        if skip_cache or self.cache is None:
            return results, contents, list(range(len(contents)))  # type: ignore

        try:
            # Efficient batch cache lookup
            cache_results = self.cache.mget(
                contents=(self._serialize_for_cache(c) for c in contents),
                model_name=self.model,
            )

            # Process cache hits and collect misses
            for i, (content, cache_result) in enumerate(zip(contents, cache_results)):
                if cache_result:
                    results[i] = cache_result["embedding"]
                else:
                    cache_misses.append(content)
                    cache_miss_indices.append(i)

            logger.debug(
                f"Cache hits: {len(contents) - len(cache_misses)}, misses: {len(cache_misses)}"
            )
        except Exception as e:
            logger.warning(f"Error accessing embedding cache in batch: {str(e)}")
            # On cache error, process all data
            cache_misses = contents
            cache_miss_indices = list(range(len(contents)))

        return results, cache_misses, cache_miss_indices  # type: ignore

    async def _aget_from_cache_batch(
        self, contents: List[Any], skip_cache: bool
    ) -> tuple[List[Optional[List[float]]], List[str], List[int]]:
        """Asynchronously get vector embeddings from cache and track cache misses.

        Args:
            contents: List of content to get from cache
            skip_cache: Whether to skip cache lookup

        Returns:
            Tuple of (results, cache_misses, cache_miss_indices)
        """
        results = [None] * len(contents)
        cache_misses = []
        cache_miss_indices = []

        # Skip cache if requested or no cache available
        if skip_cache or self.cache is None:
            return results, contents, list(range(len(contents)))  # type: ignore

        try:
            # Efficient batch cache lookup
            cache_results = await self.cache.amget(
                contents=(self._serialize_for_cache(c) for c in contents),
                model_name=self.model,
            )

            # Process cache hits and collect misses
            for i, (content, cache_result) in enumerate(zip(contents, cache_results)):
                if cache_result:
                    results[i] = cache_result["embedding"]
                else:
                    cache_misses.append(content)
                    cache_miss_indices.append(i)

            logger.debug(
                f"Async cache hits: {len(contents) - len(cache_misses)}, misses: {len(cache_misses)}"
            )
        except Exception as e:
            logger.warning(
                f"Error accessing embedding cache in batch asynchronously: {str(e)}"
            )
            # On cache error, process all data
            cache_misses = contents
            cache_miss_indices = list(range(len(contents)))

        return results, cache_misses, cache_miss_indices  # type: ignore

    def _store_in_cache_batch(
        self,
        contents: List[Any],
        embeddings: List[List[float]],
        metadata: Dict,
        skip_cache: bool,
    ) -> None:
        """Store a batch of vector embeddings in the cache.

        Args:
            contents: List of content that was embedded
            embeddings: List of vector embeddings
            metadata: Metadata to store with the embeddings
            skip_cache: Whether to skip cache storage
        """
        if skip_cache or self.cache is None:
            return

        try:
            # Prepare batch cache storage items
            cache_items = [
                {
                    "content": self._serialize_for_cache(content),
                    "model_name": self.model,
                    "embedding": emb,
                    "metadata": metadata,
                }
                for content, emb in zip(contents, embeddings)
            ]
            self.cache.mset(items=cache_items)
        except Exception as e:
            logger.warning(f"Error storing batch in embedding cache: {str(e)}")

    async def _astore_in_cache_batch(
        self,
        contents: List[Any],
        embeddings: List[List[float]],
        metadata: Dict,
        skip_cache: bool,
    ) -> None:
        """Asynchronously store a batch of vector embeddings in the cache.

        Args:
            contents: List of content that was embedded
            embeddings: List of vector embeddings
            metadata: Metadata to store with the embeddings
            skip_cache: Whether to skip cache storage
        """
        if skip_cache or self.cache is None:
            return

        try:
            # Prepare batch cache storage items
            cache_items = [
                {
                    "content": self._serialize_for_cache(content),
                    "model_name": self.model,
                    "embedding": emb,
                    "metadata": metadata,
                }
                for content, emb in zip(contents, embeddings)
            ]
            await self.cache.amset(items=cache_items)
        except Exception as e:
            logger.warning(
                f"Error storing batch in embedding cache asynchronously: {str(e)}"
            )

    def batchify(self, seq: list, size: int, preprocess: Optional[Callable] = None):
        """Split a sequence into batches of specified size.

        Args:
            seq: Sequence to split into batches
            size: Batch size
            preprocess: Optional function to preprocess each item

        Yields:
            Batches of the sequence
        """
        for pos in range(0, len(seq), size):
            if preprocess is not None:
                yield [preprocess(chunk) for chunk in seq[pos : pos + size]]
            else:
                yield seq[pos : pos + size]

    def _process_embedding(
        self, embedding: Optional[List[float]], as_buffer: bool, dtype: str
    ):
        """Process the vector embedding format based on the as_buffer flag."""
        if embedding is not None:
            if as_buffer:
                return array_to_buffer(embedding, dtype)
        return embedding

    def _serialize_for_cache(self, content: Any) -> Union[bytes, str]:
        """Convert content to a cacheable format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, bytes):
            return content
        elif isinstance(content, Path):
            return content.read_bytes()
        elif _PILLOW_INSTALLED and isinstance(content, Image):
            buffer = io.BytesIO()
            content.save(buffer, format="PNG")
            return buffer.getvalue()

        raise NotImplementedError(
            f"Content type {type(content)} is not supported for caching."
        )

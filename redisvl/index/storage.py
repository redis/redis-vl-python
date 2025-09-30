from collections.abc import Collection
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from pydantic import BaseModel, ValidationError
from redis import __version__ as redis_version

# Add imports for Pipeline types
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline

# Redis 5.x compatibility (6 fixed the import path)
if redis_version.startswith("5"):
    from redis.commands.search.indexDefinition import (  # type: ignore[import-untyped]
        IndexType,
    )
else:
    from redis.commands.search.index_definition import (  # type: ignore[no-redef]
        IndexType,
    )

import json

from redisvl.exceptions import SchemaValidationError
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema
from redisvl.schema.validation import validate_object
from redisvl.types import (
    AsyncRedisClient,
    AsyncRedisClientOrPipeline,
    AsyncRedisPipeline,
    RedisClientOrPipeline,
    SyncRedisClient,
    SyncRedisPipeline,
)
from redisvl.utils.log import get_logger
from redisvl.utils.utils import create_ulid

logger = get_logger(__name__)


class BaseStorage(BaseModel):
    """
    Base class for internal storage handling in Redis.

    Provides foundational methods for key management, data preprocessing,
    validation, and basic read/write operations (both sync and async).
    """

    type: IndexType
    """Type of index used in storage"""
    index_schema: IndexSchema
    """Index schema definition"""
    default_batch_size: int = 200
    """Default size for batch operations"""

    @staticmethod
    def _key(id: str, prefix: str, key_separator: str) -> str:
        """Create a Redis key using a combination of a prefix, separator, and
        the identifider.

        Args:
            id (str): The unique identifier for the Redis entry.
            prefix (str): A prefix to append before the key value.
            key_separator (str): A separator to insert between prefix
                and key value.

        Returns:
            str: The fully formed Redis key.
        """
        if not prefix:
            return id
        else:
            # Normalize prefix by removing trailing separators to avoid doubles
            normalized_prefix = (
                prefix.rstrip(key_separator) if key_separator else prefix
            )
            if normalized_prefix:
                return f"{normalized_prefix}{key_separator}{id}"
            else:
                # If prefix was only separators, just return the id
                return id

    def _create_key(self, obj: Dict[str, Any], id_field: Optional[str] = None) -> str:
        """Construct a Redis key for a given object, optionally using a
        specified field from the object as the key.

        Args:
            obj (Dict[str, Any]): The object from which to construct the key.
            id_field (Optional[str], optional): The field to use as the
                key, if provided.

        Returns:
            str: The constructed Redis key for the object.

        Raises:
            ValueError: If the id_field is not found in the object.
        """
        if id_field is None:
            key_value = create_ulid()
        else:
            try:
                key_value = obj[id_field]  # type: ignore
            except KeyError:
                raise ValueError(f"Key field {id_field} not found in record {obj}")

        # Normalize prefix: use first prefix if multiple are configured
        prefix = self.index_schema.index.prefix
        normalized_prefix = prefix[0] if isinstance(prefix, list) else prefix

        return self._key(
            key_value,
            prefix=normalized_prefix,
            key_separator=self.index_schema.index.key_separator,
        )

    @staticmethod
    def _preprocess(obj: Any, preprocess: Optional[Callable] = None) -> Dict[str, Any]:
        """Apply a preprocessing function to the object if provided.

        Args:
            preprocess (Optional[Callable], optional): Function to
                process the object.
            obj (Any): Object to preprocess.

        Returns:
            Dict[str, Any]: Processed object as a dictionary.
        """
        # optionally preprocess object
        if preprocess:
            obj = preprocess(obj)
        return obj

    @staticmethod
    def _set(
        client: RedisClientOrPipeline, key: str, obj: Dict[str, Any]
    ) -> Union[SyncRedisPipeline, Dict[str, Any]]:
        """Synchronously set the value in Redis for the given key.

        Args:
            client (RedisClientOrPipeline): The Redis client instance.
            key (str): The key under which to store the object.
            obj (Dict[str, Any]): The object to store in Redis.
        """
        raise NotImplementedError

    @staticmethod
    async def _aset(
        client: AsyncRedisClientOrPipeline, key: str, obj: Dict[str, Any]
    ) -> Union[AsyncRedisPipeline, Dict[str, Any]]:
        """Asynchronously set data in Redis using the provided client or pipeline."""
        raise NotImplementedError

    @staticmethod
    def _get(
        client: RedisClientOrPipeline, key: str
    ) -> Union[SyncRedisPipeline, Dict[str, Any]]:
        """Synchronously get data from Redis using the provided client or pipeline."""
        raise NotImplementedError

    @staticmethod
    async def _aget(
        client: AsyncRedisClientOrPipeline, key: str
    ) -> Union[AsyncRedisPipeline, Dict[str, Any]]:
        """Asynchronously get data from Redis using the provided client or pipeline."""
        raise NotImplementedError

    def _validate(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an object against the schema using Pydantic-based validation.

        Args:
            obj: The object to validate

        Returns:
            Validated object with any type coercions applied

        Raises:
            ValueError: If validation fails
        """
        # Pass directly to validation function and let any errors propagate
        return validate_object(self.index_schema, obj)

    def _get_keys(
        self,
        objects: List[Any],
        keys: Optional[Iterable[str]] = None,
        id_field: Optional[str] = None,
    ) -> List[str]:
        """Generate Redis keys for a list of objects."""
        generated_keys: List[str] = []
        keys_iterator = iter(keys) if keys else None

        if keys and len(list(keys)) != len(objects):
            raise ValueError(
                "Length of provided keys does not match the length of objects."
            )

        for obj in objects:
            if keys_iterator:
                key = next(keys_iterator)
            else:
                key = self._create_key(obj, id_field)
            generated_keys.append(key)
        return generated_keys

    def _create_readable_validation_error_message(
        self, validation_error: ValidationError, obj_index: int, obj: Dict[str, Any]
    ) -> str:
        """
        Create a human-readable error message from a Pydantic ValidationError.

        Args:
            validation_error: The Pydantic ValidationError
            obj_index: The index of the object that failed validation
            obj: The object that failed validation

        Returns:
            A detailed, actionable error message
        """
        error_details = []

        for error in validation_error.errors():
            field_name = ".".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            error_msg = error["msg"]
            input_value = error.get("input", "N/A")

            # Create a more descriptive error message based on error type
            if error_type == "bytes_type":
                if isinstance(input_value, bool):
                    suggestion = (
                        f"Field '{field_name}' expects bytes (vector data), but got boolean value '{input_value}'. "
                        f"If this should be a vector field, provide a list of numbers or bytes. "
                        f"If this should be a different field type, check your schema definition."
                    )
                else:
                    suggestion = (
                        f"Field '{field_name}' expects bytes (vector data), but got {type(input_value).__name__} value '{input_value}'. "
                        f"For vector fields, provide a list of numbers or bytes."
                    )
            elif error_type == "bool_type":
                suggestion = (
                    f"Field '{field_name}' cannot be boolean. Got '{input_value}' of type {type(input_value).__name__}. "
                    f"Provide a valid numeric value instead."
                )
            elif error_type == "string_type":
                suggestion = (
                    f"Field '{field_name}' expects a string, but got {type(input_value).__name__} value '{input_value}'. "
                    f"Convert the value to a string or check your data types."
                )
            elif error_type == "list_type":
                suggestion = (
                    f"Field '{field_name}' expects a list (for vector data), but got {type(input_value).__name__} value '{input_value}'. "
                    f"Provide the vector as a list of numbers."
                )
            elif "dimensions" in error_msg.lower():
                suggestion = (
                    f"Vector field '{field_name}' has incorrect dimensions. {error_msg}"
                )
            elif "range" in error_msg.lower():
                suggestion = f"Vector field '{field_name}' has values outside the allowed range. {error_msg}"
            else:
                suggestion = f"Field '{field_name}': {error_msg}"

            error_details.append(f"  • {suggestion}")

        # Create the final error message
        if len(error_details) == 1:
            detail_msg = error_details[0].strip("  • ")
        else:
            detail_msg = "Multiple validation errors:\n" + "\n".join(error_details)

        return (
            f"Schema validation failed for object at index {obj_index}. {detail_msg}\n"
            f"Object data: {json.dumps(obj, default=str, indent=2)[:200]}{'...' if len(str(obj)) > 200 else ''}\n"
            f"Hint: Check that your data types match the schema field definitions. "
            f"Use index.schema.fields to view expected field types."
        )

    def _preprocess_and_validate_objects(
        self,
        objects: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        preprocess: Optional[Callable] = None,
        validate: bool = False,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Preprocess and validate a list of objects with fail-fast approach.

        Args:
            objects: List of objects to preprocess and validate
            id_field: Field to use as the key
            keys: Optional iterable of keys
            preprocess: Optional preprocessing function
            validate: Whether to validate against schema

        Returns:
            List of tuples (key, processed_obj) for valid objects

        Raises:
            SchemaValidationError: If validation fails, with context about which object failed
            ValueError: If any other processing errors occur
        """
        prepared_objects = []
        keys_iterator = iter(keys) if keys else None

        for i, obj in enumerate(objects):
            try:
                # Generate key
                key = (
                    next(keys_iterator)
                    if keys_iterator
                    else self._create_key(obj, id_field)
                )

                # Preprocess
                processed_obj = self._preprocess(obj, preprocess)

                # Schema validation if enabled
                if validate:
                    processed_obj = self._validate(processed_obj)

                # Store valid object with its key for writing
                prepared_objects.append((key, processed_obj))

            except ValidationError as e:
                # Create detailed, readable error message
                detailed_message = self._create_readable_validation_error_message(
                    e, i, obj
                )
                raise SchemaValidationError(detailed_message) from e
            except Exception as e:
                # Capture other exceptions with context
                object_id = f"at index {i}"
                raise ValueError(
                    f"Error processing object {object_id}: {str(e)}"
                ) from e

        return prepared_objects

    def write(
        self,
        redis_client: SyncRedisClient,
        objects: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        validate: bool = False,
    ) -> List[str]:
        """Write a batch of objects to Redis as hash entries. This method
        returns a list of Redis keys written to the database.

        Args:
            redis_client (RedisClient): A Redis client used for writing data.
            objects (Iterable[Any]): An iterable of objects to store.
            id_field (Optional[str], optional): Field used as the key for
                each object. Defaults to None.
            keys (Optional[Iterable[str]], optional): Optional iterable of
                keys, must match the length of objects if provided.
            ttl (Optional[int], optional): Time-to-live in seconds for each
                key. Defaults to None.
            preprocess (Optional[Callable], optional): A function to preprocess
                objects before storage. Defaults to None.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution.
            validate (bool, optional): Whether to validate objects against schema.
                Defaults to False.

        Raises:
            ValueError: If the length of provided keys does not match the
                length of objects, or if validation fails.
        """
        if keys and len(keys) != len(objects):  # type: ignore
            raise ValueError("Length of keys does not match the length of objects")

        if batch_size is None:
            batch_size = self.default_batch_size

        if not objects:
            return []

        # Pass 1: Preprocess and validate all objects
        prepared_objects = self._preprocess_and_validate_objects(
            list(objects),  # Convert Iterable to List
            id_field=id_field,
            keys=keys,
            preprocess=preprocess,
            validate=validate,
        )

        # Pass 2: Write all valid objects in batches
        added_keys = []

        with redis_client.pipeline(transaction=False) as pipe:
            for i, (key, obj) in enumerate(prepared_objects, start=1):
                self._set(pipe, key, obj)

                # Set TTL if provided
                if ttl:
                    pipe.expire(key, ttl)

                added_keys.append(key)

                # Execute in batches
                if i % batch_size == 0:
                    pipe.execute()

            # Execute any remaining commands
            if len(prepared_objects) % batch_size != 0:
                pipe.execute()

        return added_keys

    async def awrite(
        self,
        redis_client: AsyncRedisClient,
        objects: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        batch_size: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        validate: bool = False,
    ) -> List[str]:
        """Asynchronously write objects to Redis as hash entries using pipeline batching.
        The method returns a list of keys written to the database.

        Args:
            redis_client (AsyncRedisClient): An asynchronous Redis client used
                for writing data.
            objects (Iterable[Any]): An iterable of objects to store.
            id_field (Optional[str], optional): Field used as the key for each
                object. Defaults to None.
            keys (Optional[Iterable[str]], optional): Optional iterable of keys.
                Must match the length of objects if provided.
            ttl (Optional[int], optional): Time-to-live in seconds for each key.
                Defaults to None.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution.
            preprocess (Optional[Callable], optional): An async function to
                preprocess objects before storage. Defaults to None.
            validate (bool, optional): Whether to validate objects against schema.
                Defaults to False.

        Returns:
            List[str]: List of Redis keys loaded to the databases.

        Raises:
            ValueError: If the length of provided keys does not match the
                length of objects, or if validation fails.
        """
        if keys and len(keys) != len(objects):  # type: ignore
            raise ValueError("Length of keys does not match the length of objects")

        if batch_size is None:
            batch_size = self.default_batch_size

        if not objects:
            return []

        # Pass 1: Preprocess and validate all objects
        prepared_objects = self._preprocess_and_validate_objects(
            list(objects),  # Convert Iterable to List
            id_field=id_field,
            keys=keys,
            preprocess=preprocess,
            validate=validate,
        )

        # Pass 2: Write all valid objects in batches using pipeline
        added_keys = []

        async with redis_client.pipeline(transaction=False) as pipe:
            for i, (key, obj) in enumerate(prepared_objects, start=1):
                await self._aset(pipe, key, obj)

                # Set TTL if provided
                if ttl:
                    await pipe.expire(key, ttl)

                added_keys.append(key)

                # Execute in batches
                if i % batch_size == 0:
                    await pipe.execute()

            # Execute any remaining commands
            if len(prepared_objects) % batch_size != 0:
                await pipe.execute()

        return added_keys

    def get(
        self,
        redis_client: SyncRedisClient,
        keys: Collection[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve objects from Redis by keys.

        Args:
            redis_client (SyncRedisClient): Synchronous Redis client.
            keys (Collection[str]): Keys to retrieve from Redis.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            List[Dict[str, Any]]: List of objects pulled from redis.
        """
        results: List = []

        if not isinstance(keys, Collection):
            raise TypeError("Keys must be a collection of strings")

        if len(keys) == 0:
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        # Use a pipeline to batch the retrieval
        with redis_client.pipeline(transaction=False) as pipe:
            for i, key in enumerate(keys, start=1):
                self._get(pipe, key)
                if i % batch_size == 0:
                    results.extend(pipe.execute())
            if i % batch_size != 0:
                results.extend(pipe.execute())

        # Process results
        return convert_bytes(results)

    async def aget(
        self,
        redis_client: AsyncRedisClient,
        keys: Collection[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve objects from Redis by keys.

        Args:
            redis_client (AsyncRedisClient): Asynchronous Redis client.
            keys (Collection[str]): Keys to retrieve from Redis.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            Dict[str, Any]: Dictionary with keys and their corresponding
                objects.
        """
        results: List = []

        if not isinstance(keys, Collection):
            raise TypeError("Keys must be a collection of strings")

        if len(keys) == 0:
            return []

        if batch_size is None:
            batch_size = self.default_batch_size

        # Use a pipeline to batch the retrieval
        async with redis_client.pipeline(transaction=False) as pipe:
            for i, key in enumerate(keys, start=1):
                await self._aget(pipe, key)
                if i % batch_size == 0:
                    results.extend(await pipe.execute())
            if i % batch_size != 0:
                results.extend(await pipe.execute())

        # Process results
        return convert_bytes(results)


class HashStorage(BaseStorage):
    """
    Internal subclass of BaseStorage for the Redis hash data type.

    Implements hash-specific logic for validation and read/write operations
    (both sync and async) in Redis.
    """

    type: IndexType = IndexType.HASH
    """Hash data type for the index"""

    @staticmethod
    def _set(client: RedisClientOrPipeline, key: str, obj: Dict[str, Any]):
        """Synchronously set a hash value in Redis for the given key.

        Args:
            client (SyncRedisClient): The Redis client instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        client.hset(name=key, mapping=obj)

    @staticmethod
    async def _aset(client: AsyncRedisClientOrPipeline, key: str, obj: Dict[str, Any]):
        """Asynchronously set a hash value in Redis for the given key.

        Args:
            client (AsyncClientOrPipeline): The async Redis client or pipeline instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        if isinstance(client, (AsyncPipeline, AsyncClusterPipeline)):
            client.hset(name=key, mapping=obj)  # type: ignore
        else:
            await client.hset(name=key, mapping=obj)  # type: ignore

    @staticmethod
    def _get(client: SyncRedisClient, key: str) -> Dict[str, Any]:
        """Synchronously retrieve a hash value from Redis for the given key.

        Args:
            client (SyncRedisClient): The Redis client instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        return client.hgetall(key)  # type: ignore

    @staticmethod
    async def _aget(
        client: AsyncRedisClientOrPipeline, key: str
    ) -> Union[AsyncRedisPipeline, Dict[str, Any]]:
        """Asynchronously retrieve a hash value from Redis for the given key.

        Args:
            client (AsyncRedisClient): The async Redis client or pipeline instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        if isinstance(client, (AsyncPipeline, AsyncClusterPipeline)):
            return client.hgetall(key)  # type: ignore[return-value]
        else:
            return await client.hgetall(key)  # type: ignore[return-value, misc]


class JsonStorage(BaseStorage):
    """
    Internal subclass of BaseStorage for the Redis JSON data type.

    Implements json-specific logic for validation and read/write operations
    (both sync and async) in Redis.
    """

    type: IndexType = IndexType.JSON
    """JSON data type for the index"""

    @staticmethod
    def _set(client: RedisClientOrPipeline, key: str, obj: Dict[str, Any]):
        """Synchronously set a JSON obj in Redis for the given key.

        Args:
            client (SyncRedisClient): The Redis client instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        client.json().set(key, "$", obj)

    @staticmethod
    async def _aset(client: AsyncRedisClientOrPipeline, key: str, obj: Dict[str, Any]):
        """Asynchronously set a JSON obj in Redis for the given key.

        Args:
            client (AsyncClientOrPipeline): The async Redis client or pipeline instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        if isinstance(client, (AsyncPipeline, AsyncClusterPipeline)):
            client.json().set(key, "$", obj)  # type: ignore[return-value, misc]
        else:
            await client.json().set(key, "$", obj)  # type: ignore[return-value, misc]

    @staticmethod
    def _get(client: RedisClientOrPipeline, key: str) -> Dict[str, Any]:
        """Synchronously retrieve a JSON obj from Redis for the given key.

        Args:
            client (SyncRedisClient): The Redis client instance.
            key (str): The key for which to retrieve the JSON obj.

        Returns:
            Dict[str, Any]: The retrieved JSON obj from Redis.
        """
        return client.json().get(key)  # type: ignore[return-value, misc]

    @staticmethod
    async def _aget(client: AsyncRedisClientOrPipeline, key: str) -> Dict[str, Any]:
        """Asynchronously retrieve a JSON object from Redis for the given key.

        Args:
            client (AsyncRedisClient): The async Redis client or pipeline instance.
            key (str): The key for which to retrieve the JSON object.

        Returns:
            Dict[str, Any]: The retrieved JSON object from Redis.
        """
        if isinstance(client, (AsyncPipeline, AsyncClusterPipeline)):
            return client.json().get(key)  # type: ignore[return-value, misc]
        else:
            return await client.json().get(key)  # type: ignore[return-value, misc]

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ValidationError
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.indexDefinition import IndexType

from redisvl.exceptions import SchemaValidationError
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema
from redisvl.schema.validation import validate_object
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
            return f"{prefix}{key_separator}{id}"

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

        return self._key(
            key_value,
            prefix=self.index_schema.index.prefix,
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
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """Synchronously set the value in Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key under which to store the object.
            obj (Dict[str, Any]): The object to store in Redis.
        """
        raise NotImplementedError

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """Asynchronously set the value in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the object.
            obj (Dict[str, Any]): The object to store in Redis.
        """
        raise NotImplementedError

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """Synchronously get the value from Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key for which to retrieve the object.

        Returns:
            Dict[str, Any]: The retrieved object from Redis.
        """
        raise NotImplementedError

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """Asynchronously get the value from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the object.

        Returns:
            Dict[str, Any]: The retrieved object from Redis.
        """
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
                # Convert Pydantic ValidationError to SchemaValidationError with index context
                raise SchemaValidationError(str(e), index=i) from e
            except Exception as e:
                # Capture other exceptions with context
                object_id = f"at index {i}"
                raise ValueError(
                    f"Error processing object {object_id}: {str(e)}"
                ) from e

        return prepared_objects

    def write(
        self,
        redis_client: Redis,
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
            redis_client (Redis): A Redis client used for writing data.
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
        redis_client: AsyncRedis,
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
            redis_client (AsyncRedis): An asynchronous Redis client used
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
        self, redis_client: Redis, keys: Iterable[str], batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve objects from Redis by keys.

        Args:
            redis_client (Redis): Synchronous Redis client.
            keys (Iterable[str]): Keys to retrieve from Redis.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            List[Dict[str, Any]]: List of objects pulled from redis.
        """
        results: List = []

        if not isinstance(keys, Iterable):  # type: ignore
            raise TypeError("Keys must be an iterable of strings")

        if len(keys) == 0:  # type: ignore
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
        redis_client: AsyncRedis,
        keys: Iterable[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve objects from Redis by keys.

        Args:
            redis_client (AsyncRedis): Asynchronous Redis client.
            keys (Iterable[str]): Keys to retrieve from Redis.
            batch_size (Optional[int], optional): Number of objects to write
                in a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            Dict[str, Any]: Dictionary with keys and their corresponding
                objects.
        """
        results: List = []

        if not isinstance(keys, Iterable):  # type: ignore
            raise TypeError("Keys must be an iterable of strings")

        if len(keys) == 0:  # type: ignore
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
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """Synchronously set a hash value in Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        client.hset(name=key, mapping=obj)  # type: ignore

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """Asynchronously set a hash value in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        await client.hset(name=key, mapping=obj)  # type: ignore

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """Synchronously retrieve a hash value from Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        return client.hgetall(key)

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """Asynchronously retrieve a hash value from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        return await client.hgetall(key)


class JsonStorage(BaseStorage):
    """
    Internal subclass of BaseStorage for the Redis JSON data type.

    Implements json-specific logic for validation and read/write operations
    (both sync and async) in Redis.
    """

    type: IndexType = IndexType.JSON
    """JSON data type for the index"""

    @staticmethod
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """Synchronously set a JSON obj in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        client.json().set(key, "$", obj)

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """Asynchronously set a JSON obj in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        await client.json().set(key, "$", obj)

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """Synchronously retrieve a JSON obj from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the JSON obj.

        Returns:
            Dict[str, Any]: The retrieved JSON obj from Redis.
        """
        return client.json().get(key)

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """Asynchronously retrieve a JSON obj from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the JSON obj.

        Returns:
            Dict[str, Any]: The retrieved JSON obj from Redis.
        """
        return await client.json().get(key)

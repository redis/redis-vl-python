import asyncio
import uuid

from typing import Callable, List, Optional, Dict, Any, Iterable

from redisvl.utils.utils import convert_bytes
from redisvl.schema import SchemaModel

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.indexDefinition import IndexType


class BaseStorage:
    type: IndexType = None
    DEFAULT_BATCH_SIZE: int = 200
    DEFAULT_WRITE_CONCURRENCY: int = 20

    def __init__(self, prefix: str, key_separator: str):
        self._prefix = prefix
        self._key_separator = key_separator

    @staticmethod
    def _key(
        key_value: str,
        prefix: str = "",
        key_separator: str = ""
    ) -> str:
        """
        Create a redis key as a combination of an index key prefix (optional) and specified key value.
        The key value is typically a unique identifier, created at random, or derived from
        some specified metadata.

        Args:
            key_value (str): The specified unique identifier for a particular document
                             indexed in Redis.

        Returns:
            str: The full Redis key including key prefix and value as a string.
        """
        return f"{prefix}{key_separator}{key_value}"

    def _create_key(
        self, record: Dict[str, Any], key_field: Optional[str] = None
    ) -> str:
        """Construct the Redis top level key.

        Args:
            record (Dict[str, Any]): A dictionary containing the record to be indexed.
            key_field (Optional[str], optional): A field within the record
                to use in the Redis key.

        Returns:
            str: The key to be used for a given record in Redis.

        Raises:
            ValueError: If the key field is not found in the record.
        """
        if key_field is None:
            key_value = uuid.uuid4().hex
        else:
            try:
                key_value = record[key_field]  # type: ignore
            except KeyError:
                raise ValueError(f"Key field {key_field} not found in record {record}")

        return self._key(key_value, prefix=self._prefix, sep=self._key_separator)

    def _preprocess(self, preprocess: Callable, obj: Any) -> Dict[str, Any]:
        # optionally preprocess object
        if preprocess:
            obj = preprocess(obj)
        return obj

    async def _apreprocess(self, preprocess: Callable, obj: Any) -> Dict[str, Any]:
        # optionally async preprocess object
        if preprocess:
            obj = await preprocess(obj)
        return obj

    def _validate(self, obj: dict):
        raise NotImplementedError

    def _set(client, key: str, obj: dict):
        """Storage type-specific set operator."""
        raise NotImplementedError

    async def _aset(client, key: str, obj: dict):
        """Storage type-specific async set operator."""
        raise NotImplementedError

    def _get(client, key: str):
        """Storage type-specific get operator."""
        raise NotImplementedError

    async def _aget(client, key: str):
        """Storage type-specific get operator."""
        raise NotImplementedError

    def write(
        self,
        redis_client: Redis,
        objects: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None
    ):
        """
        Write a batch of objects to Redis as hash entries.

        Args:
            redis_client (Redis): A Redis client used for writing data.
            objects (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): A function to preprocess objects before storage. Defaults to None.
            batch_size (Optional[int]): Number of objects to write in a single Redis pipeline execution. Defaults to class's default batch size.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.
        """
        if keys and len(keys) != len(objects):
            raise ValueError("Length of keys does not match the length of objects")

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE  # Use default or calculate based on the input data

        keys_iterator = iter(keys) if keys else None

        with redis_client.pipeline(transaction=False) as pipe:
            for i, obj in enumerate(objects, start=1):
                key = next(keys_iterator) if keys_iterator else self._create_key(obj, key_field)
                obj = self._preprocess(preprocess, obj)
                self._validate(obj)
                self._set(pipe, key, obj)
                if ttl:
                    pipe.expire(key, ttl)  # Set TTL if provided
                # execute mini batch
                if i % batch_size == 0:
                    pipe.execute()
            # clean up batches if needed
            if i % batch_size != 0:
                pipe.execute()

    async def awrite(
        self,
        redis_client: AsyncRedis,
        objects: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
    ):
        """
        Asynchronously write objects to Redis as hash entries with concurrency control.

        Args:
            redis_client (AsyncRedis): An asynchronous Redis client used for writing data.
            objects (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): An async function to preprocess objects before storage. Defaults to None.
            concurrency (Optional[int]): The maximum number of concurrent write operations. Defaults to class's default concurrency level.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.
        """
        if keys and len(keys) != len(objects):
            raise ValueError("Length of keys does not match the length of objects")

        if not concurrency:
            concurrency = self.DEFAULT_WRITE_CONCURRENCY

        semaphore = asyncio.Semaphore(concurrency)
        keys_iterator = iter(keys) if keys else None

        async def _load(obj: Dict[str, Any], key: str = None) -> None:
            async with semaphore:
                if key is None:
                    key = self._create_key(obj, key_field)
                obj = await self._apreprocess(preprocess, obj)
                self._validate(obj)
                await self._aset(redis_client, key, obj)
                if ttl:
                    await redis_client.expire(key)

        if keys_iterator:
            tasks = [asyncio.create_task(_load(obj, next(keys_iterator))) for obj in objects]
        else:
            tasks = [asyncio.create_task(_load(obj)) for obj in objects]

        await asyncio.gather(*tasks)

    def get(
        self,
        redis_client: Redis,
        keys: Iterable[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve objects from Redis by keys.

        Args:
            redis_client (Redis): Synchronous Redis client.
            keys (Iterable[str]): Keys to retrieve from Redis.

        Returns:
            List[Dict[str, Any]]: List of objects pulled from redis.
        """
        results: List = []

        if not isinstance(keys, Iterable):
            raise TypeError("Keys must be an iterable of strings")

        if len(keys) == 0:
            return []

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE  # Use default or calculate based on the input data

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
        concurrency: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve objects from Redis by keys, with concurrency control.

        Args:
            redis_client (AsyncRedis): Asynchronous Redis client.
            keys (Iterable[str]): Keys to retrieve from Redis.
            concurrency (int): The number of concurrent requests to make.

        Returns:
            Dict[str, Any]: Dictionary with keys and their corresponding objects.
        """
        if not isinstance(keys, Iterable):
            raise TypeError("Keys must be an iterable of strings")

        if len(keys) == 0:
            return []

        if not concurrency:
            concurrency = self.DEFAULT_WRITE_CONCURRENCY

        semaphore = asyncio.Semaphore(concurrency)

        async def _get(key: str) -> Dict[str, Any]:
            async with semaphore:
                result = await self._aget(redis_client, key)
                return result

        tasks = [asyncio.create_task(_get(key)) for key in keys]
        results = await asyncio.gather(*tasks)
        return convert_bytes(results)


class HashStorage(BaseStorage):
    type: IndexType = IndexType.HASH

    def __init__(self, schema: SchemaModel):
        """
        Initialize the HashStorage with a schema model.

        Args:
            schema (SchemaModel): A schema model that defines the structure and rules for stored data.
        """

        super().__init__(schema)

    def _validate(self, obj: dict):
        if not isinstance(obj, dict):
            raise TypeError("Object must be a dictionary.")


    def _set(client, key: str, obj: dict):
        client.hset(key, mapping=obj)

    async def _aset(client, key: str, obj: dict):
        await client.hset(key, mapping=obj)

    def _get(client, key: str):
        return client.hgetall(key)

    async def _aget(client, key: str):
        return await client.hgetall(key)


class JsonStorage(BaseStorage):
    type: IndexType = IndexType.JSON

    def __init__(self, schema: SchemaModel):
        """
        Initialize the JsonStorage with a schema model.

        Args:
            schema (SchemaModel): A schema model that defines the structure and rules for stored data.
        """
        super().__init__(schema)

    def _validate(self, obj: dict):
        if not isinstance(obj, dict):
            raise TypeError("Object must be a dictionary.")

    def _set(client, key: str, obj: dict):
        client.json().set(key, "$", obj)

    async def _aset(client, key: str, obj: dict):
        await client.json().set(key, "$", obj)

    def _get(client, key: str):
        return client.json().get(key)

    async def _aget(client, key: str):
       return await client.json().get(key)
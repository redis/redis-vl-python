import asyncio
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.indexDefinition import IndexType

from redisvl.utils.utils import convert_bytes


class BaseStorage:
    type: IndexType = None
    DEFAULT_BATCH_SIZE: int = 200
    DEFAULT_WRITE_CONCURRENCY: int = 20

    def __init__(self, prefix: str, key_separator: str):
        """
        Initialize the BaseStorage with a specific prefix and key separator for Redis keys.

        Args:
            prefix (str): The prefix to prepend to each Redis key.
            key_separator (str): The separator to use between the prefix and the key value.
        """
        self._prefix = prefix
        self._key_separator = key_separator

    @staticmethod
    def _key(key_value: str, prefix: str = "", key_separator: str = "") -> str:
        """
        Create a Redis key using a combination of a prefix, separator, and the key value.

        Args:
            key_value (str): The unique identifier for the Redis entry.
            prefix (str): An optional prefix to append before the key value.
            key_separator (str): An optional separator to insert between prefix and key value.

        Returns:
            str: The fully formed Redis key.
        """
        return f"{prefix}{key_separator}{key_value}"

    def _create_key(self, obj: Dict[str, Any], key_field: Optional[str] = None) -> str:
        """
        Construct a Redis key for a given object, optionally using a specified field from the object as the key.

        Args:
            obj (Dict[str, Any]): The object from which to construct the key.
            key_field (Optional[str]): The field to use as the key, if provided.

        Returns:
            str: The constructed Redis key for the object.

        Raises:
            ValueError: If the key_field is not found in the object.
        """
        if key_field is None:
            key_value = uuid.uuid4().hex
        else:
            try:
                key_value = obj[key_field]  # type: ignore
            except KeyError:
                raise ValueError(f"Key field {key_field} not found in record {obj}")

        return self._key(key_value, prefix=self._prefix, sep=self._key_separator)

    @staticmethod
    def _preprocess(preprocess: Callable, obj: Any) -> Dict[str, Any]:
        """
        Apply a preprocessing function to the object if provided.

        Args:
            preprocess (Callable): Function to process the object.
            obj (Any): Object to preprocess.

        Returns:
            Dict[str, Any]: Processed object as a dictionary.
        """
        # optionally preprocess object
        if preprocess:
            obj = preprocess(obj)
        return obj

    @staticmethod
    async def _apreprocess(preprocess: Callable, obj: Any) -> Dict[str, Any]:
        """
        Asynchronously apply a preprocessing function to the object if provided.

        Args:
            preprocess (Callable): Async function to process the object.
            obj (Any): Object to preprocess.

        Returns:
            Dict[str, Any]: Processed object as a dictionary.
        """
        # optionally async preprocess object
        if preprocess:
            obj = await preprocess(obj)
        return obj

    def _validate(self, obj: Dict[str, Any]):
        """
        Validate the object before writing to Redis. This method should be implemented by subclasses.

        Args:
            obj (Dict[str, Any]): The object to validate.
        """
        raise NotImplementedError

    @staticmethod
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """
        Synchronously set the value in Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key under which to store the object.
            obj (Dict[str, Any]): The object to store in Redis.
        """
        raise NotImplementedError

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """
        Asynchronously set the value in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the object.
            obj (Dict[str, Any]): The object to store in Redis.
        """
        raise NotImplementedError

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """
        Synchronously get the value from Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key for which to retrieve the object.

        Returns:
            Dict[str, Any]: The retrieved object from Redis.
        """
        raise NotImplementedError

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """
        Asynchronously get the value from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the object.

        Returns:
            Dict[str, Any]: The retrieved object from Redis.
        """
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

    def __init__(self, prefix: str, key_separator: str):
        """ Initialize the HashStorage.

        Args:
            prefix (str): The key prefix appended before the separator and value.
            separator (str): The key separator used to join the prefix and value.
        """
        super().__init__(prefix, key_separator)

    def _validate(self, obj: Dict[str, Any]):
        """
        Validate that the given object is a dictionary, suitable for storage as a Redis hash.

        Args:
            obj (Dict[str, Any]): The object to validate.

        Raises:
            TypeError: If the object is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise TypeError("Object must be a dictionary.")

    @staticmethod
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """
        Synchronously set a hash value in Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        client.hset(name=key, mapping=obj)

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """
        Asynchronously set a hash value in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the hash.
            obj (Dict[str, Any]): The hash to store in Redis.
        """
        await client.hset(name=key, mapping=obj)

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """
        Synchronously retrieve a hash value from Redis for the given key.

        Args:
            client (Redis): The Redis client instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        return client.hgetall(key)

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """
        Asynchronously retrieve a hash value from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the hash.

        Returns:
            Dict[str, Any]: The retrieved hash from Redis.
        """
        return await client.hgetall(key)


class JsonStorage(BaseStorage):
    type: IndexType = IndexType.JSON

    def __init__(self, prefix: str, key_separator: str):
        """ Initialize the JsonStorage.

        Args:
            prefix (str): The key prefix appended before the separator and value.
            separator (str): The key separator used to join the prefix and value.
        """
        super().__init__(prefix, key_separator)

    def _validate(self, obj: Dict[str, Any]):
        """
        Validate that the given object is a dictionary, suitable for JSON serialization.

        Args:
            obj (Dict[str, Any]): The object to validate.

        Raises:
            TypeError: If the object is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise TypeError("Object must be a dictionary.")

    @staticmethod
    def _set(client: Redis, key: str, obj: Dict[str, Any]):
        """
        Synchronously set a JSON obj in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        client.json().set(key, "$", obj)

    @staticmethod
    async def _aset(client: AsyncRedis, key: str, obj: Dict[str, Any]):
        """
        Asynchronously set a JSON obj in Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key under which to store the JSON obj.
            obj (Dict[str, Any]): The JSON obj to store in Redis.
        """
        await client.json().set(key, "$", obj)

    @staticmethod
    def _get(client: Redis, key: str) -> Dict[str, Any]:
        """
        Synchronously retrieve a JSON obj from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the JSON obj.

        Returns:
            Dict[str, Any]: The retrieved JSON obj from Redis.
        """
        return client.json().get(key)

    @staticmethod
    async def _aget(client: AsyncRedis, key: str) -> Dict[str, Any]:
        """
        Asynchronously retrieve a JSON obj from Redis for the given key.

        Args:
            client (AsyncRedis): The Redis client instance.
            key (str): The key for which to retrieve the JSON obj.

        Returns:
            Dict[str, Any]: The retrieved JSON obj from Redis.
        """
        return await client.json().get(key)
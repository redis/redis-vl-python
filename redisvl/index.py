import json
import os
import yaml

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union
from functools import wraps

if TYPE_CHECKING:
    from redis.commands.search.field import Field
    from redis.commands.search.document import Document
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
from redis.commands.search.indexDefinition import IndexDefinition

from redisvl.query.query import BaseQuery, CountQuery, FilterQuery
from redisvl.schema import (
    SchemaModel,
    StorageType,
    read_schema,
    FieldsModel,
    MetadataSchemaGenerator,
)
from redisvl.storage import (
    BaseStorage,
    HashStorage,
    JsonStorage
)
from redisvl.utils import (
    get_async_redis_connection,
    get_redis_connection,
    check_redis_modules_exist,
    convert_bytes,
    make_dict,
)


def process_results(
    results: "Result", query: BaseQuery, storage_type: StorageType
) -> List[Dict[str, Any]]:
    """
    Convert a list of search Result objects into a list of document dictionaries.

    This function processes results from Redis, handling different storage types
    and query types. For JSON storage with empty return fields, it unpacks the JSON object
    while retaining the document ID. The 'payload' field is also removed from all
    documents for consistency.

    Args:
        results (Result): The search results from Redis.
        query (BaseQuery): The query object used for the search.
        storage_type (StorageType): The storage type of the search index (e.g., json or hash).

    Returns:
        List[Dict[str, Any]]: A list of processed document dictionaries.
    """
    # Handle count queries
    if isinstance(query, CountQuery):
        return results.total

    # Determine if unpacking JSON is needed
    unpack_json = (
        (storage_type == StorageType.JSON)
        and isinstance(query, FilterQuery)
        and not query._return_fields
    )

    # Process records
    def _process(doc: "Document") -> Dict[str, Any]:
        doc_dict = doc.__dict__

        # Unpack and Project JSON fields properly
        if unpack_json and "json" in doc_dict:
            json_data = doc_dict.get("json", {})
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            if isinstance(json_data, dict):
                return {"id": doc_dict.get("id"), **json_data}
            raise ValueError(f"Unable to parse json data from Redis {json_data}")

        # Remove 'payload' if present
        doc_dict.pop("payload", None)

        return doc_dict

    return [_process(doc) for doc in results.docs]

def check_index_exists():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.exists():
                raise ValueError(
                    f"Index has not been created. Must be created before calling {func.__name__}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


# TODO -- need a way to handle async version of this
def check_modules_present(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            client = getattr(self, client_variable_name)
            check_redis_modules_exist(client)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def check_connected(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, client_variable_name) is None:
                raise ValueError(
                    f"SearchIndex.connect() must be called before calling {func.__name__}"
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class SearchIndexBase:

    STORAGE_MAP = {
        StorageType.HASH.value: HashStorage,
        StorageType.JSON.value: JsonStorage,
    }

    def __init__(
        self,
        schema: Union[SchemaModel, Dict[str, str], str, os.PathLike],
        **kwargs,
    ):
        # read / load index schema
        self._schema = read_schema(schema)
        # configure index and storage specs
        self._storage = self.STORAGE_MAP[self._schema.index.storage_type](
            self._schema.index.prefix, self._schema.index.key_separator
        )
        # init empty redis conn
        self._redis_conn: Optional[redis.Redis] = None
        if "redis_url" in kwargs:
            redis_url = kwargs.pop("redis_url")
            self.connect(redis_url, **kwargs)

    @property
    def schema(self) -> Dict[str, List[Any]]:
        """Return the schema of the index."""
        # TODO -- might want to return SchemaModel
        # Otherwise there's no easy way to get to prefix, name, etc
        return self._schema.dict()

    def write_schema(self, path: Union[str, os.PathLike]) -> None:
        """Write the schema to a yaml file."""
        self.schema.dump(path)

    # TODO -- i moved these back
    # My thought is users will expect index.name if they are
    # building an index with a particular name
    # flushes all of these to the highest level of abstraction
    # index._schema.name is less ideal for an end user
    @property
    def name(self) -> str:
        """The name of the Redis search index."""
        return self._schema.index.name

    @property
    def prefix(self) -> str:
        """The optional key prefix that comes before a unique key value in forming a Redis key."""
        return self._schema.index.prefix

    @property
    def key_separator(self) -> str:
        """The optional separator between a defined prefix and key value in forming a Redis key."""
        return self._schema.index.key_separator

    @property
    def storage(self) -> BaseStorage:
        """The Storage class that handles all upserts and reads to/from the Redis instances."""
        return self._storage

    @property
    def storage_type(self) -> str:
        """The underlying storage type for the search index: hash or json."""
        return self._schema.index.storage_type

    @classmethod
    def from_yaml(cls, schema_path: str, **kwargs):
        """Create a SearchIndex from a YAML schema file.
        Args:
            schema_path (str): Path to the YAML schema file.

        Returns:
            SearchIndex: A SearchIndex object.
        """
        schema = read_schema(schema_path)
        return cls(schema, **kwargs)

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any], **kwargs):
        """Create a SearchIndex from a dictionary.
        Args:
            schema_dict (Dict[str, Any]): A dictionary containing the schema.

        Returns:
            SearchIndex: A SearchIndex object.
        """
        schema = SchemaModel(**schema_dict)
        return cls(schema, **kwargs)

    def set_client(self, client: redis.Redis):
        self._redis_conn = client

    @property
    @check_connected("_redis_conn")
    def client(self) -> redis.Redis:
        """The underlying redis-py client object."""
        return self._redis_conn  # type: ignore

    @classmethod
    def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        key_separator: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        raise NotImplementedError

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        raise NotImplementedError

    def connect(self, url: str, **kwargs):
        """Connect to a Redis instance."""
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis instance."""
        self._redis_conn = None
        return self

    def key(self, key_value: str) -> str:
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
        return self._storage._key(
            key_value, self._schema.index.prefix, self._schema.index.key_separator
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    def list_all(self) -> List[str]:
        raise NotImplementedError

    def create(self, overwrite: Optional[bool] = False):
        raise NotImplementedError

    def delete(self, drop: bool = True):
        raise NotImplementedError

    def load(
        self,
        data: Iterable[Dict[str, Any]],
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        **kwargs,
    ):
        raise NotImplementedError


class SearchIndex(SearchIndexBase):
    """A class for interacting with Redis as a vector database.

    This class is a wrapper around the redis-py client that provides
    purpose-built methods for interacting with Redis as a vector database.

    Example:
        >>> from redisvl.index import SearchIndex
        >>> index = SearchIndex.from_yaml("schema.yaml")
        >>> index.create(overwrite=True)
        >>> index.load(data) # data is an iterable of dictionaries
    """

    def __init__(self, schema: SchemaModel):
        super().__init__(schema)

    @classmethod
    def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        key_separator: Optional[str] = None,
        **kwargs,
    ):
        """Create a SearchIndex from an existing index in Redis.

        Args:
            name (str): Index name.
            url (Optional[str], optional): Redis URL. REDIS_URL env var
                is used if not provided. Defaults to None.
            fields (Optional[List[Field]], optional): List of Redis search
                fields to include in the schema. Defaults to None.
            key_separator (Optional[str], optional): The character to use as a
                separator between prefix and key value in Redis. Defaults to None.

        Returns:
            SearchIndex: A SearchIndex object.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
            ValueError: If the REDIS_URL env var is not set and url is not provided.

        """
        client = get_redis_connection(url, **kwargs)
        info = convert_bytes(client.ft(name).info())
        index_definition = make_dict(info["index_definition"])
        schema = SchemaModel.from_args(
            name=name,
            storage_type=index_definition["key_type"].lower(),
            prefix=index_definition["prefixes"][0],
            key_separator=key_separator if key_separator is not None else ":",
            fields=fields
        )
        instance = cls(schema)
        instance.set_client(client)
        return instance

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance.

        Args:
            url (str): Redis URL. REDIS_URL env var is used if not provided.

        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails.
            ValueError: If the REDIS_URL env var is not set and url is not provided.
        """
        self._redis_conn = get_redis_connection(url, **kwargs)
        return self

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    def create(self, overwrite: Optional[bool] = False) -> None:
        """
        Create an index in Redis from this SearchIndex object.

        Args:
            overwrite (Optional[bool], optional): Whether to overwrite the index if it
                already exists. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
            ValueError: If no fields are defined for the index.
        """
        # Check that fields are defined.
        if self._schema.fields.is_empty:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if self.exists():
            if not overwrite:
                print("Index already exists, not overwriting.")
                return None
            print("Index already exists, overwriting.")
            self.delete()

        # Create the index with the specified fields and settings.
        self._redis_conn.ft(self._schema.index.name).create_index(  # type: ignore
            fields=self._schema.index_fields,
            definition=IndexDefinition(
                prefix=[self._schema.index.prefix], index_type=self._storage.type
            ),
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        self._redis_conn.ft(
            self._schema.index.name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    def load(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Load a batch of objects to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): A function to preprocess objects before storage. Defaults to None.
            batch_size (Optional[int]): Number of objects to write in a single Redis pipeline execution. Defaults to class's default batch size.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> index.load(data, preprocess=func)
        """
        self._storage.write(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            batch_size=batch_size,
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    def upsert(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        generate_schema: bool = True,
        **kwargs,
    ):
        """
        Load a batch of objects to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): A function to preprocess objects before storage. Defaults to None.
            batch_size (Optional[int]): Number of objects to write in a single Redis pipeline execution. Defaults to class's default batch size.
            generate_schema (bool): Whether or not to automatically attempt to generate schema for the metadata. Defaults to True.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> index.upsert(data, preprocess=func)
        """
        # TODO -- do we want to do this?
        # Let's think more on it
        if self._schema.fields.is_empty and generate_schema:
            field_map = MetadataSchemaGenerator(data).generate()
            self._schema.fields = FieldsModel(field_map)

        # TODO -- not sure this would work
        # if it doesn't exist, then no need to not overwrite?
        # why do we want to do this here?
        # I imagine users and higher level abstractions, like LangChain, will want to control this
        if not self.exists():
            self.create(overwrite=False)

        self._storage.write(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            batch_size=batch_size,
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = self._redis_conn.ft(self._schema.index.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        """Run a query on this index.

        This is similar to the search method, but takes a BaseQuery
        object directly (does not allow for the usage of a raw
        redis query string) and post-processes results of the search.

        Args:
            query (BaseQuery): The query to run.

        Returns:
            List[Result]: A list of search results.
        """
        results = self.search(query.query, query_params=query.params)
        # post process the results
        return process_results(
            results, query=query, storage_type=self._schema.index.storage_type
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = self.list_all()
        return self._schema.index.name in indices

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    @check_index_exists()
    def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            self._redis_conn.ft(self._schema.index.name).info()  # type: ignore
        )

    @check_connected("_redis_conn")
    @check_modules_present("_redis_conn")
    def list_all(self) -> List[str]:
        """List all search indices in Redis.

        Returns:
            List[str]: List of search index names stored in Redis.
        """
        return convert_bytes(
            self._redis_conn.execute_command("FT._LIST")
        )


class AsyncSearchIndex(SearchIndexBase):
    """A class for interacting with Redis as a vector database asynchronously.

    This class is a wrapper around the redis-py client that provides
    purpose-built methods for interacting with Redis as a vector database.

    Example:
        >>> from redisvl.index import AsyncSearchIndex
        >>> index = AsyncSearchIndex.from_yaml("schema.yaml")
        >>> await index.create(overwrite=True)
        >>> await index.load(data) # data is an iterable of dictionaries
    """

    def __init__(self, schema: SchemaModel):
        super().__init__(schema)

    @classmethod
    async def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        key_separator: Optional[str] = None,
        **kwargs,
    ):
        """Create a SearchIndex from an existing index in Redis.

        Args:
            name (str): Index name.
            url (Optional[str], optional): Redis URL. REDIS_URL env var
                is used if not provided. Defaults to None.
            fields (Optional[List[Field]], optional): List of Redis search
                fields to include in the schema. Defaults to None.
            key_separator (Optional[str], optional): The character to use as a
                separator between prefix and key value in Redis. Defaults to None.

        Returns:
            AsyncSearchIndex: An AsyncSearchIndex object.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
            ValueError: If the REDIS_URL env var is not set and url is not provided.

        """
        client = get_async_redis_connection(url, **kwargs)
        info = convert_bytes(await client.ft(name).info())
        index_definition = make_dict(info["index_definition"])
        schema = SchemaModel.from_args(
            name=name,
            storage_type=index_definition["key_type"].lower(),
            prefix=index_definition["prefixes"][0],
            key_separator=key_separator if key_separator is not None else ":",
            fields=fields
        )
        instance = cls(schema)
        instance.set_client(client)
        return instance

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance.

        Args:
            url (str): Redis URL. REDIS_URL env var is used if not provided.

        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails.
            ValueError: If no Redis URL is provided and REDIS_URL env var is not set.
        """
        self._redis_conn = get_async_redis_connection(url, **kwargs)
        return self

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def create(self, overwrite: Optional[bool] = False) -> None:
        """
        Asynchronously create an index in Redis from this SearchIndex object.

        Args:
            overwrite (Optional[bool], optional): Whether to overwrite the index if it
                already exists. Defaults to False.
        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
        """
        if self._schema.fields.is_empty:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if await self.exists():
            if not overwrite:
                print("Index already exists, not overwriting.")
                return None
            print("Index already exists, overwriting.")
            await self.delete()

        # Create Index with proper IndexType
        await self._redis_conn.ft(self._schema.index.name).create_index(  # type: ignore
            fields=self._schema.index_fields,
            definition=IndexDefinition(
                prefix=[self._schema.index.prefix], index_type=self._storage.type
            ),
        )

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        await self._redis_conn.ft(
            self._schema.index.name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def load(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
        **kwargs,
    ):
        """
        Asynchronously load objects to Redis with concurrency control.

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

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> await index.load(data, preprocess=func)
        """
        await self._storage.awrite(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            concurrency=concurrency,
        )

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def upsert(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        generate: bool = True,
        **kwargs,
    ):
        """
        Load a batch of objects to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): A function to preprocess objects before storage. Defaults to None.
            batch_size (Optional[int]): Number of objects to write in a single Redis pipeline execution. Defaults to class's default batch size.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> index.load(data, preprocess=func)
        """

        if not self.exists():
            await self.create(overwrite=False)

        await self._storage.awrite(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            batch_size=batch_size,
        )


    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = await self._redis_conn.ft(self._schema.index.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        """Run a query on this index.

        This is similar to the search method, but takes a BaseQuery
        object directly (does not allow for the usage of a raw
        redis query string) and post-processes results of the search.

        Args:
            query (BaseQuery): The query to run.

        Returns:
            List[Result]: A list of search results.
        """
        results = await self.search(query.query, query_params=query.params)
        # post process the results
        return process_results(
            results, query=query, storage_type=self._schema.index.storage_type
        )

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = await self.list_all()
        return self._schema.index.name in convert_bytes(indices)

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            await self._redis_conn.ft(self._schema.index.name).info()  # type: ignore
        )

    @check_connected("_redis_conn")
    # @check_modules_present("_redis_conn")
    async def list_all(self) -> List[str]:
        """List all search indices in Redis.

        Returns:
            List[str]: List of search index names stored in Redis.
        """
        return convert_bytes(
            await self._redis_conn.execute_command("FT._LIST")
        )

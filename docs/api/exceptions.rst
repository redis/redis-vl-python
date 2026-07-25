***********
Exceptions
***********

RedisVL defines its custom exceptions in ``redisvl.exceptions``. Every one of them
inherits from :class:`RedisVLError`, so catching that single base class is enough to
handle any error the library raises on its own behalf. Catch the more specific
subclasses when you want to react differently to, for example, a schema validation
failure than to a Redis connection problem.

.. code-block:: text

    Exception
    └── RedisVLError
        ├── RedisSearchError
        ├── SchemaValidationError
        ├── QueryValidationError
        └── RedisModuleVersionError

.. note::

   Exceptions raised by the underlying ``redis-py`` client, such as
   ``redis.exceptions.ConnectionError``, are not part of this hierarchy. Where
   RedisVL performs an index or search operation on your behalf it wraps those
   errors in a :class:`RedisSearchError` and chains the original exception, so the
   underlying cause is still available on ``__cause__``.


When each error is raised
=========================

.. list-table::
   :widths: 28 42 30
   :header-rows: 1

   * - Exception
     - Raised when
     - Typical entry points
   * - :class:`SchemaValidationError`
     - An object does not match the index schema. Only raised when the index was
       created with ``validate_on_load=True``.
     - ``load()``
   * - :class:`QueryValidationError`
     - A query is not valid for the index it targets, for example setting
       ``ef_runtime`` on a vector field that uses the ``flat`` algorithm.
     - ``query()``
   * - :class:`RedisSearchError`
     - An index or search operation fails, including errors returned by Redis
       itself.
     - ``create()``, ``delete()``, ``search()``, ``aggregate()``
   * - :class:`RedisModuleVersionError`
     - The connected Redis or Redis Search version does not support a requested
       feature, such as an ``svs-vamana`` vector field.
     - ``create()``
   * - :class:`RedisVLError`
     - A load operation fails for a reason not covered by a more specific error.
       Also the base class for everything above.
     - ``load()``

All of the above apply equally to :class:`~redisvl.index.SearchIndex` and
:class:`~redisvl.index.AsyncSearchIndex`.


Handling errors
===============

Validating data on load
-----------------------

Schema validation is off by default. Pass ``validate_on_load=True`` to have RedisVL
check each object against the index schema before writing it, and raise
:class:`SchemaValidationError` on the first object that does not match.

.. code-block:: python

    from redisvl.index import SearchIndex
    from redisvl.exceptions import SchemaValidationError

    index = SearchIndex.from_yaml(
        "schema.yaml",
        redis_url="redis://localhost:6379",
        validate_on_load=True,
    )

    try:
        index.load(data)
    except SchemaValidationError as e:
        # The message identifies the offending object by its position in the
        # input and describes which field failed and why.
        print(f"Invalid record: {e}")

The error message reports the index of the object within the batch you passed, so a
failure part way through a large load still points at a specific record.

Handling query failures
-----------------------

:class:`QueryValidationError` signals a query that cannot run against this index. It
is a programming error rather than a transient one, so it is usually worth failing
loudly instead of retrying.

.. code-block:: python

    from redisvl.query import VectorQuery
    from redisvl.exceptions import QueryValidationError

    query = VectorQuery(
        vector=[0.1, 0.2, 0.3],
        vector_field_name="embedding",
        return_fields=["title"],
        ef_runtime=50,  # only supported by the 'hnsw' algorithm
    )

    try:
        results = index.query(query)
    except QueryValidationError as e:
        print(f"Query rejected: {e}")

Separating configuration problems from Redis problems
-----------------------------------------------------

:class:`RedisModuleVersionError` is a subclass of :class:`RedisVLError`, not of
:class:`RedisSearchError`, so ordering the ``except`` clauses lets you distinguish an
unsupported feature from a genuine Redis failure.

.. code-block:: python

    from redisvl.exceptions import RedisModuleVersionError, RedisSearchError

    try:
        index.create(overwrite=True)
    except RedisModuleVersionError as e:
        # The deployment does not support the requested feature, for example an
        # 'svs-vamana' field on a Redis version without a new enough Redis Search.
        print(f"Unsupported by this Redis deployment: {e}")
    except RedisSearchError as e:
        # Something went wrong talking to Redis, or the index definition was
        # rejected. The original redis-py exception is available as e.__cause__.
        print(f"Index creation failed: {e}")

Catching everything
-------------------

When the calling code only needs to know that the operation failed, catch the base
class.

.. code-block:: python

    from redisvl.exceptions import RedisVLError

    try:
        index.load(data)
        results = index.query(query)
    except RedisVLError as e:
        logger.error("RedisVL operation failed: %s", e)
        raise

Because RedisVL chains the underlying exception when it wraps one, ``e.__cause__``
still holds the original ``redis-py`` error where there was one.


Exception classes
=================

.. currentmodule:: redisvl.exceptions

RedisVLError
------------

.. autoclass:: RedisVLError
   :members:
   :show-inheritance:

RedisSearchError
----------------

.. autoclass:: RedisSearchError
   :members:
   :show-inheritance:

SchemaValidationError
---------------------

.. autoclass:: SchemaValidationError
   :members:
   :show-inheritance:

QueryValidationError
--------------------

.. autoclass:: QueryValidationError
   :members:
   :show-inheritance:

RedisModuleVersionError
-----------------------

.. autoclass:: RedisModuleVersionError
   :members:
   :show-inheritance:

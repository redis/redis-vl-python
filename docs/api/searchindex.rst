********************
Search Index Classes
********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :ref:`searchindex_api`
     - Primary class to write, read, and search across data structures in Redis.
   * - :ref:`asyncsearchindex_api`
     - Async version of the SearchIndex to write, read, and search across data structures in Redis.
   * - :ref:`searchresults_api`
     - List of result documents returned by a query, which also reports result completeness.
   * - :ref:`bulkresult_api`
     - Outcome of a bulk operation: how many documents matched, how many were processed, and whether the run completed.

Serializing index configurations
================================

Both index classes provide synchronous ``to_dict()`` and ``to_yaml()`` methods.
They export the schema by default, without opening a connection or reading
``REDIS_URL``. Existing schema-only ``from_dict()`` and ``from_yaml()`` calls
continue to work.

.. code-block:: python

    config = index.to_dict()
    restored = SearchIndex.from_dict(config, redis_url="redis://localhost:6379")
    index.to_yaml("index.yaml", overwrite=False)

Pass ``include_connection=True`` to include a ``connection`` mapping alongside
``index``, ``fields`` and ``version``:

.. code-block:: yaml

    connection:
      redis_url: redis://localhost:6379/0
      connection_kwargs:
        socket_timeout: 2.0
      requires_override: false

Connection exports retain the URL endpoint and path, including IPv6 brackets
and Sentinel host lists. They omit URL user information, **all** query
parameters and fragments. Only boolean ``ssl``, ``ssl_check_hostname``,
``decode_responses``, ``socket_keepalive``, ``retry_on_timeout`` and ``cluster``;
finite numeric ``db``, ``protocol``, ``socket_timeout``,
``socket_connect_timeout``, ``health_check_interval`` and ``max_connections``;
and ``ssl_cert_reqs`` values ``none``, ``optional`` or ``required`` are exported
from connection kwargs. Other options, such as passwords, certificate paths,
credential providers and custom connection pools, are omitted.

Any omission sets ``requires_override: true``. Configurations sourced from an
injected client or the environment also require an override: their connection
internals are not inspected. Restoring such a configuration without a nonempty
``redis_url`` or a ``redis_client`` raises ``ValueError``, even if ``REDIS_URL``
is set. This also applies to instances created by ``from_existing()``, even
with an explicit URL, because that factory supplies a client internally.
Supply the complete connection configuration at restore time:

.. code-block:: python

    import os

    config = index.to_dict(include_connection=True)
    restored = SearchIndex.from_dict(config, redis_url=os.environ["REDIS_URL"])

A replacement URL or client discards stored connection kwargs; pass any required
``connection_kwargs`` explicitly. Without a replacement endpoint, explicit
``connection_kwargs`` replace the stored mapping rather than merging with it.
Unredacted configurations restore their stored endpoint and allowed options.
YAML imports use the same rules.

This is configuration serialization, not a snapshot of live Redis data or
client state. Endpoints, paths and the schema may themselves be private: use the
default schema-only export when sharing artifacts outside your deployment.

.. _searchindex_api:

SearchIndex
===========

.. currentmodule:: redisvl.index

.. autoclass:: SearchIndex
   :inherited-members:
   :members:

.. _asyncsearchindex_api:

AsyncSearchIndex
================

.. currentmodule:: redisvl.index

.. autoclass:: AsyncSearchIndex
   :inherited-members:
   :members:

.. _searchresults_api:

SearchResults
=============

.. currentmodule:: redisvl.index

.. autoclass:: SearchResults
   :members:

.. _bulkresult_api:

BulkResult
==========

.. currentmodule:: redisvl.index

.. autoclass:: BulkResult
   :members:

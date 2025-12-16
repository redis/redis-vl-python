
*****
Query
*****

Query classes in RedisVL provide a structured way to define simple or complex
queries for different use cases. Each query class wraps the ``redis-py`` Query module
https://github.com/redis/redis-py/blob/master/redis/commands/search/query.py with extended functionality for ease-of-use.


VectorQuery
===========

.. currentmodule:: redisvl.query


.. autoclass:: VectorQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

.. note::
   **Runtime Parameters for Performance Tuning**

   VectorQuery supports runtime parameters for HNSW and SVS-VAMANA indexes that can be adjusted at query time without rebuilding the index:

   **HNSW Parameters:**

   - ``ef_runtime``: Controls search accuracy (higher = better recall, slower search)

   **SVS-VAMANA Parameters:**

   - ``search_window_size``: Size of search window for KNN searches
   - ``use_search_history``: Whether to use search buffer (OFF/ON/AUTO)
   - ``search_buffer_capacity``: Tuning parameter for 2-level compression

   Example with HNSW runtime parameters:

   .. code-block:: python

      from redisvl.query import VectorQuery

      query = VectorQuery(
          vector=[0.1, 0.2, 0.3],
          vector_field_name="embedding",
          num_results=10,
          ef_runtime=150  # Higher for better recall
      )

   Example with SVS-VAMANA runtime parameters:

   .. code-block:: python

      query = VectorQuery(
          vector=[0.1, 0.2, 0.3],
          vector_field_name="embedding",
          num_results=10,
          search_window_size=20,
          use_search_history='ON',
          search_buffer_capacity=30
      )


VectorRangeQuery
================


.. currentmodule:: redisvl.query


.. autoclass:: VectorRangeQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

.. note::
   **Runtime Parameters for Range Queries**

   VectorRangeQuery supports runtime parameters for controlling range search behavior:

   **HNSW & SVS-VAMANA Parameters:**

   - ``epsilon``: Range search approximation factor (default: 0.01)

   **SVS-VAMANA Parameters:**

   - ``search_window_size``: Size of search window
   - ``use_search_history``: Whether to use search buffer (OFF/ON/AUTO)
   - ``search_buffer_capacity``: Tuning parameter for 2-level compression

   Example:

   .. code-block:: python

      from redisvl.query import VectorRangeQuery

      query = VectorRangeQuery(
          vector=[0.1, 0.2, 0.3],
          vector_field_name="embedding",
          distance_threshold=0.3,
          epsilon=0.05,              # Approximation factor
          search_window_size=20,     # SVS-VAMANA only
          use_search_history='AUTO'  # SVS-VAMANA only
      )

AggregateHybridQuery
================


.. currentmodule:: redisvl.query


.. autoclass:: AggregateHybridQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

.. note::
   The ``stopwords`` parameter in :class:`AggregateHybridQuery` (and :class:`HybridQuery`) controls query-time stopword filtering (client-side).
   For index-level stopwords configuration (server-side), see :class:`redisvl.schema.IndexInfo.stopwords`.
   Using query-time stopwords with index-level ``STOPWORDS 0`` is counterproductive.

.. note::
   :class:`HybridQuery` and :class:`AggregateHybridQuery` apply linear combination inconsistently. :class:`HybridQuery` uses ``linear_alpha`` to weight the text score, while :class:`AggregateHybridQuery` uses ``alpha`` to weight the vector score. When switching between the two classes, take care to revise your ``alpha`` setting.

.. note::
   **Runtime Parameters for Hybrid Queries**

   **Important:** AggregateHybridQuery uses FT.AGGREGATE commands which do NOT support runtime parameters.
   Runtime parameters (``ef_runtime``, ``search_window_size``, ``use_search_history``, ``search_buffer_capacity``)
   are only supported with FT.SEARCH commands.

   For runtime parameter support, use :class:`HybridQuery`, :class:`VectorQuery`, or :class:`VectorRangeQuery` instead of AggregateHybridQuery.

   Example with HybridQuery (supports runtime parameters):

   .. code-block:: python

      from redisvl.query import HybridQuery

      query = HybridQuery(
          text="query string",
          text_field_name="description",
          vector=[0.1, 0.2, 0.3],
          vector_field_name="embedding",
          vector_search_method="KNN",
          knn_ef_runtime=150,  # Runtime parameters work with HybridQuery
          return_fields=["description"],
          num_results=10,
      )

HybridQuery
================


.. currentmodule:: redisvl.query.hybrid


.. autoclass:: HybridQuery
   :members:
   :inherited-members:
   :show-inheritance:

.. note::
   The ``stopwords`` parameter in :class:`HybridQuery` (and :class:`AggregateHybridQuery`) controls query-time stopword filtering (client-side).
   For index-level stopwords configuration (server-side), see :class:`redisvl.schema.IndexInfo.stopwords`.
   Using query-time stopwords with index-level ``STOPWORDS 0`` is counterproductive.

.. note::
   :class:`HybridQuery` and :class:`AggregateHybridQuery` apply linear combination inconsistently. :class:`HybridQuery` uses ``linear_alpha`` to weight the text score, while :class:`AggregateHybridQuery` uses ``alpha`` to weight the vector score. When switching between the two classes, take care to revise your ``alpha`` setting.

TextQuery
================


.. currentmodule:: redisvl.query


.. autoclass:: TextQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

.. note::
   The ``stopwords`` parameter in :class:`TextQuery` controls query-time stopword filtering (client-side).
   For index-level stopwords configuration (server-side), see :class:`redisvl.schema.IndexInfo.stopwords`.
   Using query-time stopwords with index-level ``STOPWORDS 0`` is counterproductive.


FilterQuery
===========


.. currentmodule:: redisvl.query


.. autoclass:: FilterQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize



CountQuery
==========

.. currentmodule:: redisvl.query


.. autoclass:: CountQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize



MultiVectorQuery
==========

.. currentmodule:: redisvl.query


.. autoclass:: MultiVectorQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

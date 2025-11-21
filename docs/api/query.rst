
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


VectorRangeQuery
================


.. currentmodule:: redisvl.query


.. autoclass:: VectorRangeQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize

HybridQuery
================


.. currentmodule:: redisvl.query


.. autoclass:: HybridQuery
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: add_filter,get_args,highlight,return_field,summarize


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
   See the `Stopwords Interaction Guide <../stopwords_interaction_guide.html>`_ for details.


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

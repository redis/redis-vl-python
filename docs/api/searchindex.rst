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

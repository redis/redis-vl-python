
***********
Schema
***********

Schema in RedisVL provides a structured format to define index settings and
field configurations using the following three components:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Component
     - Description
   * - `version`
     - The version of the schema spec. Current supported version is `0.1.0`.
   * - `index`
     - Index specific settings like name, key prefix, key separator, and storage type.
   * - `fields`
     - Subset of fields within your data to include in the index and any custom settings.

IndexSchema
===========

.. _indexschema_api:

.. currentmodule:: redisvl.schema

.. autoclass:: IndexSchema
   :members:
   :exclude-members: generate_fields,validate_and_create_fields

Supported Field Types
=====================

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Field Type
     - Description
   * - `vector`
     - Vector embeddings data typically generated from another AI/ML model to represent unstructured data.
   * - `text`
     - Full text data that enable full text search and filtering operations.
   * - `tag`
     - Label-like fields that are used for exact matches and filtering operations.
   * - `numeric`
     - Numeric fields used for range filters.
   * - `geo`
     - Geographic coordinates used for geo search.

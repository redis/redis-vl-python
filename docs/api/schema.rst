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
   :exclude-members: generate_fields,validate_and_create_fields,redis_fields


Defining Fields
===============

Fields in the schema can be defined in YAML format or as a Python dictionary, specifying a name, type, an optional path, and attributes for customization.

**YAML Example**:

.. code-block:: yaml

    - name: title
      type: text
      path: $.document.title
      attrs:
        weight: 1.0
        no_stem: false
        withsuffixtrie: true

**Python Dictionary Example**:

.. code-block:: python

    {
        "name": "location",
        "type": "geo",
        "attrs": {
            "sortable": true
        }
    }

Supported Field Types and Attributes
====================================

Each field type supports specific attributes that customize its behavior. Below are the field types and their available attributes:

**Text Field Attributes**:

- `weight`: Importance of the field in result calculation.
- `no_stem`: Disables stemming during indexing.
- `withsuffixtrie`: Optimizes queries by maintaining a suffix trie.
- `phonetic_matcher`: Enables phonetic matching.
- `sortable`: Allows sorting on this field.
- `no_index`: When True, field is not indexed but can be returned in results (requires `sortable=True`).
- `unf`: Un-normalized form. When True, preserves original case for sorting (requires `sortable=True`).

**Tag Field Attributes**:

- `separator`: Character for splitting text into individual tags.
- `case_sensitive`: Case sensitivity in tag matching.
- `withsuffixtrie`: Suffix trie optimization for queries.
- `sortable`: Enables sorting based on the tag field.
- `no_index`: When True, field is not indexed but can be returned in results (requires `sortable=True`).

**Numeric Field Attributes**:

- `sortable`: Enables sorting on the numeric field.
- `no_index`: When True, field is not indexed but can be returned in results (requires `sortable=True`).
- `unf`: Un-normalized form. When True, maintains original numeric representation for sorting (requires `sortable=True`).

**Geo Field Attributes**:

- `sortable`: Enables sorting based on the geo field.
- `no_index`: When True, field is not indexed but can be returned in results (requires `sortable=True`).

**Common Vector Field Attributes**:

- `dims`: Dimensionality of the vector.
- `algorithm`: Indexing algorithm (`flat` or `hnsw`).
- `datatype`: Float datatype of the vector (`bfloat16`, `float16`, `float32`, `float64`).
- `distance_metric`: Metric for measuring query relevance (`COSINE`, `L2`, `IP`).

**HNSW Vector Field Specific Attributes**:

- `m`: Max outgoing edges per node in each layer.
- `ef_construction`: Max edge candidates during build time.
- `ef_runtime`: Max top candidates during search.
- `epsilon`: Range search boundary factor.

Note:
    See fully documented Redis-supported fields and options here: https://redis.io/commands/ft.create/
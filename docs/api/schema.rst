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
- `algorithm`: Indexing algorithm (`flat`, `hnsw`, or `svs-vamana`).
- `datatype`: Float datatype of the vector (`bfloat16`, `float16`, `float32`, `float64`). Note: SVS-VAMANA only supports `float16` and `float32`.
- `distance_metric`: Metric for measuring query relevance (`COSINE`, `L2`, `IP`).
- `initial_cap`: Initial capacity for the index (optional).
- `index_missing`: When True, allows searching for documents missing this field (optional).

**FLAT Vector Field Specific Attributes**:

- `block_size`: Block size for the FLAT index (optional).

**HNSW Vector Field Specific Attributes**:

- `m`: Max outgoing edges per node in each layer (default: 16).
- `ef_construction`: Max edge candidates during build time (default: 200).
- `ef_runtime`: Max top candidates during search (default: 10).
- `epsilon`: Range search boundary factor (default: 0.01).

**SVS-VAMANA Vector Field Specific Attributes**:

SVS-VAMANA (Scalable Vector Search with VAMANA graph algorithm) provides fast approximate nearest neighbor search with optional compression support. This algorithm is optimized for Intel hardware and offers reduced memory usage through vector compression.

- `graph_max_degree`: Maximum degree of the Vamana graph, i.e., the number of edges per node (default: 40).
- `construction_window_size`: Size of the candidate list during graph construction. Higher values yield better quality graphs but increase construction time (default: 250).
- `search_window_size`: Size of the candidate list during search. Higher values increase accuracy but also increase search latency (default: 20).
- `epsilon`: Relative factor for range query boundaries (default: 0.01).
- `compression`: Optional vector compression algorithm. Supported types:

  - `LVQ4`: 4-bit Learned Vector Quantization
  - `LVQ4x4`: 4-bit LVQ with 4x compression
  - `LVQ4x8`: 4-bit LVQ with 8x compression
  - `LVQ8`: 8-bit Learned Vector Quantization
  - `LeanVec4x8`: 4-bit LeanVec with 8x compression and dimensionality reduction
  - `LeanVec8x8`: 8-bit LeanVec with 8x compression and dimensionality reduction

- `reduce`: Reduced dimensionality for LeanVec compression. Must be less than `dims`. Only valid with `LeanVec4x8` or `LeanVec8x8` compression types. Lowering this value can speed up search and reduce memory usage (optional).
- `training_threshold`: Minimum number of vectors required before compression training begins. Must be less than 100 * 1024 (default: 10 * 1024).

**SVS-VAMANA Example**:

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: svs-vamana
        dims: 768
        distance_metric: cosine
        datatype: float32
        graph_max_degree: 64
        construction_window_size: 500
        search_window_size: 40
        compression: LeanVec4x8
        reduce: 384
        training_threshold: 1000

Note:
    - SVS-VAMANA requires Redis >= 8.2 with RediSearch >= 2.8.10.
    - SVS-VAMANA only supports `float16` and `float32` datatypes.
    - The `reduce` parameter is only valid with LeanVec compression types (`LeanVec4x8` or `LeanVec8x8`).
    - Intel's proprietary LVQ and LeanVec optimizations are not available in Redis Open Source. On non-Intel platforms and Redis Open Source, SVS-VAMANA with compression falls back to basic 8-bit scalar quantization.
    - See fully documented Redis-supported fields and options here: https://redis.io/commands/ft.create/
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

Basic Field Types
=================

RedisVL supports several basic field types for indexing different kinds of data. Each field type has specific attributes that customize its indexing and search behavior.

Text Fields
-----------

Text fields support full-text search with stemming, phonetic matching, and other text analysis features.

.. currentmodule:: redisvl.schema.fields

.. autoclass:: TextField
   :members:
   :show-inheritance:

.. autoclass:: TextFieldAttributes
   :members:
   :undoc-members:

Tag Fields
----------

Tag fields are optimized for exact-match filtering and faceted search on categorical data.

.. autoclass:: TagField
   :members:
   :show-inheritance:

.. autoclass:: TagFieldAttributes
   :members:
   :undoc-members:

Numeric Fields
--------------

Numeric fields support range queries and sorting on numeric data.

.. autoclass:: NumericField
   :members:
   :show-inheritance:

.. autoclass:: NumericFieldAttributes
   :members:
   :undoc-members:

Geo Fields
----------

Geo fields enable location-based search with geographic coordinates.

.. autoclass:: GeoField
   :members:
   :show-inheritance:

.. autoclass:: GeoFieldAttributes
   :members:
   :undoc-members:

Vector Field Types
==================

Vector fields enable semantic similarity search using various algorithms. All vector fields share common attributes but have algorithm-specific configurations.

Common Vector Attributes
------------------------

All vector field types share these base attributes:

.. autoclass:: BaseVectorFieldAttributes
   :members:
   :undoc-members:

**Key Attributes:**

- `dims`: Dimensionality of the vector (e.g., 768, 1536).
- `algorithm`: Indexing algorithm for vector search:

  - `flat`: Brute-force exact search. 100% recall, slower for large datasets. Best for <10K vectors.
  - `hnsw`: Graph-based approximate search. Fast with high recall (95-99%). Best for general use.
  - `svs-vamana`: SVS-VAMANA (Scalable Vector Search with VAMANA graph algorithm) provides fast approximate nearest neighbor search with optional compression support. This algorithm is optimized for Intel hardware and offers reduced memory usage through vector compression.

  .. note::
     For detailed algorithm comparison and selection guidance, see :ref:`vector-algorithm-comparison`.

- `datatype`: Float precision (`bfloat16`, `float16`, `float32`, `float64`). Note: SVS-VAMANA only supports `float16` and `float32`.
- `distance_metric`: Similarity metric (`COSINE`, `L2`, `IP`).
- `initial_cap`: Initial capacity hint for memory allocation (optional).
- `index_missing`: When True, allows searching for documents missing this field (optional).

HNSW Vector Fields
------------------

HNSW (Hierarchical Navigable Small World) - Graph-based approximate search with excellent recall. **Best for general-purpose vector search (10K-1M+ vectors).**

.. dropdown:: When to use HNSW & Performance Details
   :color: info

   **Use HNSW when:**

    - Medium to large datasets (10K-1M+ vectors) requiring high recall rates
    - Search accuracy is more important than memory usage
    - Need general-purpose vector search with balanced performance
    - Cross-platform deployments where hardware-specific optimizations aren't available

   **Performance characteristics:**

    - **Search speed**: Very fast approximate search with tunable accuracy
    - **Memory usage**: Higher than compressed SVS-VAMANA but reasonable for most applications
    - **Recall quality**: Excellent recall rates (95-99%), often better than other approximate methods
    - **Build time**: Moderate construction time, faster than SVS-VAMANA for smaller datasets

.. autoclass:: HNSWVectorField
   :members:
   :show-inheritance:

.. autoclass:: HNSWVectorFieldAttributes
   :members:
   :undoc-members:

**HNSW Examples:**

**Balanced configuration (recommended starting point):**

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: hnsw
        dims: 768
        distance_metric: cosine
        datatype: float32
        # Balanced settings for good recall and performance
        m: 16
        ef_construction: 200
        ef_runtime: 10

**High-recall configuration:**

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: hnsw
        dims: 768
        distance_metric: cosine
        datatype: float32
        # Tuned for maximum accuracy
        m: 32
        ef_construction: 400
        ef_runtime: 50

SVS-VAMANA Vector Fields
------------------------

SVS-VAMANA (Scalable Vector Search with VAMANA graph algorithm) provides fast approximate nearest neighbor search with optional compression support. This algorithm is optimized for Intel hardware and offers reduced memory usage through vector compression. **Best for large datasets (>100K vectors) on Intel hardware with memory constraints.**

.. dropdown:: When to use SVS-VAMANA & Detailed Guide
   :color: info

   **Requirements:**
    - Redis >= 8.2.0 with RediSearch >= 2.8.10
    - datatype must be 'float16' or 'float32' (float64/bfloat16 not supported)

   **Use SVS-VAMANA when:**
    - Large datasets where memory is expensive
    - Cloud deployments with memory-based pricing
    - When 90-95% recall is acceptable
    - High-dimensional vectors (>1024 dims) with LeanVec compression

   **Performance vs other algorithms:**
    - **vs FLAT**: Much faster search, significantly lower memory usage with compression, but approximate results

    - **vs HNSW**: Better memory efficiency with compression, similar or better recall, Intel-optimized

   **Compression selection guide:**

    - **No compression**: Best performance, standard memory usage

    - **LVQ4/LVQ8**: Good balance of compression (2x-4x) and performance

    - **LeanVec4x8/LeanVec8x8**: Maximum compression (up to 8x) with dimensionality reduction

   **Memory Savings Examples (1M vectors, 768 dims):**
    - No compression (float32): 3.1 GB

    - LVQ4x4 compression: 1.6 GB (~48% savings)

    - LeanVec4x8 + reduce to 384: 580 MB (~81% savings)

.. autoclass:: SVSVectorField
   :members:
   :show-inheritance:

.. autoclass:: SVSVectorFieldAttributes
   :members:
   :undoc-members:

**SVS-VAMANA Examples:**

**Basic configuration (no compression):**

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: svs-vamana
        dims: 768
        distance_metric: cosine
        datatype: float32
        # Standard settings for balanced performance
        graph_max_degree: 40
        construction_window_size: 250
        search_window_size: 20

**High-performance configuration with compression:**

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: svs-vamana
        dims: 768
        distance_metric: cosine
        datatype: float32
        # Tuned for better recall
        graph_max_degree: 64
        construction_window_size: 500
        search_window_size: 40
        # Maximum compression with dimensionality reduction
        compression: LeanVec4x8
        reduce: 384  # 50% dimensionality reduction
        training_threshold: 1000

**Important Notes:**

- **Requirements**: SVS-VAMANA requires Redis >= 8.2 with RediSearch >= 2.8.10.
- **Datatype limitations**: SVS-VAMANA only supports `float16` and `float32` datatypes (not `bfloat16` or `float64`).
- **Compression compatibility**: The `reduce` parameter is only valid with LeanVec compression types (`LeanVec4x8` or `LeanVec8x8`).
- **Platform considerations**: Intel's proprietary LVQ and LeanVec optimizations are not available in Redis Open Source. On non-Intel platforms and Redis Open Source, SVS-VAMANA with compression falls back to basic 8-bit scalar quantization.
- **Performance tip**: Start with default parameters and tune `search_window_size` first for your speed vs accuracy requirements.

FLAT Vector Fields
------------------

FLAT - Brute-force exact search. **Best for small datasets (<10K vectors) requiring 100% accuracy.**

.. dropdown:: When to use FLAT & Performance Details
   :color: info

   **Use FLAT when:**
    - Small datasets (<100K vectors) where exact results are required
    - Search accuracy is critical and approximate results are not acceptable
    - Baseline comparisons when evaluating approximate algorithms
    - Simple use cases where setup simplicity is more important than performance

   **Performance characteristics:**
    - **Search accuracy**: 100% exact results (no approximation)
    - **Search speed**: Linear time O(n) - slower as dataset grows
    - **Memory usage**: Minimal overhead, stores vectors as-is
    - **Build time**: Fastest index construction (no preprocessing)

   **Trade-offs vs other algorithms:**
    - **vs HNSW**: Much slower search but exact results, faster index building
    - **vs SVS-VAMANA**: Slower search and higher memory usage, but exact results

.. autoclass:: FlatVectorField
   :members:
   :show-inheritance:

.. autoclass:: FlatVectorFieldAttributes
   :members:
   :undoc-members:

**FLAT Example:**

.. code-block:: yaml

    - name: embedding
      type: vector
      attrs:
        algorithm: flat
        dims: 768
        distance_metric: cosine
        datatype: float32
        # Optional: tune for batch processing
        block_size: 1024

**Note**: FLAT is recommended for small datasets or when exact results are mandatory. For larger datasets, consider HNSW or SVS-VAMANA for better performance.

SVS-VAMANA Configuration Utilities
==================================

For SVS-VAMANA indices, RedisVL provides utilities to help configure compression settings and estimate memory savings.

CompressionAdvisor
------------------

.. currentmodule:: redisvl.utils.compression

.. autoclass:: CompressionAdvisor
   :members:
   :show-inheritance:

SVSConfig
---------

.. autoclass:: SVSConfig
   :members:
   :show-inheritance:

.. _vector-algorithm-comparison:

Vector Algorithm Comparison
===========================

This section provides detailed guidance for choosing between vector search algorithms.

Algorithm Selection Guide
-------------------------

.. list-table:: Vector Algorithm Comparison
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Algorithm
     - Best For
     - Performance
     - Memory Usage
     - Trade-offs
   * - **FLAT**
     - Small datasets (<100K vectors)
     - 100% recall, O(n) search
     - Minimal overhead
     - Exact but slow for large data
   * - **HNSW**
     - General purpose (100K-1M+ vectors)
     - 95-99% recall, O(log n) search
     - Moderate (graph overhead)
     - Fast approximate search
   * - **SVS-VAMANA**
     - Large datasets with memory constraints
     - 90-95% recall, O(log n) search
     - Low (with compression)
     - Intel-optimized, requires newer Redis

When to Use Each Algorithm
--------------------------

**Choose FLAT when:**
 - Dataset size < 100,000 vectors
 - Exact results are mandatory
 - Simple setup is preferred
 - Query latency is not critical

**Choose HNSW when:**
 - Dataset size 100K - 1M+ vectors
 - Need balanced speed and accuracy
 - Cross-platform compatibility required
 - Most common choice for production

**Choose SVS-VAMANA when:**
 - Dataset size > 100K vectors
 - Memory usage is a primary concern
 - Running on Intel hardware
 - Can accept 90-95% recall for memory savings

Performance Characteristics
---------------------------

**Search Speed:**
 - FLAT: Linear time O(n) - gets slower as data grows
 - HNSW: Logarithmic time O(log n) - scales well
 - SVS-VAMANA: Logarithmic time O(log n) - scales well

**Memory Usage (1M vectors, 768 dims, float32):**
 - FLAT: ~3.1 GB (baseline)
 - HNSW: ~3.7 GB (20% overhead for graph)
 - SVS-VAMANA: 1.6-3.1 GB (depends on compression)

**Recall Quality:**
 - FLAT: 100% (exact search)
 - HNSW: 95-99% (tunable via ef_runtime)
 - SVS-VAMANA: 90-95% (depends on compression)

Migration Considerations
------------------------

**From FLAT to HNSW:**
 - Straightforward migration
 - Expect slight recall reduction but major speed improvement
 - Tune ef_runtime to balance speed vs accuracy

**From HNSW to SVS-VAMANA:**
 - Requires Redis >= 8.2 with RediSearch >= 2.8.10
 - Change datatype to float16 or float32 if using others
 - Consider compression options for memory savings

**From SVS-VAMANA to others:**
 - May need to change datatype back if using float64/bfloat16
 - HNSW provides similar performance with broader compatibility

For complete Redis field documentation, see: https://redis.io/commands/ft.create/
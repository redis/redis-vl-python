---
description: Query builders for RedisVL: vector, range, hybrid, text, filter, count, multi-vector, and SQL.
---

# Query

Query classes provide a structured way to define simple or complex queries for
different use cases. Each class wraps the underlying redis-py
[Query module](https://github.com/redis/redis-py/blob/master/redis/commands/search/query.py)
with extended functionality for ease of use.

## VectorQuery

::: redisvl.query.VectorQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

!!! note "Runtime parameters for performance tuning"

    `VectorQuery` supports runtime parameters for HNSW and SVS-VAMANA indexes
    that can be adjusted at query time without rebuilding the index.

    **HNSW:** `ef_runtime` controls search accuracy (higher = better recall,
    slower search).

    **SVS-VAMANA:** `search_window_size`, `use_search_history`, and
    `search_buffer_capacity`.

    ```python
    from redisvl.query import VectorQuery

    query = VectorQuery(
        vector=[0.1, 0.2, 0.3],
        vector_field_name="embedding",
        num_results=10,
        ef_runtime=150,  # higher = better recall
    )
    ```

## VectorRangeQuery

::: redisvl.query.VectorRangeQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

!!! note "Runtime parameters for range queries"

    `VectorRangeQuery` supports `epsilon` (HNSW & SVS-VAMANA) plus the
    SVS-VAMANA-specific `search_window_size`, `use_search_history`, and
    `search_buffer_capacity`.

## AggregateHybridQuery

::: redisvl.query.AggregateHybridQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

!!! note

    `AggregateHybridQuery` uses `FT.AGGREGATE` and does **not** support runtime
    parameters. For `ef_runtime` / `search_window_size` etc. use
    [`HybridQuery`](#hybridquery), [`VectorQuery`](#vectorquery), or
    [`VectorRangeQuery`](#vectorrangequery).

!!! note

    `HybridQuery` and `AggregateHybridQuery` apply linear combination
    inconsistently. `HybridQuery` uses `linear_alpha` to weight the text
    score, while `AggregateHybridQuery` uses `alpha` to weight the vector
    score. Take care when switching between them.

## HybridQuery

::: redisvl.query.hybrid.HybridQuery
    options:
      show_root_heading: true
      inherited_members: true

## TextQuery

::: redisvl.query.TextQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

## FilterQuery

::: redisvl.query.FilterQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

## CountQuery

::: redisvl.query.CountQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

## MultiVectorQuery

::: redisvl.query.MultiVectorQuery
    options:
      show_root_heading: true
      inherited_members: true
      filters:
        - "!^_"
        - "!^add_filter$"
        - "!^get_args$"
        - "!^highlight$"
        - "!^return_field$"
        - "!^summarize$"

## SQLQuery

::: redisvl.query.SQLQuery
    options:
      show_root_heading: true

!!! note

    `SQLQuery` requires the optional `sql-redis` package. Install with
    `pip install redisvl[sql-redis]`.

    It accepts a `sql_redis_options` dictionary that is passed through to the
    `sql-redis` executor. The most common option is `schema_cache_strategy`:
    `"lazy"` (default) loads schemas on demand, `"load_all"` loads them all up
    front.

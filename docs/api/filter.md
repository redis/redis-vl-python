---
description: Filter expressions for RedisVL queries.
---

# Filter

Filter expressions are composable predicates over indexed fields. Combine them
with `&` and `|` to build complex `WHERE`-style conditions for any RedisVL
query.

## FilterExpression

::: redisvl.query.filter.FilterExpression
    options:
      show_root_heading: true

## Tag

::: redisvl.query.filter.Tag
    options:
      show_root_heading: true
      filters:
        - "!^__hash__$"

## Text

::: redisvl.query.filter.Text
    options:
      show_root_heading: true
      filters:
        - "!^__hash__$"

## Num

::: redisvl.query.filter.Num
    options:
      show_root_heading: true
      filters:
        - "!^__hash__$"

## Geo

::: redisvl.query.filter.Geo
    options:
      show_root_heading: true
      filters:
        - "!^__hash__$"

## GeoRadius

::: redisvl.query.filter.GeoRadius
    options:
      show_root_heading: true
      filters:
        - "!^__hash__$"

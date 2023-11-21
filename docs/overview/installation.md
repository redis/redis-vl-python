---
myst:
  html_meta:
    "description lang=en": |
      Installation instructions for RedisVL
---


## Install RedisVL

Install `redisvl` into your Python (>=3.8) environment using `pip`:

```bash
pip install redisvl
```


## Choosing a Redis Instance


### Redis Cloud

Then make sure to have [Redis](https://redis.io) accessible with Search & Query features enabled on [Redis Cloud](https://redis.com/try-free) or locally in docker with [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/):

### Redis Stack (local)


```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This will also spin up the [Redis Insight GUI](https://redis.com/redis-enterprise/redis-insight/) at `http://localhost:8001`.
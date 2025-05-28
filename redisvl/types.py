from typing import Union

from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.client import Pipeline as SyncPipeline
from redis.cluster import ClusterPipeline as SyncClusterPipeline
from redis.cluster import RedisCluster as SyncRedisCluster

SyncRedisClient = Union[SyncRedis, SyncRedisCluster]
AsyncRedisClient = Union[AsyncRedis, AsyncRedisCluster]
RedisClient = Union[SyncRedisClient, AsyncRedisClient]

SyncRedisPipeline = Union[SyncPipeline, SyncClusterPipeline]
AsyncRedisPipeline = Union[AsyncPipeline, AsyncClusterPipeline]

RedisClientOrPipeline = Union[SyncRedisClient, SyncRedisPipeline]
AsyncRedisClientOrPipeline = Union[AsyncRedisClient, AsyncRedisPipeline]

from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.client import Pipeline as SyncPipeline
from redis.cluster import ClusterPipeline as SyncClusterPipeline
from redis.cluster import RedisCluster as SyncRedisCluster

SyncRedisClient = SyncRedis | SyncRedisCluster
AsyncRedisClient = AsyncRedis | AsyncRedisCluster
RedisClient = SyncRedisClient | AsyncRedisClient

SyncRedisPipeline = SyncPipeline | SyncClusterPipeline
AsyncRedisPipeline = AsyncPipeline | AsyncClusterPipeline

RedisClientOrPipeline = SyncRedisClient | SyncRedisPipeline
AsyncRedisClientOrPipeline = AsyncRedisClient | AsyncRedisPipeline

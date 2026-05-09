# [DevBounty AI]: File optimized for resolution.


```python
class SearchIndex:
    # ...

    def drop_keys(self, keys):
        if isinstance(self._redis_client, RedisCluster) and not _keys_share_hash_tag(keys):
            raise ValueError("All keys must share a hash tag when using Redis Cluster.")
        # ... rest of the method implementation ...
# TODO add username and ACL/TCL support


def get_redis_connection(host: str, port: int, password: str = None):
    from redis import Redis

    connection_args = {"host": host, "port": port}
    if password:
        connection_args.update({"password": password})
    client = Redis(**connection_args)
    return client


# should this be async?
def get_async_redis_connection(host: str, port: int, password: str = None):
    from redis.asyncio import Redis as ARedis

    connection_args = {"host": host, "port": port}
    if password:
        connection_args.update({"password": password})
    return ARedis(**connection_args)

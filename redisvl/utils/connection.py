def get_async_redis_connection(host: str, port: int, password: str = None):
    # TODO add username and ACL/TCL support
    from redis.asyncio import Redis

    connection_args = {"host": host, "port": port}
    if password:
        connection_args.update({"password": password})
    return Redis(**connection_args)

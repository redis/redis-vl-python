import os
from argparse import Namespace


def create_redis_url(args: Namespace) -> str:
    env_address = os.getenv("REDIS_ADDRESS")
    if env_address:
        return env_address
    else:
        url = "redis://"
        if args.ssl:
            url += "rediss://"
        if args.user:
            url += args.user
            if args.password:
                url += ":" + args.password
            url += "@"
        url += args.host + ":" + str(args.port)
        return url

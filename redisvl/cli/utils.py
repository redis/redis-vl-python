import os
from argparse import Namespace


def create_redis_url(args: Namespace) -> str:
    if os.getenv("REDIS_ADDRESS"):
        return os.getenv("REDIS_ADDRESS")
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

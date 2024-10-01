class RedisVLException(Exception):
    """Base RedisVL exception"""


class RedisModuleVersionError(RedisVLException):
    """Invalid module versions installed"""

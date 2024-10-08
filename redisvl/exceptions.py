class RedisVLException(Exception):
    """Base RedisVL exception"""


class RedisModuleVersionError(RedisVLException):
    """Invalid module versions installed"""


class RedisSearchError(RedisVLException):
    """Error while performing a search or aggregate request"""

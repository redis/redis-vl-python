# required modules
DEFAULT_REQUIRED_MODULES = [
    {"name": "search", "ver": 20600},
    {"name": "searchlight", "ver": 20600},
]

# SVS-VAMANA requires Redis 8.2+ with RediSearch 2.8.10+
SVS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20810},  # RediSearch 2.8.10+
    {"name": "searchlight", "ver": 20810},
]

# Minimum Redis version for SVS-VAMANA
SVS_MIN_REDIS_VERSION = "8.2.0"
# Minimum search module version for SVS-VAMANA (2.8.10)
SVS_MIN_SEARCH_VERSION = 20810

# default tag separator
REDIS_TAG_SEPARATOR = ","


REDIS_URL_ENV_VAR = "REDIS_URL"

import logging
import sys

# Set up custom logging
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)s] %(message)s"
    )
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.handlers = [handler]
root_logger.setLevel(logging.INFO)

# Log before import
app_logger = logging.getLogger("app")
app_logger.info("PRE_IMPORT_FORMAT")

# Import RedisVL
from redisvl.query.filter import Text  # noqa: E402, F401

# Log after import
app_logger.info("POST_IMPORT_FORMAT")

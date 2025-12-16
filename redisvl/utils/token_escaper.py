import re
from typing import Optional, Pattern


class TokenEscaper:
    """Escape punctuation within an input string.

    Adapted from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    # Same as above but excludes * to allow wildcard patterns
    ESCAPED_CHARS_NO_WILDCARD = r"[,.<>{}\[\]\\\"\':;!@#$%^&()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)
        self.escaped_chars_no_wildcard_re = re.compile(self.ESCAPED_CHARS_NO_WILDCARD)

    def escape(self, value: str, preserve_wildcards: bool = False) -> str:
        """Escape special characters in a string for use in Redis queries.

        Args:
            value: The string value to escape.
            preserve_wildcards: If True, preserves * characters for wildcard
                matching. Defaults to False.

        Returns:
            The escaped string.

        Raises:
            TypeError: If value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError(
                f"Value must be a string object for token escaping, got type {type(value)}"
            )

        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        if preserve_wildcards:
            return self.escaped_chars_no_wildcard_re.sub(escape_symbol, value)
        return self.escaped_chars_re.sub(escape_symbol, value)

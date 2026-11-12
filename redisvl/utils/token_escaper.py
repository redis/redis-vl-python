import re
from typing import Pattern


def _char_class(chars: str) -> str:
    return "[" + "".join(re.escape(char) for char in chars) + "]"


# Building blocks for the escape sets in `TokenEscaper`.
_QUERY_SYNTAX = ",.<>{}[]\\\"':;!@#$%^&()-+=~/ "
_WILDCARDS = "*?"
# The query parser reads these as a term break inside a tag value unless they
# are escaped: on an unescaped query `@f:{a`b}` matches a document tagged `a b`,
# `!=` excludes it, and the pattern `a`*` is a syntax error. The other C0
# controls and DEL (0x01-0x08, 0x0E-0x1F, 0x7F) break the term even when
# escaped, so adding them here wouldn't make such values match.
_TAG_SEPARATORS = "`\t\n\r\x0b\x0c"
# A HASH index trims whitespace from both ends of a tag value when it indexes
# it, and a JSON index keeps it. The escaper can't tell which storage it is
# querying, so these stay unescaped at the ends of a query value: the query
# drops them the same way and matches the trimmed tag on HASH, while a JSON
# value that starts or ends with one of them can't be matched. Spaces go the
# other way: they stay escaped, so " sp " matches on JSON but not on HASH,
# where it is indexed as "sp".
_TAG_EDGE_WHITESPACE = "\t\n\r\x0b\x0c"


class TokenEscaper:
    """Escape punctuation within an input string.

    Adapted from RedisOM Python.
    """

    # Characters that Redis Search requires us to escape during queries.
    # Source: https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/escaping/
    DEFAULT_ESCAPED_CHARS = _char_class(_QUERY_SYNTAX + _WILDCARDS)

    # Same as above but excludes * and ? to allow wildcard patterns
    ESCAPED_CHARS_NO_WILDCARD = _char_class(_QUERY_SYNTAX)

    # Tag equality (`==`, `!=`): every character inside the value is literal,
    # including `|` (see `_TAG_EDGE_WHITESPACE` for the ends of a value). A tag
    # clause holds its alternatives directly in braces -- `@f:{a|b}` -- so an
    # unescaped `|` inside a single value silently turns an equality match into
    # a union.
    #
    # `|` is deliberately not in the default set: a text query string joins its
    # own terms with `|`, so escaping it there turns a documented union into a
    # literal that matches nothing.
    TAG_VALUE_ESCAPED_CHARS = _char_class(
        _QUERY_SYNTAX + _WILDCARDS + "|" + _TAG_SEPARATORS
    )

    # Tag patterns (`%`): `*`, `?` and `|` stay live, because `Tag.__mod__`
    # documents `|` as a union between patterns.
    TAG_PATTERN_ESCAPED_CHARS = _char_class(_QUERY_SYNTAX + _TAG_SEPARATORS)

    # The tag equality set under its earlier name.
    TAG_ESCAPED_CHARS = TAG_VALUE_ESCAPED_CHARS

    # Whitespace left unescaped at the ends of a value; only `for_tags()` sets it.
    _raw_edge_chars: str = ""

    def __init__(self, escape_chars_re: Pattern | None = None):
        if escape_chars_re is not None:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)
        self.escaped_chars_no_wildcard_re = re.compile(self.ESCAPED_CHARS_NO_WILDCARD)

    @classmethod
    def for_tags(cls) -> "TokenEscaper":
        """Return the escaper for tag values and tag patterns.

        ``escape()`` uses ``TAG_VALUE_ESCAPED_CHARS``, or
        ``TAG_PATTERN_ESCAPED_CHARS`` when ``preserve_wildcards`` is True. Tabs,
        newlines and the other ASCII whitespace except the space are left
        unescaped at either end of the value, because a HASH index trims them
        from tag values when it indexes them. A JSON index keeps them, so a value
        that starts or ends with one of them can't be matched there.
        """
        escaper = cls(re.compile(cls.TAG_VALUE_ESCAPED_CHARS))
        escaper.escaped_chars_no_wildcard_re = re.compile(cls.TAG_PATTERN_ESCAPED_CHARS)
        escaper._raw_edge_chars = _TAG_EDGE_WHITESPACE
        return escaper

    def escape(self, value: str, preserve_wildcards: bool = False) -> str:
        """Escape special characters in a string for use in Redis queries.

        Args:
            value: The string value to escape.
            preserve_wildcards: If True, leaves ``*`` and ``?`` unescaped for
                wildcard matching, and for the tag escaper also ``|`` as a
                union between patterns. Defaults to False.

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

        pattern = (
            self.escaped_chars_no_wildcard_re
            if preserve_wildcards
            else self.escaped_chars_re
        )
        if not self._raw_edge_chars:
            return pattern.sub(escape_symbol, value)

        core = value.strip(self._raw_edge_chars)
        if not core:
            return value
        start = len(value) - len(value.lstrip(self._raw_edge_chars))
        end = start + len(core)
        return value[:start] + pattern.sub(escape_symbol, core) + value[end:]

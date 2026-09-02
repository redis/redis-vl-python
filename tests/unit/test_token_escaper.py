import pytest

from redisvl.utils.token_escaper import TokenEscaper


@pytest.fixture
def escaper():
    return TokenEscaper()


@pytest.mark.parametrize(
    ("test_input,expected"),
    [
        (r"a [big] test.", r"a\ \[big\]\ test\."),
        (r"hello, world!", r"hello\,\ world\!"),
        (
            r'special "quotes" (and parentheses)',
            r"special\ \"quotes\"\ \(and\ parentheses\)",
        ),
        (
            r"& symbols, like * and ?",
            r"\&\ symbols\,\ like\ \*\ and\ \?",
        ),
        # underscores are ignored
        (r"-dashes_and_underscores-", r"\-dashes_and_underscores\-"),
    ],
    ids=["brackets", "commas", "quotes", "symbols", "underscores"],
)
def test_escape_text_chars(escaper, test_input, expected):
    assert escaper.escape(test_input) == expected


@pytest.mark.parametrize(
    ("test_input,expected"),
    [
        # Simple tags
        ("user:name", r"user\:name"),
        ("123#comment", r"123\#comment"),
        ("hyphen-separated", r"hyphen\-separated"),
        # Tags with special characters
        ("price$", r"price\$"),
        ("super*star", r"super\*star"),
        ("tag&value", r"tag\&value"),
        ("@username", r"\@username"),
        # Space-containing tags often used in search scenarios
        ("San Francisco", r"San\ Francisco"),
        ("New Zealand", r"New\ Zealand"),
        # Multi-special-character tags
        ("complex/tag:value", r"complex\/tag\:value"),
        ("$special$tag$", r"\$special\$tag\$"),
        ("tag-with-hyphen", r"tag\-with\-hyphen"),
        # Tags with less common, but legal characters
        ("_underscore_", r"_underscore_"),
        ("dot.tag", r"dot\.tag"),
        # ("pipe|tag", r"pipe\|tag"), #TODO - pipes are not caught?
        # More edge cases with special characters
        ("(parentheses)", r"\(parentheses\)"),
        ("[brackets]", r"\[brackets\]"),
        ("{braces}", r"\{braces\}"),
        ("question?mark", r"question\?mark"),
        # Unicode characters in tags
        ("你好", r"你好"),  # Assuming non-Latin characters don't need escaping
        ("emoji:😊", r"emoji\:😊"),
        # ...other cases as needed...
    ],
    ids=[
        ":",
        "#",
        "-",
        "$",
        "*",
        "&",
        "@",
        "space",
        "space-2",
        "complex",
        "special",
        "hyphen",
        "underscore",
        "dot",
        "parentheses",
        "brackets",
        "braces",
        "question",
        "non-latin",
        "emoji",
    ],
)
def test_escape_tag_like_values(escaper, test_input, expected):
    assert escaper.escape(test_input) == expected


@pytest.mark.parametrize("test_input", [123, 45.67, None, [], {}])
def test_escape_non_string_input(escaper, test_input):
    with pytest.raises(TypeError):
        escaper.escape(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # ('你好，世界！', r'你好\，世界\！'), # TODO - non latin chars?
        ("😊 ❤️ 👍", r"😊\ ❤️\ 👍"),
        # ...other cases as needed...
    ],
    ids=["emoji"],
)
def test_escape_unicode_characters(escaper, test_input, expected):
    assert escaper.escape(test_input) == expected


def test_escape_empty_string(escaper):
    assert escaper.escape("") == ""


def test_escape_long_string(escaper):
    # Construct a very long string
    long_str = "a," * 1000  # This creates a string "a,a,a,a,...a,"
    expected = r"a\," * 1000  # Expected escaped string

    # Use pytest's benchmark fixture to check performance
    escaped = escaper.escape(long_str)
    assert escaped == expected


@pytest.mark.parametrize(
    ("test_input,expected"),
    [
        ("acme|victim", r"acme\|victim"),
        ("a|b|c", r"a\|b\|c"),
        ("|leading", r"\|leading"),
    ],
    ids=["pair", "chain", "leading"],
)
def test_escape_pipe_on_the_default_path(escaper, test_input, expected):
    # A `|` is RediSearch's union operator and binds looser than the implicit
    # intersection, so one reaching the query string raw widens its clause.
    assert escaper.escape(test_input) == expected


def test_pipe_stays_live_when_wildcards_are_preserved(escaper):
    # The `%` operator documents `|` as a union between wildcard patterns, so
    # the no-wildcard class must not escape it.
    assert escaper.escape("elec*|*soft", preserve_wildcards=True) == "elec*|*soft"

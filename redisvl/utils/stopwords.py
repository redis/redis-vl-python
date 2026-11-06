"""Static, bundled stopword lists.

These were extracted once from NLTK's ``stopwords`` corpus so RedisVL can filter
query stopwords without depending on (and downloading data for) the whole NLTK
package. The word lists are small and change rarely; see ``stopwords.json``.
"""

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _stopwords_by_language() -> dict[str, list[str]]:
    data = resources.files(__package__).joinpath("stopwords.json").read_text("utf-8")
    return json.loads(data)


def get_stopwords(language: str) -> set[str]:
    """Return the bundled stopword set for ``language`` (e.g. "english").

    Raises:
        ValueError: If no stopwords are bundled for the given language.
    """
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    words = _stopwords_by_language()
    if language not in words:
        raise ValueError(
            f"No stopwords available for '{language}'. "
            f"Available languages: {sorted(words)}"
        )
    return set(words[language])

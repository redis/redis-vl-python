from typing import Dict, List, Optional, Set, Tuple, Union

from redisvl.query.filter import FilterExpression
from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.utils import lazy_import

nltk = lazy_import("nltk")
nltk_stopwords = lazy_import("nltk.corpus.stopwords")


def _parse_text_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    parsed_weights: Dict[str, float] = {}
    if not weights:
        return parsed_weights
    for word, weight in weights.items():
        word = word.strip().lower()
        if not word or " " in word:
            raise ValueError(
                f"Only individual words may be weighted. Got {{ {word}:{weight} }}"
            )
        if not (isinstance(weight, float) or isinstance(weight, int)) or weight < 0.0:
            raise ValueError(
                f"Weights must be positive number. Got {{ {word}:{weight} }}"
            )
        parsed_weights[word] = weight
    return parsed_weights


class FullTextQueryHelper:
    """Convert raw user queries into Redis full-text queries - tokenizes, escapes, and filters stopwords from the query."""

    def __init__(
        self,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        text_weights: Optional[Dict[str, float]] = None,
    ):
        self._stopwords = self._get_stopwords(stopwords)
        self._text_weights = _parse_text_weights(text_weights)

    @property
    def stopwords(self) -> Set[str]:
        """Return the stopwords used in the query.
        Returns:
            Set[str]: The stopwords used in the query.
        """
        return self._stopwords.copy() if self._stopwords else set()

    @property
    def text_weights(self) -> Dict[str, float]:
        """Get the text weights.

        Returns:
            Dictionary of word:weight mappings.
        """
        return self._text_weights

    def build_query_string(
        self,
        text: str,
        text_field_name: str,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
    ) -> str:
        """Build the full-text query string for text search with optional filtering."""
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)

        query = f"(~@{text_field_name}:({self._tokenize_and_escape_query(text)})"

        if filter_expression and filter_expression != "*":
            query += f" AND {filter_expression}"

        return query + ")"

    def _get_stopwords(
        self, stopwords: Optional[Union[str, Set[str]]] = "english"
    ) -> Set[str]:
        """Get the stopwords to use in the query.

        Args:
            stopwords (Optional[Union[str, Set[str]]]): The stopwords to use. If a string
                such as "english" "german" is provided then a default set of stopwords for that
                language will be used. if a list, set, or tuple of strings is provided then those
                will be used as stopwords. Defaults to "english". if set to "None" then no stopwords
                will be removed.

        Returns:
            The set of stopwords to use.

        Raises:
            TypeError: If the stopwords are not a set, list, or tuple of strings.
        """
        if not stopwords:
            return set()
        elif isinstance(stopwords, str):
            try:
                nltk.download("stopwords", quiet=True)
                return set(nltk_stopwords.words(stopwords))
            except ImportError:
                raise ValueError(
                    f"Loading stopwords for {stopwords} failed: nltk is not installed."
                )
            except Exception as e:
                raise ValueError(f"Error trying to load {stopwords} from nltk. {e}")
        elif isinstance(stopwords, (Set, List, Tuple)) and all(  # type: ignore
            isinstance(word, str) for word in stopwords
        ):
            return set(stopwords)
        else:
            raise TypeError("stopwords must be a set, list, or tuple of strings")

    def set_text_weights(self, weights: Dict[str, float]):
        """Set or update the text weights for the query.

        Args:
            weights: Dictionary of word:weight mappings
        """
        self._text_weights = _parse_text_weights(weights)

    def _tokenize_and_escape_query(self, user_query: str) -> str:
        """Convert a raw user query to a redis full text query joined by ORs

        Args:
            user_query (str): The user query to tokenize and escape.

        Returns:
            str: The tokenized and escaped query string.

        Raises:
            ValueError: If the text string becomes empty after stopwords are removed.
        """
        escaper = TokenEscaper()

        tokens = [
            escaper.escape(
                token.strip().strip(",").replace("“", "").replace("”", "").lower()
            )
            for token in user_query.split()
        ]

        token_list = [
            token for token in tokens if token and token not in self._stopwords
        ]
        for i, token in enumerate(token_list):
            if token in self._text_weights:
                token_list[i] = f"{token}=>{{$weight:{self._text_weights[token]}}}"

        if not token_list:
            raise ValueError("text string cannot be empty after removing stopwords")
        return " | ".join(token_list)

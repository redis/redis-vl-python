from typing import Any, Dict, List, Optional, Set, Tuple, Union

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from redis.commands.search.aggregation import AggregateRequest, Desc

from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.token_escaper import TokenEscaper


class AggregationQuery(AggregateRequest):
    """
    Base class for aggregation queries used to create aggregation queries for Redis.
    """

    def __init__(self, query_string):
        super().__init__(query_string)


class HybridQuery(AggregationQuery):
    """
    HybridQuery combines text and vector search in Redis.
    It allows you to perform a hybrid search using both text and vector similarity.
    It scores documents based on a weighted combination of text and vector similarity.

     .. code-block:: python

        from redisvl.query import HybridQuery
        from redisvl.index import SearchIndex

        index = SearchIndex.from_yaml("path/to/index.yaml")

        query = HybridQuery(
            text="example text",
            text_field_name="text_field",
            vector=[0.1, 0.2, 0.3],
            vector_field_name="vector_field",
            text_scorer="BM25STD",
            filter_expression=None,
            alpha=0.7,
            dtype="float32",
            num_results=10,
            return_fields=["field1", "field2"],
            stopwords="english",
            dialect=2,
        )

        results = index.query(query)

    """

    DISTANCE_ID: str = "vector_distance"
    VECTOR_PARAM: str = "vector"

    def __init__(
        self,
        text: str,
        text_field_name: str,
        vector: Union[bytes, List[float]],
        vector_field_name: str,
        text_scorer: str = "BM25STD",
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        alpha: float = 0.7,
        dtype: str = "float32",
        num_results: int = 10,
        return_fields: Optional[List[str]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        dialect: int = 2,
    ):
        """
        Instantiates a HybridQuery object.

        Args:
            text (str): The text to search for.
            text_field_name (str): The text field name to search in.
            vector (Union[bytes, List[float]]): The vector to perform vector similarity search.
            vector_field_name (str): The vector field name to search in.
            text_scorer (str, optional): The text scorer to use. Options are {TFIDF, TFIDF.DOCNORM,
                BM25, DISMAX, DOCSCORE, BM25STD}. Defaults to "BM25STD".
            filter_expression (Optional[FilterExpression], optional): The filter expression to use.
                Defaults to None.
            alpha (float, optional): The weight of the vector similarity. Documents will be scored
                as: hybrid_score = (alpha) * vector_score + (1-alpha) * text_score.
                Defaults to 0.7.
            dtype (str, optional): The data type of the vector. Defaults to "float32".
            num_results (int, optional): The number of results to return. Defaults to 10.
            return_fields (Optional[List[str]], optional): The fields to return. Defaults to None.
            stopwords (Optional[Union[str, Set[str]]], optional): The stopwords to remove from the
                provided text prior to searchuse. If a string such as "english" "german" is
                provided then a default set of stopwords for that language will be used. if a list,
                set, or tuple of strings is provided then those will be used as stopwords.
                Defaults to "english". if set to "None" then no stopwords will be removed.
            dialect (int, optional): The Redis dialect version. Defaults to 2.

        Raises:
            ValueError: If the text string is empty, or if the text string becomes empty after
                stopwords are removed.
            TypeError: If the stopwords are not a set, list, or tuple of strings.
        """

        if not text.strip():
            raise ValueError("text string cannot be empty")

        self._text = text
        self._text_field = text_field_name
        self._vector = vector
        self._vector_field = vector_field_name
        self._filter_expression = filter_expression
        self._alpha = alpha
        self._dtype = dtype
        self._num_results = num_results
        self._set_stopwords(stopwords)

        query_string = self._build_query_string()
        super().__init__(query_string)

        self.scorer(text_scorer)  # type: ignore[attr-defined]
        self.add_scores()  # type: ignore[attr-defined]
        self.apply(
            vector_similarity=f"(2 - @{self.DISTANCE_ID})/2", text_score="@__score"
        )
        self.apply(hybrid_score=f"{1-alpha}*@text_score + {alpha}*@vector_similarity")
        self.sort_by(Desc("@hybrid_score"), max=num_results)
        self.dialect(dialect)  # type: ignore[attr-defined]
        if return_fields:
            self.load(*return_fields)

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the aggregation.

        Returns:
            Dict[str, Any]: The parameters for the aggregation.
        """
        if isinstance(self._vector, bytes):
            vector = self._vector
        else:
            vector = array_to_buffer(self._vector, dtype=self._dtype)

        params = {self.VECTOR_PARAM: vector}

        return params

    @property
    def stopwords(self) -> Set[str]:
        """Return the stopwords used in the query.
        Returns:
            Set[str]: The stopwords used in the query.
        """
        return self._stopwords.copy() if self._stopwords else set()

    def _set_stopwords(self, stopwords: Optional[Union[str, Set[str]]] = "english"):
        """Set the stopwords to use in the query.
        Args:
            stopwords (Optional[Union[str, Set[str]]]): The stopwords to use. If a string
                such as "english" "german" is provided then a default set of stopwords for that
                language will be used. if a list, set, or tuple of strings is provided then those
                will be used as stopwords. Defaults to "english". if set to "None" then no stopwords
                will be removed.
        Raises:
            TypeError: If the stopwords are not a set, list, or tuple of strings.
        """
        if not stopwords:
            self._stopwords = set()
        elif isinstance(stopwords, str):
            try:
                nltk.download("stopwords", quiet=True)
                self._stopwords = set(nltk_stopwords.words(stopwords))
            except Exception as e:
                raise ValueError(f"Error trying to load {stopwords} from nltk. {e}")
        elif isinstance(stopwords, (Set, List, Tuple)) and all(  # type: ignore
            isinstance(word, str) for word in stopwords
        ):
            self._stopwords = set(stopwords)
        else:
            raise TypeError("stopwords must be a set, list, or tuple of strings")

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
        tokenized = " | ".join(
            [token for token in tokens if token and token not in self._stopwords]
        )

        if not tokenized:
            raise ValueError("text string cannot be empty after removing stopwords")
        return tokenized

    def _build_query_string(self) -> str:
        """Build the full query string for text search with optional filtering."""
        if isinstance(self._filter_expression, FilterExpression):
            filter_expression = str(self._filter_expression)
        else:
            filter_expression = ""

        # base KNN query
        knn_query = f"KNN {self._num_results} @{self._vector_field} ${self.VECTOR_PARAM} AS {self.DISTANCE_ID}"

        text = f"(~@{self._text_field}:({self._tokenize_and_escape_query(self._text)})"

        if filter_expression and filter_expression != "*":
            text += f" AND {filter_expression}"

        return f"{text})=>[{knn_query}]"

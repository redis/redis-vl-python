from typing import Any, Dict, List, Optional, Set, Tuple, Union

from redis.commands.search.aggregation import AggregateRequest, Desc

from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.utils import lazy_import

nltk = lazy_import("nltk")
nltk_stopwords = lazy_import("nltk.corpus.stopwords")


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

        self.scorer(text_scorer)
        self.add_scores()
        self.apply(
            vector_similarity=f"(2 - @{self.DISTANCE_ID})/2", text_score="@__score"
        )
        self.apply(hybrid_score=f"{1-alpha}*@text_score + {alpha}*@vector_similarity")
        self.sort_by(Desc("@hybrid_score"), max=num_results)  # type: ignore
        self.dialect(dialect)
        if return_fields:
            self.load(*return_fields)  # type: ignore[arg-type]

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the aggregation.

        Returns:
            Dict[str, Any]: The parameters for the aggregation.
        """
        if isinstance(self._vector, list):
            vector = array_to_buffer(self._vector, dtype=self._dtype)
        else:
            vector = self._vector

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
            except ImportError:
                raise ValueError(
                    f"Loading stopwords for {stopwords} failed: nltk is not installed."
                )
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
        filter_expression = self._filter_expression
        if isinstance(self._filter_expression, FilterExpression):
            filter_expression = str(self._filter_expression)

        # base KNN query
        knn_query = f"KNN {self._num_results} @{self._vector_field} ${self.VECTOR_PARAM} AS {self.DISTANCE_ID}"

        text = f"(~@{self._text_field}:({self._tokenize_and_escape_query(self._text)})"

        if filter_expression and filter_expression != "*":
            text += f" AND {filter_expression}"

        return f"{text})=>[{knn_query}]"

    def __str__(self) -> str:
        """Return the string representation of the query."""
        return " ".join([str(x) for x in self.build_args()])


class MultiVectorQuery(AggregationQuery):
    """
    MultiVectorQuery allows for search over multiple vector fields in a document simulateously.
    The final score will be a weighted combination of the individual vector similarity scores
    following the formula:

    score = (w_1 * score_1 + w_2 * score_2 + w_3 * score_3 + ... )

    Vectors may be of different size and datatype, but must be indexed using the 'cosine' distance_metric.

    .. code-block:: python

        from redisvl.query import MultiVectorQuery
        from redisvl.index import SearchIndex

        index = SearchIndex.from_yaml("path/to/index.yaml")

        query = MultiVectorQuery(
            vectors=[[0.1, 0.2, 0.3], [0.5, 0.5], [0.1, 0.1, 0.1, 0.1]],
            vector_field_names=["text_vector", "image_vector", "feature_vector"]
            filter_expression=None,
            weights=[0.7, 0.2, 0.5],
            dtypes=["float32", "bfloat16", "float64"],
            num_results=10,
            return_fields=["field1", "field2"],
            dialect=2,
        )

        results = index.query(query)
    """

    def __init__(
        self,
        vectors: Union[bytes, List[bytes], List[float], List[List[float]]],
        vector_field_names: Union[str, List[str]],
        weights: List[float] = [1.0],
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        dtypes: List[str] = ["float32"],
        num_results: int = 10,
        return_score: bool = False,
        dialect: int = 2,
    ):
        """
        Instantiates a MultiVectorQuery object.

        Args:
            vectors (Union[bytes, List[bytes], List[float], List[List[float]]): The vectors to perform vector similarity search.
            vector_field_names (Union[str, List[str]]): The vector field names to search in.
            weights (List[float]): The weights of the vector similarity.
                Documents will be scored as:
                score = (w1) * score1 + (w2) * score2 + (w3) * score3 + ...
                Defaults to [1.0], which corresponds to equal weighting
            return_fields (Optional[List[str]], optional): The fields to return. Defaults to None.
            filter_expression (Optional[Union[str, FilterExpression]]): The filter expression to use.
                Defaults to None.
            dtypes (List[str]): The data types of the vectors. Defaults to ["float32"] for all vectors.
            num_results (int, optional): The number of results to return. Defaults to 10.
            return_score (bool): Whether to return the combined vector similarity score.
                Defaults to False.
            dialect (int, optional): The Redis dialect version. Defaults to 2.

        Raises:
            ValueError: The number of vectors, vector field names, and weights do not agree.
        """

        self._filter_expression = filter_expression
        self._dtypes = dtypes
        self._num_results = num_results

        if any([len(x) == 0 for x in [vectors, vector_field_names, weights, dtypes]]):
            raise ValueError(
                f"""The number of vectors and vector field names must be equal.
                    If weights or dtypes are specified their number must match the number of vectors and vector field names also.
                    Length of vectors list: {len(vectors) = }
                    Length of vector_field_names list: {len(vector_field_names) = }
                    Length of weights list: {len(weights) = }
                    length of dtypes list: {len(dtypes) = }
                    """
            )

        if isinstance(vectors, bytes) or isinstance(vectors[0], float):
            self._vectors = [vectors]
        else:
            self._vectors = vectors  # type: ignore

        if isinstance(vector_field_names, str):
            self._vector_field_names = [vector_field_names]
        else:
            self._vector_field_names = vector_field_names

        if len(weights) == 1:
            self._weights = weights * len(vectors)
        else:
            self._weights = weights

        if len(dtypes) == 1:
            self._dtypes = dtypes * len(vectors)
        else:
            self._dtypes = dtypes

        num_vectors = len(self._vectors)
        if any(
            [
                len(x) != num_vectors  # type: ignore
                for x in [self._vector_field_names, self._weights, self._dtypes]
            ]
        ):
            raise ValueError(
                f"""The number of vectors and vector field names must be equal.
                    If weights or dtypes are specified their number must match the number of vectors and vector field names also.
                    Length of vectors list: {len(self._vectors) = }
                    Length of vector_field_names list: {len(self._vector_field_names) = }
                    Length of weights list: {len(self._weights) = }
                    Length of dtypes list: {len(self._dtypes) = }
                    """
            )

        query_string = self._build_query_string()
        super().__init__(query_string)

        # calculate the respective vector similarities
        for i in range(len(vectors)):
            self.apply(**{f"score_{i}": f"(2 - @distance_{i})/2"})

        # construct the scoring string based on the vector similarity scores and weights
        combined_scores = []
        for i, w in enumerate(self._weights):
            combined_scores.append(f"@score_{i} * {w}")
        combined_score_string = " + ".join(combined_scores)

        self.apply(combined_score=combined_score_string)

        self.sort_by(Desc("@combined_score"), max=num_results)  # type: ignore
        self.dialect(dialect)
        if return_fields:
            self.load(*return_fields)  # type: ignore[arg-type]

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the aggregation.

        Returns:
            Dict[str, Any]: The parameters for the aggregation.
        """
        params = {}
        for i, (vector, dtype) in enumerate(zip(self._vectors, self._dtypes)):
            if isinstance(vector, list):
                vector = array_to_buffer(vector, dtype=dtype)  # type: ignore
            params[f"vector_{i}"] = vector
        return params

    def _build_query_string(self) -> str:
        """Build the full query string for text search with optional filtering."""

        # base KNN query
        range_queries = []
        for i, (vector, field) in enumerate(
            zip(self._vectors, self._vector_field_names)
        ):
            range_queries.append(
                f"@{field}:[VECTOR_RANGE 2.0 $vector_{i}]=>{{$YIELD_DISTANCE_AS: distance_{i}}}"
            )

        range_query = " | ".join(range_queries)

        filter_expression = self._filter_expression
        if isinstance(self._filter_expression, FilterExpression):
            filter_expression = str(self._filter_expression)

        if filter_expression:
            return f"({range_query}) AND ({filter_expression})"
        else:
            return f"{range_query}"

    def __str__(self) -> str:
        """Return the string representation of the query."""
        return " ".join([str(x) for x in self.build_args()])

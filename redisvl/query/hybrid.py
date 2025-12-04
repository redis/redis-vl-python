from typing import Any, Dict, List, Literal, Optional, Set, Union

from redisvl.utils.full_text_query_helper import FullTextQueryHelper

try:
    from redis.commands.search.hybrid_query import HybridQuery as _HybridQuery
    from redis.commands.search.hybrid_query import (
        HybridSearchQuery,
        HybridVsimQuery,
        VectorSearchMethods,
    )
except ImportError:
    raise ImportError("Hybrid queries require redis>=7.1.0")

from redisvl.query.filter import FilterExpression


class HybridQuery(_HybridQuery):
    """TBD"""

    def __init__(
        self,
        text: str,
        text_field_name: str,
        vector: Union[bytes, List[float]],
        vector_field_name: str,
        text_scorer: str = "BM25STD",
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        vector_search_method: Optional[Literal["KNN", "RANGE"]] = None,
        vector_search_method_params: Optional[Dict[str, Any]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        text_weights: Optional[Dict[str, float]] = None,
    ):
        self._ft_helper = FullTextQueryHelper(
            stopwords=stopwords,
            text_weights=text_weights,
        )

        search_query = HybridSearchQuery(
            query_string=self._ft_helper.build_query_string(
                text=text,
                text_field_name=text_field_name,
                filter_expression=filter_expression,
            ),
            scorer=text_scorer,
        )

        if not isinstance(vector, bytes):
            vector_data: Union[str, bytes] = str(vector)
        else:
            vector_data = vector

        vsim_search_method = None
        if vector_search_method:
            vsim_search_method = VectorSearchMethods(vector_search_method)

        vsim_query = HybridVsimQuery(
            vector_field_name=vector_field_name,
            vector_data=vector_data,
            vsim_search_method=vsim_search_method,
            vsim_search_method_params=vector_search_method_params,
            # TODO: Implement filter
        )

        super().__init__(
            search_query=search_query,
            vector_similarity_query=vsim_query,
        )

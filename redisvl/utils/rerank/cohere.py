import os
from typing import Any, Dict, List, Optional, Union
from pydantic.v1 import PrivateAttr

from redisvl.utils.rerank.base import BaseReranker


class CohereReranker(BaseReranker):
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        rank_by: Optional[List[str]] = None,
        limit: int = 5,
        return_score: bool = True,
        api_config: Optional[Dict] = None
    ) -> None:
        # Dynamic import of the cohere module
        try:
            from cohere import Client, AsyncClient
        except ImportError:
            raise ImportError(
                "Cohere reranker requires the cohere library. \
                    Please install with `pip install cohere`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("COHERE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Cohere API key is required. "
                "Provide it in api_config or set the COHERE_API_KEY environment variable."
            )

        self._client = Client(api_key=api_key)
        self._aclient = AsyncClient(api_key=api_key)

        super().__init__(
            model=model,
            rank_by=rank_by,
            limit=limit,
            return_score=return_score,
        )

    def _preprocess(
        self,
        query: str,
        results: Union[List[Dict[str, Any]], List[str]],
        **kwargs,
    ):
        # parse optional overrides
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)
        max_chunks_per_doc = kwargs.get("max_chunks_per_doc")
        rank_by = kwargs.get("rank_by", self.rank_by)
        if isinstance(rank_by, str):
            rank_by = [rank_by]

        reranker_kwargs = {
            "model": self.model,
            "query": query,
            "top_n": limit,
            "documents": results,
            "max_chunks_per_doc": max_chunks_per_doc
        }

        if rank_by and all([isinstance(result, dict) for result in results]):
            reranker_kwargs["rank_fields"] = rank_by

        return reranker_kwargs, return_score

    @staticmethod
    def _postprocess(
        results: List[Dict[str, Any]], rankings: List[Any], return_score: bool
    ) -> List[Dict[str, Any]]:
        reranked_results = []
        for item in rankings.results:
            result = results[item.index]
            if return_score:
                result["score"] = item.relevance_score
            reranked_results.append(result)
        return reranked_results

    def rank(
        self,
        query: str,
        results: Union[List[Dict[str, Any]], List[str]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # preprocess inputs
        reranker_kwargs, return_score = self._preprocess(
            query, results, **kwargs
        )
        ranked_results = self._client.rerank(**reranker_kwargs)
        return self._postprocess(results, ranked_results, return_score)

    async def arank(
        self,
        query: str,
        results: Union[List[Dict[str, Any]], List[str]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # preprocess inputs
        reranker_kwargs, return_score = self._preprocess(
            query, results, **kwargs
        )
        ranked_results = await self._client.arerank(**reranker_kwargs)
        return self._postprocess(results, ranked_results, return_score)

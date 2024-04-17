import os
from typing import Any, Dict, List, Optional
from pydantic.v1 import PrivateAttr

from redisvl.utils.rerank.base import BaseReranker


class CohereReranker(BaseReranker):
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "rerank-english-v2.0",
        rank_by: Optional[str] = None,
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

    @staticmethod
    def _preprocess(results: List[Dict[str, Any]], rank_by: str) -> List[str]:
        try:
            docs = [result[rank_by] for result in results]
        except (TypeError, KeyError):
            raise ValueError(
                "Must provide a valid rank_by field option. "
                f"{rank_by} field is not present in the search results"
            )
        return docs

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
        results: List[Dict[str, Any]],
        max_chunks_per_doc: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)
        rank_by = kwargs.get("rank_by", self.rank_by)

        docs = self._preprocess(results, rank_by)

        rankings = self._client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=limit,
            max_chunks_per_doc=max_chunks_per_doc,
        )

        return self._postprocess(results, rankings, return_score)

    async def arank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_chunks_per_doc: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)
        rank_by = kwargs.get("rank_by", self.rank_by)

        docs = self._preprocess(results, rank_by)
        rankings = await self._aclient.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=limit,
            max_chunks_per_doc=max_chunks_per_doc,
        )

        return self._postprocess(results, rankings, return_score)

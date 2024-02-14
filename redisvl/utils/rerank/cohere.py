from typing import Any, Dict, List, Optional

import cohere

from redisvl.utils.rerank.base import BaseReranker


class CohereReranker(BaseReranker):
    def __init__(self, model: str = "rerank-english-v2.0", **data):
        super().__init__(model=model, **data)
        self.client = cohere.Client()
        self.aclient = cohere.AsyncClient()

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
        for item in rankings:
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

        rankings = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=limit,
            return_documents=False,
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

        rankings = await self.aclient.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=limit,
            return_documents=False,
            max_chunks_per_doc=max_chunks_per_doc,
        )

        return self._postprocess(results, rankings, return_score)

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import PrivateAttr

from redisvl.utils.rerank.base import BaseReranker


class HFCrossEncoderReranker(BaseReranker):
    """
    The HFCrossEncoderReranker class uses a cross-encoder models from Hugging Face
    to rerank documents based on an input query.

    This reranker loads a cross-encoder model using the `CrossEncoder` class
    from the `sentence_transformers` library. It requires the
    `sentence_transformers` library to be installed.

    .. code-block:: python

        from redisvl.utils.rerank import HFCrossEncoderReranker

        # set up the HFCrossEncoderReranker with a specific model
        reranker = HFCrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", limit=3)
        # rerank raw search results based on user input/query
        results = reranker.rank(
            query="your input query text here",
            docs=[
                {"content": "document 1"},
                {"content": "document 2"},
                {"content": "document 3"}
            ]
        )
    """

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        limit: int = 3,
        return_score: bool = True,
        **kwargs,
    ):
        """
        Initialize the HFCrossEncoderReranker with a specified model and ranking criteria.

        Parameters:
            model (str): The name or path of the cross-encoder model to use for reranking.
                Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
            limit (int): The maximum number of results to return after reranking. Must be a positive integer.
            return_score (bool): Whether to return scores alongside the reranked results.
        """
        model = model or kwargs.pop("model_name", None)
        super().__init__(
            model=model, rank_by=None, limit=limit, return_score=return_score
        )
        self._initialize_client(**kwargs)

    def _initialize_client(self, **kwargs):
        """
        Setup the huggingface cross-encoder client using optional kwargs.
        """
        # Dynamic import of the sentence-transformers module
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "HFCrossEncoder reranker requires the sentence-transformers library. \
                    Please install with `pip install sentence-transformers`"
            )

        self._client = CrossEncoder(self.model, **kwargs)

    def rank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Rerank documents based on the provided query using the loaded cross-encoder model.

        This method processes the user's query and the provided documents to rerank them
        in a manner that is potentially more relevant to the query's context.

        Parameters:
            query (str): The user's search query.
            docs (Union[List[Dict[str, Any]], List[str]]): The list of documents to be ranked,
                either as dictionaries or strings.

        Returns:
            Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
                The reranked list of documents and optionally associated scores.
        """
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)

        if not query:
            raise ValueError("query cannot be empty")

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not isinstance(docs, list):
            raise TypeError("docs must be a list")

        if not docs:
            return [] if not return_score else ([], [])

        if all(isinstance(doc, dict) for doc in docs):
            texts = [
                str(doc["content"])
                for doc in docs
                if isinstance(doc, dict) and "content" in doc
            ]
            doc_subset = [
                doc for doc in docs if isinstance(doc, dict) and "content" in doc
            ]
        else:
            texts = [str(doc) for doc in docs]
            doc_subset = [{"content": doc} for doc in docs]

        scores = self._client.predict([(query, text) for text in texts])
        scores = [float(score) for score in scores]
        docs_with_scores = list(zip(doc_subset, scores))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in docs_with_scores[:limit]]
        scores = scores[:limit]

        if return_score:
            return reranked_docs, scores  # type: ignore
        return reranked_docs

    async def arank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Asynchronously rerank documents based on the provided query using the loaded cross-encoder model.

        This method processes the user's query and the provided documents to rerank them
        in a manner that is potentially more relevant to the query's context.

        Parameters:
            query (str): The user's search query.
            docs (Union[List[Dict[str, Any]], List[str]]): The list of documents to be ranked,
                either as dictionaries or strings.

        Returns:
            Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
                The reranked list of documents and optionally associated scores.
        """
        return self.rank(query, docs, **kwargs)

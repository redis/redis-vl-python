import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import PrivateAttr

from redisvl.utils.rerank.base import BaseReranker


class VoyageAIReranker(BaseReranker):
    """
    The VoyageAIReranker class uses VoyageAI's API to rerank documents based on an
    input query.

    This reranker is designed to interact with VoyageAI's /rerank API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `VOYAGE_API_KEY`
    environment variable. User must obtain an API key from VoyageAI's website
    (https://dash.voyageai.com/). Additionally, the `voyageai` python
    client must be installed with `pip install voyageai`.

    .. code-block:: python

        from redisvl.utils.rerank import VoyageAIReranker

        # set up the VoyageAI reranker with some configuration
        reranker = VoyageAIReranker(rank_by=["content"], limit=2)
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
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        rank_by: Optional[List[str]] = None,
        limit: int = 5,
        return_score: bool = True,
        api_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize the VoyageAIReranker with specified model, ranking criteria,
        and API configuration.

        Parameters:
            model (str): The identifier for the VoyageAI model used for reranking.
            rank_by (Optional[List[str]]): Optional list of keys specifying the
                attributes in the documents that should be considered for
                ranking. None means ranking will rely on the model's default
                behavior.
            limit (int): The maximum number of results to return after
                reranking. Must be a positive integer.
            return_score (bool): Whether to return scores alongside the
                reranked results.
            api_config (Optional[Dict], optional): Dictionary containing the API key.
                Defaults to None.

        Raises:
            ImportError: If the voyageai library is not installed.
            ValueError: If the API key is not provided.
        """
        super().__init__(
            model=model, rank_by=rank_by, limit=limit, return_score=return_score
        )
        self._initialize_clients(api_config, **kwargs)

    def _initialize_clients(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the VoyageAI clients using the provided API key or an
        environment variable.
        """
        # Dynamic import of the voyageai module
        try:
            from voyageai import AsyncClient, Client
        except ImportError:
            raise ImportError(
                "VoyageAI reranker requires the voyageai library. \
                    Please install with `pip install voyageai`"
            )

        # Fetch the API key from api_config or environment variable
        api_key = (
            api_config.get("api_key") if api_config else os.getenv("VOYAGE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "VoyageAI API key is required. "
                "Provide it in api_config or set the VOYAGE_API_KEY environment variable."
            )
        self._client = Client(api_key=api_key, **kwargs)
        self._aclient = AsyncClient(api_key=api_key, **kwargs)

    def _preprocess(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ):
        """
        Prepare and validate reranking config based on provided input and
        optional overrides.
        """
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)
        truncation = kwargs.get("truncation")

        if not isinstance(docs, list):
            raise ValueError("Docs to rerank should either be a list of str or dict.")

        if all(isinstance(doc, dict) and "content" in doc for doc in docs):
            texts = [
                str(doc["content"])
                for doc in docs
                if isinstance(doc, dict) and "content" in doc
            ]
        else:
            texts = [str(doc) for doc in docs]

        reranker_kwargs = {
            "query": query,
            "documents": texts,
            "model": self.model,
            "top_k": limit,
        }
        if truncation is not None:
            reranker_kwargs["truncation"] = truncation

        return reranker_kwargs, return_score

    @staticmethod
    def _postprocess(
        docs: Union[List[Dict[str, Any]], List[str]],
        rankings: List[Any],
    ) -> Tuple[List[Any], List[float]]:
        """
        Post-process the initial list of documents to include ranking scores,
        if specified.
        """
        reranked_docs, scores = [], []
        for item in rankings.results:  # type: ignore
            scores.append(item.relevance_score)
            reranked_docs.append(docs[item.index])
        return reranked_docs, scores

    def rank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Rerank documents based on the provided query using the VoyageAI rerank API.

        This method processes the user's query and the provided documents to
        rerank them in a manner that is potentially more relevant to the
        query's context.

        Parameters:
            query (str): The user's search query.
            docs (Union[List[Dict[str, Any]], List[str]]): The list of documents
                to be ranked, either as dictionaries or strings.

        Returns:
            Union[Tuple[Union[List[Dict[str, Any]], List[str]], float], List[Dict[str, Any]]]: The reranked list of documents and optionally associated scores.
        """
        reranker_kwargs, return_score = self._preprocess(query, docs, **kwargs)
        rankings = self._client.rerank(**reranker_kwargs)
        reranked_docs, scores = self._postprocess(docs, rankings)
        if return_score:
            return reranked_docs, scores
        return reranked_docs

    async def arank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Rerank documents based on the provided query using the VoyageAI rerank API.

        This method processes the user's query and the provided documents to
        rerank them in a manner that is potentially more relevant to the
        query's context.

        Parameters:
            query (str): The user's search query.
            docs (Union[List[Dict[str, Any]], List[str]]): The list of documents
                to be ranked, either as dictionaries or strings.

        Returns:
            Union[Tuple[Union[List[Dict[str, Any]], List[str]], float], List[Dict[str, Any]]]: The reranked list of documents and optionally associated scores.
        """
        reranker_kwargs, return_score = self._preprocess(query, docs, **kwargs)
        rankings = await self._aclient.rerank(**reranker_kwargs)
        reranked_docs, scores = self._postprocess(docs, rankings)
        if return_score:
            return reranked_docs, scores
        return reranked_docs

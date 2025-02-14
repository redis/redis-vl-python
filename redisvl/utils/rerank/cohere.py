import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import PrivateAttr

from redisvl.utils.rerank.base import BaseReranker


class CohereReranker(BaseReranker):
    """
    The CohereReranker class uses Cohere's API to rerank documents based on an
    input query.

    This reranker is designed to interact with Cohere's /rerank API,
    requiring an API key for authentication. The key can be provided
    directly in the `api_config` dictionary or through the `COHERE_API_KEY`
    environment variable. User must obtain an API key from Cohere's website
    (https://dashboard.cohere.com/). Additionally, the `cohere` python
    client must be installed with `pip install cohere`.

    .. code-block:: python

        from redisvl.utils.rerank import CohereReranker

        # set up the Cohere reranker with some configuration
        reranker = CohereReranker(rank_by=["content"], limit=2)
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
        model: str = "rerank-english-v3.0",
        rank_by: Optional[List[str]] = None,
        limit: int = 5,
        return_score: bool = True,
        api_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize the CohereReranker with specified model, ranking criteria,
        and API configuration.

        Parameters:
            model (str): The identifier for the Cohere model used for reranking.
                Defaults to 'rerank-english-v3.0'.
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
            ImportError: If the cohere library is not installed.
            ValueError: If the API key is not provided.
        """
        super().__init__(
            model=model, rank_by=rank_by, limit=limit, return_score=return_score
        )
        self._initialize_clients(api_config, **kwargs)

    def _initialize_clients(self, api_config: Optional[Dict], **kwargs):
        """
        Setup the Cohere clients using the provided API key or an
        environment variable.
        """
        # Dynamic import of the cohere module
        try:
            from cohere import AsyncClient, Client
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
        self._client = Client(api_key=api_key, client_name="redisvl", **kwargs)
        self._aclient = AsyncClient(api_key=api_key, client_name="redisvl", **kwargs)

    def _preprocess(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ):
        """
        Prepare and validate reranking config based on provided input and
        optional overrides.
        """
        limit = kwargs.get("limit", self.limit)
        return_score = kwargs.get("return_score", self.return_score)
        max_chunks_per_doc = kwargs.get("max_chunks_per_doc")
        rank_by = kwargs.get("rank_by", self.rank_by) or []
        rank_by = [rank_by] if isinstance(rank_by, str) else rank_by

        reranker_kwargs = {
            "model": self.model,
            "query": query,
            "top_n": limit,
            "documents": docs,
            "max_chunks_per_doc": max_chunks_per_doc,
        }
        # if we are working with list of dicts
        if all(isinstance(doc, dict) for doc in docs):
            if rank_by:
                reranker_kwargs["rank_fields"] = rank_by
            else:
                raise ValueError(
                    "If reranking dictionary-like docs, "
                    "you must provide a list of rank_by fields"
                )

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
        Rerank documents based on the provided query using the Cohere rerank API.

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
        Rerank documents based on the provided query using the Cohere rerank API.

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

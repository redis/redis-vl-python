from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

from pydantic import BaseModel, validator


class BaseReranker(BaseModel, ABC):
    """
    Base class for reranking services that defines the essential
    framework for implementations.

    This class serves as a template for creating specialized reranker services
    that can interact with different machine learning models to rerank a list of
    docs based on a query. It uses abstract methods that must be implemented
    by subclasses to provide concrete behavior.

    Attributes:
        model (str): Identifier for the model used for reranking.
        rank_by (Optional[List[str]], optional): An optional list of keys
            specifying the attributes in the docs that should be considered
            for ranking.
        limit (int): The maximum number of results to return after reranking.
        return_score (bool): Flag indicating whether to return scores
            alongside the reranked results.
    """
    model: str
    rank_by: Optional[List[str]] = None
    limit: int
    return_score: bool

    @validator("limit")
    @classmethod
    def check_limit(cls, value):
        """ Ensures the limit is a positive integer. """
        if value <= 0:
            raise ValueError("Limit must be a positive integer.")
        return value

    @validator("rank_by")
    @classmethod
    def check_rank_by(cls, value):
        """ Ensures that rank_by is a list of strings if provided. """
        if value is not None and (
            not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            )
        ):
            raise ValueError("rank_by must be a list of strings.")
        return value

    @abstractmethod
    def rank(
        self,
        query: str,
        docs: Union[List[Dict[str, Any]], List[str]],
        **kwargs
    ) -> Union[Tuple[Union[List[Dict[str, Any]], List[str]], float], List[Dict[str, Any]]]:
        """
        Synchronously rerank the docs based on the provided query.
        """
        pass

    @abstractmethod
    async def arank(
        self,
        query: str,
        docs: Union[List[Dict[str, Any]], List[str]],
        **kwargs
    ) -> Union[Tuple[Union[List[Dict[str, Any]], List[str]], float], List[Dict[str, Any]]]:
        """
        Asynchronously rerank the docs based on the provided query.
        """
        pass

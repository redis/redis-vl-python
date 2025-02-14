from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, field_validator


class BaseReranker(BaseModel, ABC):
    model: str
    rank_by: Optional[List[str]] = None
    limit: int
    return_score: bool

    @field_validator("limit")
    @classmethod
    def check_limit(cls, value):
        """Ensures the limit is a positive integer."""
        if value <= 0:
            raise ValueError("Limit must be a positive integer.")
        return value

    @field_validator("rank_by")
    @classmethod
    def check_rank_by(cls, value):
        """Ensures that rank_by is a list of strings if provided."""
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError("rank_by must be a list of strings.")
        return value

    @abstractmethod
    def rank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Synchronously rerank the docs based on the provided query.
        """
        pass

    @abstractmethod
    async def arank(
        self, query: str, docs: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> Union[Tuple[List[Dict[str, Any]], List[float]], List[Dict[str, Any]]]:
        """
        Asynchronously rerank the docs based on the provided query.
        """
        pass

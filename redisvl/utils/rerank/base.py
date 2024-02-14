from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, validator


class BaseReranker(BaseModel, ABC):
    model: str
    rank_by: Optional[str] = None
    limit: int = 5
    return_score: bool = True

    @validator("limit")
    @classmethod
    def check_limit(cls, value):
        if value <= 0:
            raise ValueError("limit must be a positive integer")
        return value

    @abstractmethod
    def rank(
        self, query: str, results: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def arank(
        self, query: str, results: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        pass

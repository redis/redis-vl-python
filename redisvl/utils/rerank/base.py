
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic.v1 import BaseModel, validator


class BaseReranker(BaseModel, ABC):
    model: str
    rank_by: Optional[List[str]]
    limit: int
    return_score: bool

    @validator("limit")
    @classmethod
    def check_limit(cls, value):
        if value <= 0:
            raise ValueError("limit must be a positive integer")
        return value

    @abstractmethod
    def rank(
        self, query: str, results: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def arank(
        self, query: str, results: Union[List[Dict[str, Any]], List[str]], **kwargs
    ) -> List[Dict[str, Any]]:
        pass



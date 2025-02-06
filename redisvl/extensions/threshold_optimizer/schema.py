from typing import List

from pydantic import BaseModel


class TestData(BaseModel):
    query: str
    query_match: str | None
    query_embedding: List[float] | bytes

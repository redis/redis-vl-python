from typing import List

from pydantic import BaseModel, Field
from ulid import ULID


class TestData(BaseModel):
    q_id: str = Field(default_factory=lambda: str(ULID()))
    query: str
    query_match: str | None
    response: List[dict] = []

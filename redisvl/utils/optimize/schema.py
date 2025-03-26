from typing import List, Optional

from pydantic import BaseModel, Field
from ulid import ULID


class LabeledData(BaseModel):
    id: str = Field(default_factory=lambda: str(ULID()))
    query: str
    query_match: Optional[str]
    response: List[dict] = []

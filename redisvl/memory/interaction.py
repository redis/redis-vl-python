import json

from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel


class Interaction(BaseModel):
    session_id: str
    prompt: str
    response: str
    timestamp: int = int(datetime.utcnow().timestamp())
    metadata: Optional[Dict[str, str]] = {}

    @property
    def content(self):
        return f"""Prompt: {self.prompt} Response: {self.response}"""

    @classmethod
    def from_dict(cls, d: dict):
        if "metadata" in d:
            d["metadata"] = json.loads(d["metadata"])
        return cls(**d)

    def as_dict(self) -> Dict[str, str]:
        d = self.__dict__
        d["metadata"] = json.dumps(d["metadata"])
        return d

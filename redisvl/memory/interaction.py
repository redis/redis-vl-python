import json

from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, model_validator

from redisvl.utils.utils import hash_input


class Interaction(BaseModel):
    id: str
    session_id: str
    prompt: str
    response: str
    content: str
    timestamp: int = int(datetime.utcnow().timestamp())
    metadata: Optional[Dict[str, str]] = {}

    @model_validator(mode='before')
    def _set_fields(cls, values: dict) -> dict:
        if "content" not in values:
            content =  f"""Prompt: {values["prompt"]} Response: {values["response"]}"""
            values["content"] = content
        if "id" not in values:
            values["id"] = f"""{values["session_id"]}:{hash_input(content)}"""
        return values

    @classmethod
    def from_dict(cls, d: dict):
        if "metadata" in d:
            d["metadata"] = json.loads(d["metadata"])
        return cls(**d)

    def as_dict(self) -> Dict[str, str]:
        d = self.__dict__
        d["metadata"] = json.dumps(d["metadata"])
        return d

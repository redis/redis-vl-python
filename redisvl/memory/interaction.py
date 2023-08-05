import json
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, field_serializer, field_validator, model_validator


class Interaction(BaseModel):
    session_id: str
    prompt: str
    response: str
    content: str
    timestamp: int = int(datetime.utcnow().timestamp())
    metadata: Optional[Dict[str, str]] = {}

    @model_validator(mode="before")
    def set_content(cls, values: dict) -> dict:
        if "content" not in values:
            values[
                "content"
            ] = f"""Prompt: {values["prompt"]} Response: {values["response"]}"""
        return values

    @field_serializer("metadata")
    def serialize_metadata(self, metadata: dict, _info):
        return json.dumps(metadata)

    @field_validator("metadata", mode="before")
    def parse_metadata(cls, metadata: dict, _info):
        if isinstance(metadata, str):
            return json.loads(metadata)
        return metadata

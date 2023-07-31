from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel


class Interaction(BaseModel):
    session_id: str
    prompt: str
    response: str
    timestamp: str = datetime.utcnow()
    metadata: Optional[Dict[str, str]] = {}

    @property
    def content(self):
        return f"""Prompt: {self.prompt} Response: {self.response}"""

    def as_dict(self) -> Dict[str, str]:
        return self.__dict__
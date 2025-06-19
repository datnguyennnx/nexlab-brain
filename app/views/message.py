from pydantic import BaseModel
from ..models.message import MessageRole
from datetime import datetime

class MessageBase(BaseModel):
    content: str
    role: MessageRole

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: int
    conversation_id: int
    created_at: datetime

    class Config:
        from_attributes = True 
from pydantic import BaseModel
from typing import Optional, List
from .message import MessageResponse

class ConversationBase(BaseModel):
    title: Optional[str] = None
    user_id: int

class ConversationCreate(ConversationBase):
    pass

class ConversationUpdate(BaseModel):
    title: str

class ConversationPreviewResponse(ConversationBase):
    id: int

    class Config:
        from_attributes = True

class ConversationResponse(ConversationBase):
    id: int
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True 
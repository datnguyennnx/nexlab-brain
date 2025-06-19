from sqlalchemy import Column, String, Integer, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from .base import BaseModel

class MessageRole(PyEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    __tablename__ = "messages"
    
    content = Column(Text, nullable=False)
    role = Column(Enum(MessageRole, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages") 
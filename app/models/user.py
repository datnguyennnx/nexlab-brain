from sqlalchemy.orm import relationship
from .base import BaseModel

class User(BaseModel):
    __tablename__ = "users"
    
    # Add user fields here, e.g. username, email, etc.
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan") 
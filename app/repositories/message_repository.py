from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from ..models.message import Message, MessageRole
from .base_repository import BaseRepository

class MessageRepository(BaseRepository[Message]):
    def __init__(self, db: Session):
        super().__init__(db)

    async def create(self, content: str, role: MessageRole, conversation_id: int) -> Message:
        message = Message(content=content, role=role, conversation_id=conversation_id)
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get_by_id(self, id: int) -> Message | None:
        result = await self.db.execute(select(Message).filter(Message.id == id))
        return result.scalars().first()

    async def get_by_conversation_id(self, conversation_id: int) -> List[Message]:
        result = await self.db.execute(
            select(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return result.scalars().all() 
from typing import List, Optional
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.future import select
from ..models.conversation import Conversation
from .base_repository import BaseRepository

class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, db: Session):
        super().__init__(db)

    async def create(self, user_id: int, title: Optional[str] = None) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self.db.add(conversation)
        await self.db.flush() # Flush to get the conversation ID
        conversation_id = conversation.id
        await self.db.commit()
        
        # Re-fetch the conversation to eagerly load relationships
        return await self.get_by_id(conversation_id)

    async def get_by_id(self, id: int) -> Optional[Conversation]:
        result = await self.db.execute(
            select(Conversation).options(selectinload(Conversation.messages)).filter(Conversation.id == id)
        )
        return result.scalars().first()

    async def get_by_user_id(self, user_id: int) -> List[Conversation]:
        result = await self.db.execute(
            select(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc())
        )
        return result.scalars().all()

    async def delete(self, id: int) -> None:
        conversation = await self.db.get(Conversation, id)
        if conversation:
            await self.db.delete(conversation)
            await self.db.commit()

    async def update(self, id: int, title: str) -> Optional[Conversation]:
        conversation = await self.db.get(Conversation, id)
        if conversation:
            conversation.title = title
            await self.db.commit()
            await self.db.refresh(conversation)
        return conversation 
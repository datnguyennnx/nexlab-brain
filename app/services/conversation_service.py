from typing import List, Optional
from ..models.conversation import Conversation
from ..repositories.conversation_repository import ConversationRepository
from ..repositories.message_repository import MessageRepository


class ConversationService:
    def __init__(self,
                 conversation_repo: ConversationRepository,
                 message_repo: MessageRepository):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo

    async def create_conversation(self, user_id: int, title: Optional[str] = None) -> Conversation:
        return await self.conversation_repo.create(user_id=user_id, title=title)

    async def get_conversation_with_messages(self, conversation_id: int) -> Optional[Conversation]:
        return await self.conversation_repo.get_by_id(conversation_id)

    async def get_user_conversations(self, user_id: int) -> List[Conversation]:
        return await self.conversation_repo.get_by_user_id(user_id)

    async def delete_conversation(self, conversation_id: int) -> None:
        return await self.conversation_repo.delete(conversation_id)

    async def update_conversation_title(self, conversation_id: int, title: str) -> Optional[Conversation]:
        return await self.conversation_repo.update(conversation_id, title)
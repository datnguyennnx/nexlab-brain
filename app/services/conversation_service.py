from typing import List, Optional, AsyncGenerator
from ..models.conversation import Conversation
from ..repositories.conversation_repository import ConversationRepository
from ..repositories.message_repository import MessageRepository
from .openai_chat_service import OpenAIChatService
from ..models.message import MessageRole
import json

class ConversationService:
    def __init__(self, 
                 conversation_repo: ConversationRepository, 
                 message_repo: MessageRepository,
                 openai_chat_service: OpenAIChatService):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.openai_chat_service = openai_chat_service
    
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

    async def stream_ai_response(self, conversation_id: int) -> AsyncGenerator[str, None]:
        # 1. Get conversation history
        history = await self.message_repo.get_by_conversation_id(conversation_id)
        
        # 2. Format for OpenAI
        openai_messages = [{"role": msg.role.value, "content": msg.content} for msg in history]
        
        # 3. Stream from OpenAI and save the final response
        final_content = ""
        try:
            async for chunk in self.openai_chat_service.stream_chat_completion(openai_messages):
                yield chunk
                if chunk.strip().startswith('data:'):
                    try:
                        data_str = chunk.strip()[5:]
                        event_data = json.loads(data_str)
                        if event_data.get('type') == 'content_chunk':
                            final_content += event_data.get('data', {}).get('content', '')
                    except json.JSONDecodeError:
                        pass # Ignore malformed SSE data
        finally:
            if final_content:
                await self.message_repo.create(
                    content=final_content,
                    role=MessageRole.ASSISTANT,
                    conversation_id=conversation_id
                ) 
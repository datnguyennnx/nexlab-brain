from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.session import get_db
from ..repositories.conversation_repository import ConversationRepository
from ..repositories.message_repository import MessageRepository
from ..services.conversation_service import ConversationService
from ..services.rag_service import RAGService
from ..services.openai_chat_service import OpenAIChatService
from ..services.orchestrator_service import OrchestratorService

# Repositories
def get_conversation_repository(db: AsyncSession = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(db)

def get_message_repository(db: AsyncSession = Depends(get_db)) -> MessageRepository:
    return MessageRepository(db)

# Services (singletons or request-scoped)
def get_openai_chat_service() -> OpenAIChatService:
    return OpenAIChatService()

def get_conversation_service(
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository)
) -> ConversationService:
    return ConversationService(conversation_repo, message_repo)

def get_rag_service(
    message_repo: MessageRepository = Depends(get_message_repository),
    db: AsyncSession = Depends(get_db)
) -> RAGService:
    return RAGService(message_repo, db)

def get_orchestrator_service(
    rag_service: RAGService = Depends(get_rag_service),
    chat_service: OpenAIChatService = Depends(get_openai_chat_service),
    message_repo: MessageRepository = Depends(get_message_repository)
) -> OrchestratorService:
    return OrchestratorService(rag_service, chat_service, message_repo) 